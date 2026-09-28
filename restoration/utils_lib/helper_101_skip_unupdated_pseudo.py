import os

import torch
from torch.amp import autocast
from tqdm import tqdm

import utils_lib.helper as base_helper


consistency_weight_by_step = base_helper.consistency_weight_by_step
create_fgresq_metric = base_helper.create_fgresq_metric
freeze_teachers_parameters = base_helper.freeze_teachers_parameters
get_fgresq_score = base_helper.get_fgresq_score
get_musiq_score = base_helper.get_musiq_score
normalize_mode_name = base_helper.normalize_mode_name
parse_round_metric_schedule = base_helper.parse_round_metric_schedule
select_round_mode = base_helper.select_round_mode
setup_ddp = base_helper.setup_ddp
setup_runtime = base_helper.setup_runtime
save_image_tensor = base_helper.save_image_tensor


UPDATED_MARKER_SUFFIX = '.updated'


def _marker_path(pseudo_path):
    return f'{pseudo_path}{UPDATED_MARKER_SUFFIX}'


def _clear_update_marker(pseudo_path):
    marker_path = _marker_path(pseudo_path)
    if os.path.exists(marker_path):
        os.remove(marker_path)


def _write_update_marker(pseudo_path):
    marker_path = _marker_path(pseudo_path)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, 'w', encoding='ascii') as handle:
        handle.write('updated\n')


def _resolve_updated_mask(pseudo_names, count, device):
    if pseudo_names is None:
        return torch.ones(count, device=device, dtype=torch.bool)
    states = []
    for pseudo_path in pseudo_names:
        states.append(os.path.exists(pseudo_path) and os.path.exists(_marker_path(pseudo_path)))
    if len(states) != count:
        raise ValueError(f'Updated-marker size mismatch: {len(states)} vs {count}')
    return torch.tensor(states, device=device, dtype=torch.bool)


@torch.no_grad()
def initialize_pseudo_labels(
    teacher_model,
    dino_model,
    cgcd_model,
    loader,
    device,
    dino_transform,
    amp_dtype=torch.float16,
    use_class_routing=False,
    rank=0,
    show_progress=True,
):
    teacher_model.eval()
    dino_model.eval()
    cgcd_model.eval()

    if show_progress:
        desc = 'Initializing Pseudo Labels with Teacher'
        if rank != 0:
            desc = f'{desc} (rank{rank})'
        iterator = tqdm(loader, desc=desc)
    else:
        iterator = loader

    for _, batch in enumerate(iterator):
        lq_unlabeled = batch[0].to(device, non_blocking=True)
        pseudo_paths = batch[2]

        with autocast('cuda', dtype=amp_dtype):
            dino_feat = dino_model(dino_transform(lq_unlabeled)).pooler_output
            embedding, logits = cgcd_model(dino_feat)

            if use_class_routing:
                class_ids = torch.argmax(logits, dim=1)
                restored_teacher = teacher_model(lq_unlabeled, embedding, class_ids=class_ids)
            else:
                restored_teacher = teacher_model(lq_unlabeled, embedding)

        for idx, path in enumerate(pseudo_paths):
            save_image_tensor(restored_teacher[idx], path)
            _clear_update_marker(path)

    if show_progress:
        print('[INFO] Initial pseudo labeling completed (markers cleared).')


@torch.no_grad()
def copy_pseudo_labels(loader, rank=0, show_progress=True):
    if show_progress:
        desc = 'Copying Inputs to Pseudo Bank'
        if rank != 0:
            desc = f'{desc} (rank{rank})'
        iterator = tqdm(loader, desc=desc)
    else:
        iterator = loader

    copied = 0
    for _, batch in enumerate(iterator):
        lq_unlabeled = batch[0]
        pseudo_paths = batch[2]
        for idx, path in enumerate(pseudo_paths):
            save_image_tensor(lq_unlabeled[idx], path)
            _clear_update_marker(path)
            copied += 1

    if show_progress:
        print(f'[INFO] Pseudo label copy completed ({copied} files, markers cleared).')


@torch.no_grad()
def get_reliable_clip_iqa(
    iqa_metric,
    teacher_predict,
    student_predict,
    pseudo_list,
    pseudo_names,
    rank,
    update_margin=0.0,
    return_stats=False,
):
    teacher_predict = torch.clamp(teacher_predict, 0.0, 1.0)
    student_predict = torch.clamp(student_predict, 0.0, 1.0)
    pseudo_list = torch.clamp(pseudo_list, 0.0, 1.0)

    score_teacher = iqa_metric(teacher_predict).view(-1).float()
    score_student = iqa_metric(student_predict).view(-1).float()
    score_reference = iqa_metric(pseudo_list).view(-1).float()

    final_pseudo_labels = pseudo_list.clone()
    updated_mask = _resolve_updated_mask(pseudo_names, teacher_predict.shape[0], teacher_predict.device)
    active_mask_current = updated_mask.clone()
    inactive_mask = ~updated_mask
    if inactive_mask.any():
        final_pseudo_labels[inactive_mask] = student_predict.detach()[inactive_mask]

    updated_cnt = 0
    margin = float(update_margin)
    stats = base_helper._summarize_pseudo_update_delta(score_teacher, score_student, score_reference, margin)
    for idx in range(teacher_predict.shape[0]):
        threshold = torch.maximum(score_student[idx], score_reference[idx]) + margin
        if score_teacher[idx] > threshold:
            final_pseudo_labels[idx] = teacher_predict[idx]
            updated_cnt += 1
            active_mask_current[idx] = True
            save_image_tensor(teacher_predict[idx], pseudo_names[idx])
            _write_update_marker(pseudo_names[idx])

    stats['active_count_before'] = int(updated_mask.sum().item())
    stats['active_count_after'] = int(active_mask_current.sum().item())
    stats['inactive_count_before'] = int(inactive_mask.sum().item())
    stats['inactive_count_after'] = int((~active_mask_current).sum().item())
    stats['active_mask_after'] = active_mask_current.detach().clone()

    if return_stats:
        return final_pseudo_labels, updated_cnt, 'CLIP-IQA', stats
    return final_pseudo_labels, updated_cnt, 'CLIP-IQA'


@torch.no_grad()
def get_reliable(
    iqa_metric,
    teacher_predict,
    student_predict,
    score_reference,
    pseudo_list,
    pseudo_names,
    rank,
    pseudo_update_mode='musiq',
    dino_model=None,
    cgcd_model=None,
    dino_transform=None,
    epoch=0,
    orig2classifier=None,
    cgcd_score_mode='clear',
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
    mahalanobis_temp=20.0,
    mahalanobis_pca_top_k=32,
    update_margin=0.0,
    return_stats=False,
):
    total_candidates = teacher_predict.shape[0]
    updated_indices = []
    final_pseudo_labels = pseudo_list.clone()

    if pseudo_update_mode == 'musiq':
        use_musiq = True
        use_fgresq = False
    elif pseudo_update_mode == 'fgresq':
        use_musiq = True
        use_fgresq = True
    elif pseudo_update_mode == 'cgcd':
        use_musiq = False
        use_fgresq = False
    elif pseudo_update_mode == 'alternate':
        use_musiq = (epoch % 2 == 0) or (dino_model is None) or (cgcd_model is None)
        use_fgresq = False
    else:
        raise ValueError(f'Unsupported pseudo_update_mode: {pseudo_update_mode}')

    if use_musiq:
        if use_fgresq:
            score_teacher = get_fgresq_score(iqa_metric, teacher_predict)
            score_student = get_fgresq_score(iqa_metric, student_predict)
            if score_reference is None:
                score_reference = get_fgresq_score(iqa_metric, pseudo_list)
            update_mode = 'FGResQ'
        else:
            score_teacher = get_musiq_score(iqa_metric, teacher_predict)
            score_student = get_musiq_score(iqa_metric, student_predict)
            if score_reference is None:
                score_reference = get_musiq_score(iqa_metric, pseudo_list)
            update_mode = 'MUSIQ'
    else:
        clear_idx = orig2classifier[0] if orig2classifier is not None else 0
        ver2_modes = {'mahalanobis_margin', 'mahalanobis_pca'}
        score_fn = base_helper.get_cgcd_score_ver2 if cgcd_score_mode in ver2_modes else base_helper.get_cgcd_score
        score_kwargs = dict(
            mode=cgcd_score_mode,
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
        )
        if score_fn is base_helper.get_cgcd_score_ver2:
            score_kwargs['mahalanobis_temp'] = mahalanobis_temp
            score_kwargs['mahalanobis_pca_top_k'] = mahalanobis_pca_top_k

        score_teacher = score_fn(teacher_predict, dino_model, cgcd_model, dino_transform, clear_idx, **score_kwargs)
        score_student = score_fn(student_predict, dino_model, cgcd_model, dino_transform, clear_idx, **score_kwargs)
        score_reference = score_fn(pseudo_list, dino_model, cgcd_model, dino_transform, clear_idx, **score_kwargs)
        update_mode = f'CGCD_{cgcd_score_mode}'

    updated_mask = _resolve_updated_mask(pseudo_names, total_candidates, teacher_predict.device)
    active_mask_current = updated_mask.clone()
    inactive_mask = ~updated_mask
    if inactive_mask.any():
        final_pseudo_labels[inactive_mask] = student_predict.detach()[inactive_mask]

    margin = float(update_margin)
    stats = base_helper._summarize_pseudo_update_delta(score_teacher, score_student, score_reference, margin)
    for idx in range(total_candidates):
        threshold = torch.maximum(score_student[idx], score_reference[idx]) + margin
        if score_teacher[idx] > threshold:
            final_pseudo_labels[idx] = teacher_predict[idx]
            updated_indices.append(idx)
            active_mask_current[idx] = True
            save_image_tensor(teacher_predict[idx], pseudo_names[idx])
            _write_update_marker(pseudo_names[idx])

    stats['active_count_before'] = int(updated_mask.sum().item())
    stats['active_count_after'] = int(active_mask_current.sum().item())
    stats['inactive_count_before'] = int(inactive_mask.sum().item())
    stats['inactive_count_after'] = int((~active_mask_current).sum().item())
    stats['active_mask_after'] = active_mask_current.detach().clone()

    if return_stats:
        return final_pseudo_labels, len(updated_indices), update_mode, stats
    return final_pseudo_labels, len(updated_indices), update_mode
