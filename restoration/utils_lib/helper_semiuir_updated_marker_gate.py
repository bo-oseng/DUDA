import os

import torch
from torch.amp import autocast

import utils_lib.helper as base_helper
import utils_lib.helper_101_skip_unupdated_pseudo as base_skip
import utils_lib.helper_semiuir_score as semiuir_score_helper


UPDATED_MARKER_SUFFIX = base_skip.UPDATED_MARKER_SUFFIX


def _iter_loader(loader, rank, show_progress, desc):
    if not show_progress:
        return loader

    from tqdm import tqdm

    if rank != 0:
        desc = f"{desc} (rank{rank})"
    return tqdm(loader, desc=desc)


def _clear_markers_for_paths(pseudo_paths):
    cleared = 0
    for path in pseudo_paths:
        marker_path = base_skip._marker_path(path)
        if os.path.exists(marker_path):
            base_skip._clear_update_marker(path)
            cleared += 1
    return cleared


def _compute_scores(
    iqa_metric,
    teacher_predict,
    student_predict,
    score_reference,
    pseudo_list,
    pseudo_update_mode,
    dino_model,
    cgcd_model,
    dino_transform,
    epoch,
    orig2classifier,
    amp_dtype,
    cgcd_anchor_clear_weight,
    cgcd_anchor_neg_weight,
    cgcd_anchor_softmin_tau,
):
    return semiuir_score_helper.compute_reliable_score_triplet(
        iqa_metric=iqa_metric,
        teacher_predict=teacher_predict,
        student_predict=student_predict,
        score_reference=score_reference,
        pseudo_list=pseudo_list,
        pseudo_update_mode=pseudo_update_mode,
        dino_model=dino_model,
        cgcd_model=cgcd_model,
        dino_transform=dino_transform,
        epoch=epoch,
        orig2classifier=orig2classifier,
        amp_dtype=amp_dtype,
        clear_weight=cgcd_anchor_clear_weight,
        neg_weight=cgcd_anchor_neg_weight,
        softmin_tau=cgcd_anchor_softmin_tau,
    )


@torch.no_grad()
def initialize_pseudo_labels_semiuir(
    teacher_model,
    dino_model,
    cgcd_model,
    loader,
    device,
    dino_transform,
    amp_dtype=torch.float16,
    rank=0,
    show_progress=True,
):
    teacher_model.eval()
    dino_model.eval()
    cgcd_model.eval()

    iterator = _iter_loader(loader, rank, show_progress, "Initializing Pseudo Labels with Teacher")
    cleared = 0
    for batch in iterator:
        weak_unlabeled = batch[0].to(device, non_blocking=True)
        pseudo_paths = batch[3]

        with autocast("cuda", dtype=amp_dtype):
            dino_feat = dino_model(dino_transform(weak_unlabeled)).pooler_output
            embedding, _ = cgcd_model(dino_feat)
            restored_teacher = teacher_model(weak_unlabeled, embedding)

        for idx, path in enumerate(pseudo_paths):
            base_skip.save_image_tensor(restored_teacher[idx], path)
        cleared += _clear_markers_for_paths(pseudo_paths)

    if show_progress:
        print(f"✓ Initial Pseudo Labeling Completed! (markers_cleared={cleared})")


@torch.no_grad()
def copy_pseudo_labels_semiuir(loader, rank=0, show_progress=True):
    iterator = _iter_loader(loader, rank, show_progress, "Copying Inputs to Pseudo Bank")
    copied = 0
    cleared = 0
    for batch in iterator:
        weak_unlabeled = batch[0]
        pseudo_paths = batch[3]
        for idx, path in enumerate(pseudo_paths):
            base_skip.save_image_tensor(weak_unlabeled[idx], path)
            copied += 1
        cleared += _clear_markers_for_paths(pseudo_paths)

    if show_progress:
        print(f"✓ Pseudo Label Copy Completed! (copied={copied}, markers_cleared={cleared})")


@torch.no_grad()
def initialize_zero_pseudo_labels_semiuir(loader, rank=0, show_progress=True):
    iterator = _iter_loader(loader, rank, show_progress, "Initializing Zero Pseudo Slots")
    created = 0
    skipped = 0
    cleared = 0
    for batch in iterator:
        weak_unlabeled = batch[0]
        pseudo_paths = batch[3]
        cleared += _clear_markers_for_paths(pseudo_paths)
        for idx, path in enumerate(pseudo_paths):
            if os.path.exists(path):
                skipped += 1
                continue
            zero_slot = torch.zeros_like(weak_unlabeled[idx])
            base_skip.save_image_tensor(zero_slot, path)
            created += 1

    if show_progress:
        print(
            "✓ Zero Pseudo Slot Init Completed! "
            f"(created={created}, skipped={skipped}, markers_cleared={cleared})"
        )



@torch.no_grad()
def _prepare_reliable_tensors(teacher_predict, student_predict, pseudo_list):
    teacher_predict = torch.clamp(teacher_predict, 0.0, 1.0)
    student_predict = torch.clamp(student_predict, 0.0, 1.0)
    pseudo_list = torch.clamp(pseudo_list, 0.0, 1.0)
    return teacher_predict, student_predict, pseudo_list


@torch.no_grad()
def _init_updated_gate_state(teacher_predict, student_predict, pseudo_list, pseudo_names):
    total_candidates = teacher_predict.shape[0]
    final_pseudo_labels = pseudo_list.clone()
    updated_mask = base_skip._resolve_updated_mask(pseudo_names, total_candidates, teacher_predict.device)
    active_mask_current = updated_mask.clone()
    inactive_mask = ~updated_mask
    if inactive_mask.any():
        final_pseudo_labels[inactive_mask] = student_predict.detach()[inactive_mask]
    return final_pseudo_labels, updated_mask, active_mask_current, inactive_mask


@torch.no_grad()
def _apply_teacher_updates_with_markers(
    final_pseudo_labels,
    teacher_predict,
    pseudo_names,
    accept_mask,
    updated_indices,
    active_mask_current,
):
    update_count = 0
    for idx in torch.nonzero(accept_mask, as_tuple=False).view(-1).tolist():
        final_pseudo_labels[idx] = teacher_predict[idx]
        updated_indices.append(idx)
        update_count += 1
        active_mask_current[idx] = True
        base_skip.save_image_tensor(teacher_predict[idx], pseudo_names[idx])
        base_skip._write_update_marker(pseudo_names[idx])
    return update_count


@torch.no_grad()
def _finalize_gate_stats(stats, updated_mask, inactive_mask, active_mask_current):
    stats["active_count_before"] = int(updated_mask.sum().item())
    stats["active_count_after"] = int(active_mask_current.sum().item())
    stats["inactive_count_before"] = int(inactive_mask.sum().item())
    stats["inactive_count_after"] = int((~active_mask_current).sum().item())
    stats["active_mask_after"] = active_mask_current.detach().clone()
    return stats


@torch.no_grad()
def get_reliable_semiuir_musiq(
    iqa_metric,
    teacher_predict,
    student_predict,
    score_reference,
    pseudo_list,
    pseudo_names,
    rank,
    pseudo_update_mode="musiq",
    dino_model=None,
    cgcd_model=None,
    dino_transform=None,
    epoch=0,
    orig2classifier=None,
    amp_dtype=None,
    cgcd_anchor_clear_weight=1.0,
    cgcd_anchor_neg_weight=1.0,
    cgcd_anchor_softmin_tau=0.0,
    update_margin=0.0,
    fallback_update_margin=0.0,
    return_stats=False,
):
    if iqa_metric is None:
        raise ValueError("MUSIQ metric is required for MUSIQ pseudo update mode.")

    teacher_predict, student_predict, pseudo_list = _prepare_reliable_tensors(
        teacher_predict, student_predict, pseudo_list
    )
    final_pseudo_labels, updated_mask, active_mask_current, inactive_mask = _init_updated_gate_state(
        teacher_predict, student_predict, pseudo_list, pseudo_names
    )
    updated_indices = []

    score_teacher_musiq = base_helper.get_musiq_score(iqa_metric, teacher_predict)
    score_student_musiq = base_helper.get_musiq_score(iqa_metric, student_predict)
    if score_reference is None:
        score_reference_musiq = base_helper.get_musiq_score(iqa_metric, pseudo_list)
    else:
        score_reference_musiq = score_reference.detach().float().view(-1)

    effective_reference_score_musiq = torch.where(updated_mask, score_reference_musiq, score_student_musiq)
    musiq_margin = float(update_margin)
    stats = base_helper._summarize_pseudo_update_delta(
        score_teacher_musiq,
        score_student_musiq,
        effective_reference_score_musiq,
        musiq_margin,
    )
    musiq_accept_mask = score_teacher_musiq > (
        torch.maximum(score_student_musiq, effective_reference_score_musiq) + musiq_margin
    )
    _apply_teacher_updates_with_markers(
        final_pseudo_labels,
        teacher_predict,
        pseudo_names,
        musiq_accept_mask,
        updated_indices,
        active_mask_current,
    )
    stats = _finalize_gate_stats(stats, updated_mask, inactive_mask, active_mask_current)

    if return_stats:
        return final_pseudo_labels, len(updated_indices), "MUSIQ", stats
    return final_pseudo_labels, len(updated_indices), "MUSIQ"


@torch.no_grad()
def get_reliable_semiuir_musiq_then_cgcd(
    iqa_metric,
    teacher_predict,
    student_predict,
    score_reference,
    pseudo_list,
    pseudo_names,
    rank,
    pseudo_update_mode="musiq_then_cgcd",
    dino_model=None,
    cgcd_model=None,
    dino_transform=None,
    epoch=0,
    orig2classifier=None,
    amp_dtype=None,
    cgcd_anchor_clear_weight=1.0,
    cgcd_anchor_neg_weight=1.0,
    cgcd_anchor_softmin_tau=0.0,
    update_margin=0.0,
    fallback_update_margin=0.0,
    return_stats=False,
):
    if iqa_metric is None:
        raise ValueError("MUSIQ metric is required for musiq_then_cgcd mode.")

    teacher_predict, student_predict, pseudo_list = _prepare_reliable_tensors(
        teacher_predict, student_predict, pseudo_list
    )
    final_pseudo_labels, updated_mask, active_mask_current, inactive_mask = _init_updated_gate_state(
        teacher_predict, student_predict, pseudo_list, pseudo_names
    )
    updated_indices = []

    score_teacher_musiq = base_helper.get_musiq_score(iqa_metric, teacher_predict)
    score_student_musiq = base_helper.get_musiq_score(iqa_metric, student_predict)
    if score_reference is None:
        score_reference_musiq = base_helper.get_musiq_score(iqa_metric, pseudo_list)
    else:
        score_reference_musiq = score_reference.detach().float().view(-1)

    effective_reference_score_musiq = torch.where(updated_mask, score_reference_musiq, score_student_musiq)
    musiq_margin = float(update_margin)
    musiq_stats = base_helper._summarize_pseudo_update_delta(
        score_teacher_musiq,
        score_student_musiq,
        effective_reference_score_musiq,
        musiq_margin,
    )
    musiq_accept_mask = score_teacher_musiq > (
        torch.maximum(score_student_musiq, effective_reference_score_musiq) + musiq_margin
    )
    musiq_update_count = _apply_teacher_updates_with_markers(
        final_pseudo_labels,
        teacher_predict,
        pseudo_names,
        musiq_accept_mask,
        updated_indices,
        active_mask_current,
    )

    fallback_candidate_mask = ~musiq_accept_mask
    fallback_margin = float(fallback_update_margin)
    cgcd_update_count = 0
    if bool(fallback_candidate_mask.any().item()):
        score_teacher_cgcd, score_student_cgcd, score_reference_cgcd, _ = semiuir_score_helper._compute_cgcd_anchor_scores(
            teacher_predict=teacher_predict,
            student_predict=student_predict,
            pseudo_list=pseudo_list,
            dino_model=dino_model,
            cgcd_model=cgcd_model,
            dino_transform=dino_transform,
            orig2classifier=orig2classifier,
            amp_dtype=amp_dtype,
            clear_weight=cgcd_anchor_clear_weight,
            neg_weight=cgcd_anchor_neg_weight,
            softmin_tau=cgcd_anchor_softmin_tau,
        )
        effective_reference_score_cgcd = torch.where(updated_mask, score_reference_cgcd, score_student_cgcd)
        fallback_stats = base_helper._summarize_pseudo_update_delta(
            score_teacher_cgcd[fallback_candidate_mask],
            score_student_cgcd[fallback_candidate_mask],
            effective_reference_score_cgcd[fallback_candidate_mask],
            fallback_margin,
        )
        cgcd_accept_mask = fallback_candidate_mask & (
            score_teacher_cgcd > (torch.maximum(score_student_cgcd, effective_reference_score_cgcd) + fallback_margin)
        )
        cgcd_update_count = _apply_teacher_updates_with_markers(
            final_pseudo_labels,
            teacher_predict,
            pseudo_names,
            cgcd_accept_mask,
            updated_indices,
            active_mask_current,
        )
    else:
        fallback_stats = {
            "count": 0,
            "update_count": 0,
            "delta_raw_mean": 0.0,
            "delta_update_mean": 0.0,
        }

    stats = {
        "margin": musiq_margin,
        "fallback_margin": fallback_margin,
        "update_count_musiq": int(musiq_update_count),
        "update_count_cgcd": int(cgcd_update_count),
        "fallback_candidate_count": int(fallback_candidate_mask.sum().item()),
        "delta_raw_musiq_mean": float(musiq_stats.get("delta_raw_mean", 0.0)),
        "delta_update_musiq_mean": float(musiq_stats.get("delta_update_mean", 0.0)),
        "delta_raw_cgcd_mean": float(fallback_stats.get("delta_raw_mean", 0.0)),
        "delta_update_cgcd_mean": float(fallback_stats.get("delta_update_mean", 0.0)),
    }
    stats = _finalize_gate_stats(stats, updated_mask, inactive_mask, active_mask_current)

    if return_stats:
        return final_pseudo_labels, len(updated_indices), "MUSIQ->CGCD_anchor", stats
    return final_pseudo_labels, len(updated_indices), "MUSIQ->CGCD_anchor"


@torch.no_grad()
def get_reliable_semiuir_fgresq_then_cgcd(
    iqa_metric,
    teacher_predict,
    student_predict,
    score_reference,
    pseudo_list,
    pseudo_names,
    rank,
    pseudo_update_mode="fgresq_then_cgcd",
    dino_model=None,
    cgcd_model=None,
    dino_transform=None,
    epoch=0,
    orig2classifier=None,
    amp_dtype=None,
    cgcd_anchor_clear_weight=1.0,
    cgcd_anchor_neg_weight=1.0,
    cgcd_anchor_softmin_tau=0.0,
    update_margin=0.0,
    fallback_update_margin=0.0,
    return_stats=False,
):
    if iqa_metric is None:
        raise ValueError("FGResQ metric is required for fgresq_then_cgcd mode.")

    teacher_predict, student_predict, pseudo_list = _prepare_reliable_tensors(
        teacher_predict, student_predict, pseudo_list
    )
    final_pseudo_labels, updated_mask, active_mask_current, inactive_mask = _init_updated_gate_state(
        teacher_predict, student_predict, pseudo_list, pseudo_names
    )
    updated_indices = []

    score_teacher_fgresq = base_helper.get_fgresq_score(iqa_metric, teacher_predict)
    score_student_fgresq = base_helper.get_fgresq_score(iqa_metric, student_predict)
    if score_reference is None:
        score_reference_fgresq = base_helper.get_fgresq_score(iqa_metric, pseudo_list)
    else:
        score_reference_fgresq = score_reference.detach().float().view(-1)

    effective_reference_score_fgresq = torch.where(updated_mask, score_reference_fgresq, score_student_fgresq)
    fgresq_margin = float(update_margin)
    fgresq_stats = base_helper._summarize_pseudo_update_delta(
        score_teacher_fgresq,
        score_student_fgresq,
        effective_reference_score_fgresq,
        fgresq_margin,
    )
    fgresq_accept_mask = score_teacher_fgresq > (
        torch.maximum(score_student_fgresq, effective_reference_score_fgresq) + fgresq_margin
    )
    fgresq_update_count = _apply_teacher_updates_with_markers(
        final_pseudo_labels,
        teacher_predict,
        pseudo_names,
        fgresq_accept_mask,
        updated_indices,
        active_mask_current,
    )

    fallback_candidate_mask = ~fgresq_accept_mask
    fallback_margin = float(fallback_update_margin)
    cgcd_update_count = 0
    if bool(fallback_candidate_mask.any().item()):
        score_teacher_cgcd, score_student_cgcd, score_reference_cgcd, _ = semiuir_score_helper._compute_cgcd_anchor_scores(
            teacher_predict=teacher_predict,
            student_predict=student_predict,
            pseudo_list=pseudo_list,
            dino_model=dino_model,
            cgcd_model=cgcd_model,
            dino_transform=dino_transform,
            orig2classifier=orig2classifier,
            amp_dtype=amp_dtype,
            clear_weight=cgcd_anchor_clear_weight,
            neg_weight=cgcd_anchor_neg_weight,
            softmin_tau=cgcd_anchor_softmin_tau,
        )
        effective_reference_score_cgcd = torch.where(updated_mask, score_reference_cgcd, score_student_cgcd)
        fallback_stats = base_helper._summarize_pseudo_update_delta(
            score_teacher_cgcd[fallback_candidate_mask],
            score_student_cgcd[fallback_candidate_mask],
            effective_reference_score_cgcd[fallback_candidate_mask],
            fallback_margin,
        )
        cgcd_accept_mask = fallback_candidate_mask & (
            score_teacher_cgcd > (torch.maximum(score_student_cgcd, effective_reference_score_cgcd) + fallback_margin)
        )
        cgcd_update_count = _apply_teacher_updates_with_markers(
            final_pseudo_labels,
            teacher_predict,
            pseudo_names,
            cgcd_accept_mask,
            updated_indices,
            active_mask_current,
        )
    else:
        fallback_stats = {
            "count": 0,
            "update_count": 0,
            "delta_raw_mean": 0.0,
            "delta_update_mean": 0.0,
        }

    stats = {
        "margin": fgresq_margin,
        "fallback_margin": fallback_margin,
        "update_count_fgresq": int(fgresq_update_count),
        "update_count_cgcd": int(cgcd_update_count),
        "fallback_candidate_count": int(fallback_candidate_mask.sum().item()),
        "delta_raw_fgresq_mean": float(fgresq_stats.get("delta_raw_mean", 0.0)),
        "delta_update_fgresq_mean": float(fgresq_stats.get("delta_update_mean", 0.0)),
        "delta_raw_cgcd_mean": float(fallback_stats.get("delta_raw_mean", 0.0)),
        "delta_update_cgcd_mean": float(fallback_stats.get("delta_update_mean", 0.0)),
    }
    stats = _finalize_gate_stats(stats, updated_mask, inactive_mask, active_mask_current)

    if return_stats:
        return final_pseudo_labels, len(updated_indices), "FGResQ->CGCD_anchor", stats
    return final_pseudo_labels, len(updated_indices), "FGResQ->CGCD_anchor"


@torch.no_grad()
def get_reliable_semiuir(
    iqa_metric,
    teacher_predict,
    student_predict,
    score_reference,
    pseudo_list,
    pseudo_names,
    rank,
    pseudo_update_mode="musiq",
    dino_model=None,
    cgcd_model=None,
    dino_transform=None,
    epoch=0,
    orig2classifier=None,
    amp_dtype=None,
    cgcd_anchor_clear_weight=1.0,
    cgcd_anchor_neg_weight=1.0,
    cgcd_anchor_softmin_tau=0.0,
    update_margin=0.0,
    fallback_update_margin=0.0,
    return_stats=False,
):
    normalized_mode = semiuir_score_helper.normalize_mode_name(pseudo_update_mode)
    if normalized_mode == "musiq":
        return get_reliable_semiuir_musiq(
            iqa_metric=iqa_metric,
            teacher_predict=teacher_predict,
            student_predict=student_predict,
            score_reference=score_reference,
            pseudo_list=pseudo_list,
            pseudo_names=pseudo_names,
            rank=rank,
            pseudo_update_mode=normalized_mode,
            dino_model=dino_model,
            cgcd_model=cgcd_model,
            dino_transform=dino_transform,
            epoch=epoch,
            orig2classifier=orig2classifier,
            amp_dtype=amp_dtype,
            cgcd_anchor_clear_weight=cgcd_anchor_clear_weight,
            cgcd_anchor_neg_weight=cgcd_anchor_neg_weight,
            cgcd_anchor_softmin_tau=cgcd_anchor_softmin_tau,
            update_margin=update_margin,
            fallback_update_margin=fallback_update_margin,
            return_stats=return_stats,
        )
    if normalized_mode == semiuir_score_helper.FGRESQ_HYBRID_UPDATE_MODE:
        return get_reliable_semiuir_fgresq_then_cgcd(
            iqa_metric=iqa_metric,
            teacher_predict=teacher_predict,
            student_predict=student_predict,
            score_reference=score_reference,
            pseudo_list=pseudo_list,
            pseudo_names=pseudo_names,
            rank=rank,
            pseudo_update_mode=normalized_mode,
            dino_model=dino_model,
            cgcd_model=cgcd_model,
            dino_transform=dino_transform,
            epoch=epoch,
            orig2classifier=orig2classifier,
            amp_dtype=amp_dtype,
            cgcd_anchor_clear_weight=cgcd_anchor_clear_weight,
            cgcd_anchor_neg_weight=cgcd_anchor_neg_weight,
            cgcd_anchor_softmin_tau=cgcd_anchor_softmin_tau,
            update_margin=update_margin,
            fallback_update_margin=fallback_update_margin,
            return_stats=return_stats,
        )

    if normalized_mode == semiuir_score_helper.HYBRID_UPDATE_MODE:
        return get_reliable_semiuir_musiq_then_cgcd(
            iqa_metric=iqa_metric,
            teacher_predict=teacher_predict,
            student_predict=student_predict,
            score_reference=score_reference,
            pseudo_list=pseudo_list,
            pseudo_names=pseudo_names,
            rank=rank,
            pseudo_update_mode=normalized_mode,
            dino_model=dino_model,
            cgcd_model=cgcd_model,
            dino_transform=dino_transform,
            epoch=epoch,
            orig2classifier=orig2classifier,
            amp_dtype=amp_dtype,
            cgcd_anchor_clear_weight=cgcd_anchor_clear_weight,
            cgcd_anchor_neg_weight=cgcd_anchor_neg_weight,
            cgcd_anchor_softmin_tau=cgcd_anchor_softmin_tau,
            update_margin=update_margin,
            fallback_update_margin=fallback_update_margin,
            return_stats=return_stats,
        )

    teacher_predict, student_predict, pseudo_list = _prepare_reliable_tensors(
        teacher_predict, student_predict, pseudo_list
    )
    final_pseudo_labels, updated_mask, active_mask_current, inactive_mask = _init_updated_gate_state(
        teacher_predict, student_predict, pseudo_list, pseudo_names
    )
    updated_indices = []

    score_teacher, score_student, score_reference, update_mode = _compute_scores(
        iqa_metric=iqa_metric,
        teacher_predict=teacher_predict,
        student_predict=student_predict,
        score_reference=score_reference,
        pseudo_list=pseudo_list,
        pseudo_update_mode=normalized_mode,
        dino_model=dino_model,
        cgcd_model=cgcd_model,
        dino_transform=dino_transform,
        epoch=epoch,
        orig2classifier=orig2classifier,
        amp_dtype=amp_dtype,
        cgcd_anchor_clear_weight=cgcd_anchor_clear_weight,
        cgcd_anchor_neg_weight=cgcd_anchor_neg_weight,
        cgcd_anchor_softmin_tau=cgcd_anchor_softmin_tau,
    )

    margin = float(update_margin)
    stats = base_helper._summarize_pseudo_update_delta(score_teacher, score_student, score_reference, margin)
    accept_mask = score_teacher > (torch.maximum(score_student, score_reference) + margin)
    _apply_teacher_updates_with_markers(
        final_pseudo_labels,
        teacher_predict,
        pseudo_names,
        accept_mask,
        updated_indices,
        active_mask_current,
    )
    stats = _finalize_gate_stats(stats, updated_mask, inactive_mask, active_mask_current)

    if return_stats:
        return final_pseudo_labels, len(updated_indices), update_mode, stats
    return final_pseudo_labels, len(updated_indices), update_mode
