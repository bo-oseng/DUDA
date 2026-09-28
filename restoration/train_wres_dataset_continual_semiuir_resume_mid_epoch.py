"""
Semi-UIR-style continual training entrypoint for OneRestore.

Key differences from train_wres_dataset_continual.py:
- Unlabeled NEW-class data is read directly from real_train_root/<class>/*.
- Unlabeled inputs use weak/strong views after resize instead of pre-extracted patch caches.
- Labeled supervised loss uses structure + perceptual + CE.
- Unlabeled loss keeps only L1 consistency against the reliable pseudo bank.
"""

import argparse
import math
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import yaml
import pyiqa
from torchvision.io import read_image
from torchvision.models import vgg16
from torchvision.utils import make_grid

import train_wres_dataset_continual as base
import datasets_wres_continual_semiuir as semi_data
import utils_lib.helper_semiuir_score as semiuir_score_helper
import utils_lib.helper_semiuir_updated_marker_gate as semiuir_updated_marker_gate
from utils_lib.helper_semiuir_runtime import (
    build_real_eval_step_datasets,
    compute_steps_per_epoch,
    is_supported_image_name,
    resolve_cgcd_anchor_params,
    resolve_pseudo_bank_ops,
    resolve_step_anchor,
    resolve_tensorboard_log_dir,
)
from utils_lib.loss import StructureLoss, VGGPerceptualLoss
from utils_lib.utils import dict2namespace
from utils_lib.helper import save_image_tensor
from eval_onerestore_real import create_synthetic_eval_datasets, evaluate_synthetic_paired


_SEMIUIR_PSEUDO_LOG_FIXED_SAMPLES = {}


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

    iterator = loader
    if show_progress:
        from tqdm import tqdm

        desc = "Initializing Pseudo Labels with Teacher"
        if rank != 0:
            desc = f"{desc} (rank{rank})"
        iterator = tqdm(loader, desc=desc)

    for batch in iterator:
        weak_unlabeled = batch[0].to(device, non_blocking=True)
        pseudo_paths = batch[3]

        with autocast("cuda", dtype=amp_dtype):
            dino_feat = dino_model(dino_transform(weak_unlabeled)).pooler_output
            embedding, _ = cgcd_model(dino_feat)
            restored_teacher = teacher_model(weak_unlabeled, embedding)

        for idx, path in enumerate(pseudo_paths):
            save_image_tensor(restored_teacher[idx], path)

    if show_progress:
        print("✓ Initial Pseudo Labeling Completed!")


@torch.no_grad()
def copy_pseudo_labels_semiuir(loader, rank=0, show_progress=True):
    iterator = loader
    if show_progress:
        from tqdm import tqdm

        desc = "Copying Inputs to Pseudo Bank"
        if rank != 0:
            desc = f"{desc} (rank{rank})"
        iterator = tqdm(loader, desc=desc)

    copied = 0
    for batch in iterator:
        weak_unlabeled = batch[0]
        pseudo_paths = batch[3]
        for idx, path in enumerate(pseudo_paths):
            save_image_tensor(weak_unlabeled[idx], path)
            copied += 1

    if show_progress:
        print(f"✓ Pseudo Label Copy Completed! ({copied} files)")


@torch.no_grad()
def initialize_zero_pseudo_labels_semiuir(loader, rank=0, show_progress=True):
    iterator = loader
    if show_progress:
        from tqdm import tqdm

        desc = "Initializing Zero Pseudo Slots"
        if rank != 0:
            desc = f"{desc} (rank{rank})"
        iterator = tqdm(loader, desc=desc)

    created = 0
    skipped = 0
    for batch in iterator:
        weak_unlabeled = batch[0]
        pseudo_paths = batch[3]
        for idx, path in enumerate(pseudo_paths):
            if os.path.exists(path):
                skipped += 1
                continue
            zero_slot = torch.zeros_like(weak_unlabeled[idx])
            save_image_tensor(zero_slot, path)
            created += 1

    if show_progress:
        print(f"✓ Zero Pseudo Slot Init Completed! (created={created}, skipped={skipped})")


@torch.no_grad()
def log_pseudo_labels_wres_semiuir(
    pseudo_patches_dir, writer, epoch, rank, global_step=None, reference_patches_dir=None
):
    """Log pseudo-label samples for Semi-UIR image-based NEW classes as per-class grids."""
    if rank != 0 or writer is None:
        return

    new_classes = semi_data.base_data.NEW_CLASSES
    if not new_classes:
        return

    tb_step = int(global_step) if global_step is not None else int(epoch)

    for deg_name in new_classes:
        pseudo_dir = os.path.join(pseudo_patches_dir, deg_name)
        if not os.path.isdir(pseudo_dir):
            continue

        reference_dir = pseudo_dir
        if reference_patches_dir:
            candidate_dir = os.path.join(reference_patches_dir, deg_name)
            if os.path.isdir(candidate_dir):
                reference_dir = candidate_dir

        if os.path.abspath(reference_dir) == os.path.abspath(pseudo_dir):
            reference_images = sorted([f for f in os.listdir(reference_dir) if is_supported_image_name(f)])
        else:
            reference_images = [
                os.path.basename(path)
                for path in semi_data.base_data._collect_images(reference_dir)
                if is_supported_image_name(os.path.basename(path))
            ]
        if not reference_images:
            continue

        cache_key = (os.path.abspath(reference_dir), deg_name)
        fixed_samples = _SEMIUIR_PSEUDO_LOG_FIXED_SAMPLES.get(cache_key)
        if fixed_samples is None:
            num_samples = min(20, len(reference_images))
            if num_samples == 1:
                fixed_samples = [reference_images[0]]
            else:
                step = (len(reference_images) - 1) / float(num_samples - 1)
                indices = [int(round(i * step)) for i in range(num_samples)]
                fixed_samples = [reference_images[idx] for idx in indices]
            _SEMIUIR_PSEUDO_LOG_FIXED_SAMPLES[cache_key] = fixed_samples

        image_tensors = []
        missing_images = []
        unreadable_images = []
        for image_name in fixed_samples:
            img_path = os.path.join(pseudo_dir, image_name)
            ref_path = os.path.join(reference_dir, image_name)
            try:
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    img_tensor = read_image(img_path).float() / 255.0
                else:
                    if os.path.exists(ref_path) and os.path.getsize(ref_path) > 0:
                        ref_tensor = read_image(ref_path).float() / 255.0
                        img_tensor = torch.zeros_like(ref_tensor)
                    else:
                        img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
                    missing_images.append(image_name)
            except (OSError, RuntimeError, ValueError):
                unreadable_images.append(image_name)
                if os.path.exists(ref_path):
                    try:
                        ref_tensor = read_image(ref_path).float() / 255.0
                        img_tensor = torch.zeros_like(ref_tensor)
                    except (OSError, RuntimeError, ValueError):
                        img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
                else:
                    img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            image_tensors.append(img_tensor)

        if missing_images:
            print(
                f"[PseudoLabel][info] fill {len(missing_images)} missing slots in {pseudo_dir}: "
                + ", ".join(missing_images[:5])
            )
        if unreadable_images:
            print(
                f"[PseudoLabel][warn] fallback for {len(unreadable_images)} unreadable files in {pseudo_dir}: "
                + ", ".join(unreadable_images[:5])
            )

        grid = make_grid(image_tensors, nrow=max(1, len(image_tensors) // 4), padding=2)
        writer.add_image(f"PseudoLabel/{deg_name}", grid, global_step=tb_step)



@torch.no_grad()
def _resolve_effective_reference_mask(pseudo_list, pseudo_names, device):
    reference_exists_mask = torch.tensor(
        [os.path.exists(path) for path in pseudo_names],
        device=device,
        dtype=torch.bool,
    )
    placeholder_zero_mask = pseudo_list.detach().abs().flatten(1).sum(dim=1) <= 1e-8
    return reference_exists_mask & (~placeholder_zero_mask)


@torch.no_grad()
def _seed_missing_reference_with_student(final_pseudo_labels, student_predict, effective_reference_mask):
    if effective_reference_mask.numel() > 0 and torch.any(~effective_reference_mask):
        final_pseudo_labels[~effective_reference_mask] = student_predict[~effective_reference_mask]


@torch.no_grad()
def _apply_teacher_accept_mask(final_pseudo_labels, teacher_predict, pseudo_names, accept_mask, updated_indices):
    for idx in torch.nonzero(accept_mask, as_tuple=False).view(-1).tolist():
        final_pseudo_labels[idx] = teacher_predict[idx]
        updated_indices.append(idx)
        save_image_tensor(teacher_predict[idx], pseudo_names[idx])


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
):
    if iqa_metric is None:
        raise ValueError("MUSIQ metric is required for MUSIQ pseudo update mode.")

    updated_indices = []
    final_pseudo_labels = pseudo_list.clone()
    effective_reference_mask = _resolve_effective_reference_mask(pseudo_list, pseudo_names, teacher_predict.device)

    score_teacher_musiq = base.get_musiq_score(iqa_metric, teacher_predict)
    score_student_musiq = base.get_musiq_score(iqa_metric, student_predict)
    if score_reference is None:
        score_reference_musiq = base.get_musiq_score(iqa_metric, pseudo_list)
    else:
        score_reference_musiq = score_reference.detach().float().view(-1)

    effective_reference_score_musiq = torch.where(
        effective_reference_mask, score_reference_musiq, score_student_musiq
    )
    _seed_missing_reference_with_student(final_pseudo_labels, student_predict, effective_reference_mask)

    musiq_accept_mask = score_teacher_musiq > (
        torch.maximum(score_student_musiq, effective_reference_score_musiq) + float(update_margin)
    )
    _apply_teacher_accept_mask(
        final_pseudo_labels,
        teacher_predict,
        pseudo_names,
        musiq_accept_mask,
        updated_indices,
    )
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
):
    if iqa_metric is None:
        raise ValueError("MUSIQ metric is required for musiq_then_cgcd mode.")

    updated_indices = []
    final_pseudo_labels = pseudo_list.clone()
    effective_reference_mask = _resolve_effective_reference_mask(pseudo_list, pseudo_names, teacher_predict.device)

    score_teacher_musiq = base.get_musiq_score(iqa_metric, teacher_predict)
    score_student_musiq = base.get_musiq_score(iqa_metric, student_predict)
    if score_reference is None:
        score_reference_musiq = base.get_musiq_score(iqa_metric, pseudo_list)
    else:
        score_reference_musiq = score_reference.detach().float().view(-1)

    effective_reference_score_musiq = torch.where(
        effective_reference_mask, score_reference_musiq, score_student_musiq
    )
    _seed_missing_reference_with_student(final_pseudo_labels, student_predict, effective_reference_mask)

    musiq_accept_mask = score_teacher_musiq > (
        torch.maximum(score_student_musiq, effective_reference_score_musiq) + float(update_margin)
    )
    _apply_teacher_accept_mask(
        final_pseudo_labels,
        teacher_predict,
        pseudo_names,
        musiq_accept_mask,
        updated_indices,
    )

    fallback_candidate_mask = ~musiq_accept_mask
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
        effective_reference_score_cgcd = torch.where(
            effective_reference_mask, score_reference_cgcd, score_student_cgcd
        )
        cgcd_accept_mask = fallback_candidate_mask & (
            score_teacher_cgcd > (
                torch.maximum(score_student_cgcd, effective_reference_score_cgcd) + float(fallback_update_margin)
            )
        )
        _apply_teacher_accept_mask(
            final_pseudo_labels,
            teacher_predict,
            pseudo_names,
            cgcd_accept_mask,
            updated_indices,
        )

    return final_pseudo_labels, len(updated_indices), "MUSIQ->CGCD_anchor"


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
        )
    if semiuir_score_helper.is_hybrid_mode(normalized_mode):
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
        )

    updated_indices = []
    final_pseudo_labels = pseudo_list.clone()
    effective_reference_mask = _resolve_effective_reference_mask(pseudo_list, pseudo_names, teacher_predict.device)

    score_teacher, score_student, score_reference, update_mode = semiuir_score_helper.compute_reliable_score_triplet(
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
        clear_weight=cgcd_anchor_clear_weight,
        neg_weight=cgcd_anchor_neg_weight,
        softmin_tau=cgcd_anchor_softmin_tau,
    )

    effective_reference_score = torch.where(effective_reference_mask, score_reference, score_student)
    _seed_missing_reference_with_student(final_pseudo_labels, student_predict, effective_reference_mask)

    accept_mask = score_teacher > (torch.maximum(score_student, effective_reference_score) + float(update_margin))
    _apply_teacher_accept_mask(
        final_pseudo_labels,
        teacher_predict,
        pseudo_names,
        accept_mask,
        updated_indices,
    )
    return final_pseudo_labels, len(updated_indices), update_mode


def _prefetch_train_batch_to_device(labeled_batch, unlabeled_batch, device, transfer_stream):
    hq_labeled_cpu, lq_labeled_cpu, gt_deg_label_cpu = labeled_batch
    weak_unlabeled_cpu, strong_unlabeled_cpu, pseudo_list_cpu, pseudo_names_cpu = unlabeled_batch
    with torch.cuda.stream(transfer_stream):
        hq_labeled_gpu = hq_labeled_cpu.to(device, non_blocking=True)
        lq_labeled_gpu = lq_labeled_cpu.to(device, non_blocking=True)
        gt_deg_label_gpu = gt_deg_label_cpu.to(device, non_blocking=True)
        weak_unlabeled_gpu = weak_unlabeled_cpu.to(device, non_blocking=True)
        strong_unlabeled_gpu = strong_unlabeled_cpu.to(device, non_blocking=True)
        pseudo_list_gpu = pseudo_list_cpu.to(device, non_blocking=True)
    return (
        hq_labeled_gpu,
        lq_labeled_gpu,
        gt_deg_label_gpu,
        weak_unlabeled_gpu,
        strong_unlabeled_gpu,
        pseudo_list_gpu,
        pseudo_names_cpu,
    )


def build_dataloaders(cfg, rank, world_size):
    config_dict = {
        "deg_map": vars(cfg.deg_map),
        "incremental": {
            "stage": cfg.incremental.stage,
            "inc_class_num": cfg.incremental.inc_class_num,
            "base_class_num": cfg.incremental.base_class_num,
        },
        "cgcd": {
            "class_order": cfg.cgcd.class_order,
            "class_mappings": cfg.cgcd.class_mappings,
        },
        "train": {
            "data_root_train": cfg.train.data_root_train,
            "data_root_val": cfg.train.data_root_val,
            "all_classes": cfg.train.all_classes,
            "input_subdir": getattr(cfg.train, "input_subdir", "input"),
            "gt_subdir": getattr(cfg.train, "gt_subdir", "gt"),
        },
    }
    semi_data.init_from_config(config_dict)
    if rank == 0:
        base.print_incremental_info(config_dict)

    train_data_old, _, val_data_all = base.get_train_val_data_for_stage(config_dict)

    labeled_dataset = base.create_labeled_dataset(train_data_old, cfg.train.patch_size)
    unlabeled_dataset = semi_data.create_stage1_unlabeled_dataset(
        real_train_root=getattr(cfg.train, "real_train_root", cfg.train.data_root_train),
        pseudo_patches_dir=cfg.train.pseudo_patches_dir,
        fine_size=int(getattr(cfg.train, "unlabeled_fine_size", cfg.train.patch_size)),
    )

    use_validation = bool(getattr(cfg.train, "use_validation", False))
    if use_validation:
        val_datasets = base.create_stage1_val_datasets(val_data_all)
    else:
        val_datasets = []
        if rank == 0:
            print("[WRES-SemiUIR] Validation disabled (train.use_validation=False)")

    use_real_eval = bool(getattr(cfg.train, "use_real_eval", False))
    if use_real_eval:
        real_eval_datasets = base.create_real_eval_datasets(
            getattr(cfg.train, "real_eval_root", ""),
            list(getattr(cfg.train, "real_eval_sets", [])),
        )
    else:
        real_eval_datasets = []
        if rank == 0:
            print("[WRES-SemiUIR] Real eval disabled (train.use_real_eval=False)")

    labeled_sampler = DistributedSampler(
        labeled_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed
    )
    unlabeled_sampler = DistributedSampler(
        unlabeled_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed
    )

    pin_memory = bool(getattr(cfg.train, "pin_memory", True))
    prefetch_factor = int(getattr(cfg.train, "prefetch_factor", 4))
    persistent_workers = bool(getattr(cfg.train, "persistent_workers", True))

    loader_kwargs = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if cfg.train.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        loader_kwargs["persistent_workers"] = persistent_workers

    if rank == 0:
        print(
            "[WRES-SemiUIR] Train loader config: "
            f"batch_size={loader_kwargs['batch_size']}, "
            f"num_workers={loader_kwargs['num_workers']}, "
            f"pin_memory={loader_kwargs['pin_memory']}, "
            f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'N/A')}, "
            f"persistent_workers={loader_kwargs.get('persistent_workers', False)}"
        )

    labeled_loader = DataLoader(labeled_dataset, sampler=labeled_sampler, **loader_kwargs)
    unlabeled_loader = DataLoader(unlabeled_dataset, sampler=unlabeled_sampler, **loader_kwargs)

    return (
        labeled_loader,
        unlabeled_loader,
        val_datasets,
        real_eval_datasets,
        labeled_sampler,
        unlabeled_sampler,
    )


def train_one_epoch(
    student_model,
    teacher_model,
    dino_model,
    cgcd_model,
    iqa_metrics,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    scaler,
    device,
    epoch,
    cfg,
    writer,
    global_step,
    rank,
    amp_dtype=torch.float16,
    pseudo_update_mode="musiq",
    musiq_warmup_epochs=0,
    loss_str=None,
    loss_per=None,
    checkpoint_step_callback=None,
    pseudo_log_step_callback=None,
    validation_step_callback=None,
    reliable_pseudo_fn=None,
    use_real_eval=False,
    real_eval_datasets=None,
    real_eval_freq_steps=0,
    real_eval_max_side=0,
    real_eval_allow_runtime_resize_retry=True,
    world_size=1,
    resume_skip_steps=0,
):
    current_stage = cfg.incremental.stage
    if reliable_pseudo_fn is None:
        reliable_pseudo_fn = get_reliable_semiuir

    teacher_model.eval()
    base.freeze_teachers_parameters(teacher_model)

    student_model.train()
    cgcd_model.train()

    criterion_l1 = nn.L1Loss()
    criterion_ce = nn.CrossEntropyLoss()
    perceptual_weight = float(getattr(cfg.train, "perceptual_weight", 0.3))
    structure_weight = float(getattr(cfg.train, "structure_weight", 1.0))
    max_train_steps = int(getattr(cfg.train, "max_train_steps", 0) or 0)

    total_updated_cnt = 0
    running_loss_sup = 0.0
    running_loss_unsup = 0.0
    running_loss_unsup_weighted = 0.0
    running_loss_total = 0.0
    running_psnr_labeled = 0.0

    transfer_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    labeled_steps = len(labeled_loader)
    unlabeled_steps = len(unlabeled_loader)
    step_anchor = resolve_step_anchor(cfg, labeled_steps, unlabeled_steps, rank=rank)
    if step_anchor == "unlabeled":
        num_steps = unlabeled_steps
        cycle_labeled = True
        cycle_unlabeled = False
    elif step_anchor == "labeled":
        num_steps = labeled_steps
        cycle_labeled = False
        cycle_unlabeled = True
    elif step_anchor == "max":
        num_steps = max(labeled_steps, unlabeled_steps)
        cycle_labeled = labeled_steps < num_steps
        cycle_unlabeled = unlabeled_steps < num_steps
    elif step_anchor == "min":
        num_steps = min(labeled_steps, unlabeled_steps)
        cycle_labeled = False
        cycle_unlabeled = False
    else:
        raise ValueError(f"Invalid train.step_anchor={step_anchor}. Choose one of: unlabeled, labeled, max, min.")

    resume_skip_steps = int(resume_skip_steps or 0)
    if resume_skip_steps < 0:
        raise ValueError(f"resume_skip_steps must be >= 0, got {resume_skip_steps}")
    expected_num_steps = int(getattr(cfg.train, "resume_expected_steps_per_epoch", 0) or 0)
    if resume_skip_steps > 0 and expected_num_steps > 0 and num_steps != expected_num_steps:
        raise ValueError(
            f"resume_expected_steps_per_epoch={expected_num_steps}, but current num_steps={num_steps}. "
            "Use the same DDP world size and batch setup as the interrupted run."
        )
    if resume_skip_steps >= num_steps and resume_skip_steps > 0:
        raise ValueError(f"resume_skip_steps={resume_skip_steps} must be smaller than num_steps={num_steps}")
    start_step = resume_skip_steps

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    if rank == 0:
        print(
            "[WRES-SemiUIR] Step scheduler: "
            f"anchor={step_anchor}, steps={num_steps}, "
            f"labeled_steps={labeled_steps}, unlabeled_steps={unlabeled_steps}, "
            f"cycle_labeled={cycle_labeled}, cycle_unlabeled={cycle_unlabeled}"
        )
        if start_step > 0:
            print(f"[WRES-SemiUIR][Resume] Skipping already completed epoch steps: {start_step}/{num_steps}")

    from tqdm import tqdm

    pbar = (
        tqdm(
            range(start_step, num_steps),
            desc=f"Epoch {epoch}",
            total=num_steps,
            initial=start_step,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        if rank == 0
        else range(start_step, num_steps)
    )

    rampup_epoch = float(getattr(cfg.train, "rampup_epoch", 50.0))
    rampup_w = base.get_current_consistency_weight(epoch, cfg.train.consistency_weight, rampup_epoch)
    effective_update_mode = "musiq" if epoch <= musiq_warmup_epochs else pseudo_update_mode
    cgcd_anchor_params = resolve_cgcd_anchor_params(cfg, base.get_cfg_cgcd_value)
    pseudo_update_margin = float(base.get_cfg_cgcd_value(cfg, "pseudo_update_margin", 0.0))
    pseudo_update_cgcd_fallback_margin = float(getattr(cfg.train, "pseudo_update_cgcd_fallback_margin", 0.0))

    for _ in range(start_step):
        _, labeled_iter = base._next_batch_with_cycle(labeled_iter, labeled_loader, cycle_labeled, "labeled_loader")
        _, unlabeled_iter = base._next_batch_with_cycle(
            unlabeled_iter,
            unlabeled_loader,
            cycle_unlabeled,
            "unlabeled_loader",
        )

    labeled_batch_cpu, labeled_iter = base._next_batch_with_cycle(
        labeled_iter, labeled_loader, cycle_labeled, "labeled_loader"
    )
    unlabeled_batch_cpu, unlabeled_iter = base._next_batch_with_cycle(
        unlabeled_iter,
        unlabeled_loader,
        cycle_unlabeled,
        "unlabeled_loader",
    )
    prefetched_batch = _prefetch_train_batch_to_device(labeled_batch_cpu, unlabeled_batch_cpu, device, transfer_stream)
    steps_run = 0

    for i in pbar:
        if max_train_steps > 0 and global_step >= max_train_steps:
            break
        compute_stream.wait_stream(transfer_stream)
        (
            hq_labeled,
            lq_labeled,
            gt_deg_label,
            lq_unlabeled_weak,
            lq_unlabeled_strong,
            pseudo_list,
            pseudo_names,
        ) = prefetched_batch

        with torch.no_grad():
            batch_labeled = lq_labeled.shape[0]
            lq_merged = torch.cat([lq_labeled, lq_unlabeled_weak], dim=0)
            dino_feat_merged = dino_model(base.transform_resize(lq_merged)).pooler_output
            dino_feat_labeled = dino_feat_merged[:batch_labeled]
            dino_feat_unlabeled = dino_feat_merged[batch_labeled:]

        with autocast("cuda", dtype=amp_dtype):
            embedding_labeled, pseudo_class_labeled = cgcd_model(dino_feat_labeled)
            restored_labeled = student_model(lq_labeled, embedding_labeled)

            loss_structure = loss_str(restored_labeled, hq_labeled)
            loss_perceptual = loss_per(restored_labeled, hq_labeled)
            loss_ce_labeled = cfg.train.ce_weight * criterion_ce(pseudo_class_labeled, gt_deg_label)
            loss_sup = structure_weight * loss_structure + perceptual_weight * loss_perceptual + loss_ce_labeled
            
            embedding_unlabeled, _ = cgcd_model(dino_feat_unlabeled)
            student_output_unlabeled = student_model(lq_unlabeled_strong, embedding_unlabeled)

            with torch.no_grad():
                teacher_output_unlabeled = teacher_model(lq_unlabeled_weak, embedding_unlabeled)
                if effective_update_mode == "fgresq":
                    iqa_metric = iqa_metrics.get("fgresq")
                    if iqa_metric is None:
                        raise ValueError("FGResQ metric is required but not initialized.")
                    score_reference = base.get_fgresq_score(iqa_metric, pseudo_list)
                elif effective_update_mode == "cgcd":
                    iqa_metric = iqa_metrics.get("musiq")
                    score_reference = None
                else:
                    iqa_metric = iqa_metrics.get("musiq")
                    if iqa_metric is None:
                        raise ValueError("MUSIQ metric is required but not initialized.")
                    score_reference = base.get_musiq_score(iqa_metric, pseudo_list)

                orig2classifier = base.datasets_wres.ORIG2CLASSIFIER
                if orig2classifier is None:
                    raise ValueError("ORIG2CLASSIFIER is not initialized. Call init_from_config() first.")

                cgcd_inner = cgcd_model.module if hasattr(cgcd_model, "module") else cgcd_model
                reliable_pseudo_labels, updated_cnt, update_mode = reliable_pseudo_fn(
                    iqa_metric,
                    teacher_output_unlabeled,
                    student_output_unlabeled.detach(),
                    score_reference,
                    pseudo_list,
                    pseudo_names,
                    rank,
                    pseudo_update_mode=effective_update_mode,
                    dino_model=dino_model,
                    cgcd_model=cgcd_inner,
                    dino_transform=base.transform_resize,
                    epoch=epoch,
                    orig2classifier=orig2classifier,
                    amp_dtype=amp_dtype,
                    cgcd_anchor_clear_weight=cgcd_anchor_params["clear_weight"],
                    cgcd_anchor_neg_weight=cgcd_anchor_params["neg_weight"],
                    cgcd_anchor_softmin_tau=cgcd_anchor_params["softmin_tau"],
                    update_margin=pseudo_update_margin,
                    fallback_update_margin=pseudo_update_cgcd_fallback_margin,
                )
                total_updated_cnt += updated_cnt

            with autocast("cuda", enabled=False):
                loss_unsup = criterion_l1(
                    student_output_unlabeled.float(),
                    reliable_pseudo_labels.float(),
                )
            loss_unsup_weighted = rampup_w * loss_unsup

            loss_total = loss_sup + loss_unsup_weighted

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(cgcd_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(cgcd_model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            base.update_teacher_ema_all_params(teacher_model, student_model, alpha=0.996, global_step=global_step)
            psnr_labeled = base.pt_psnr(hq_labeled, restored_labeled).mean()

        running_loss_sup += loss_sup.detach().item()
        running_loss_unsup += loss_unsup.detach().item()
        running_loss_unsup_weighted += loss_unsup_weighted.detach().item()
        running_loss_total += loss_total.detach().item()
        running_psnr_labeled += psnr_labeled.detach().item()

        display_steps = steps_run + 1
        if rank == 0 and (i + 1) % 10 == 0:
            pbar.set_description(
                f"[Stage {current_stage}] Epoch {epoch}/{cfg.train.epochs} | "
                f"Update mode: {effective_update_mode} (target: {pseudo_update_mode}), "
                f"CGCD score: anchor | "
                f"Sup: {running_loss_sup/display_steps:.4f}, UnsupRaw: {running_loss_unsup/display_steps:.4f}, "
                f"UnsupW: {running_loss_unsup_weighted/display_steps:.4f}, PSNR: {running_psnr_labeled/display_steps:.2f}, "
                f"updated: {total_updated_cnt}"
            )

        if rank == 0 and (i + 1) % cfg.train.print_freq == 0 and writer is not None:
            writer.add_scalar("train/loss_sup", loss_sup.detach().item(), global_step)
            writer.add_scalar("train/loss_unsup", loss_unsup.detach().item(), global_step)
            writer.add_scalar("train/loss_unsup_raw", loss_unsup.detach().item(), global_step)
            writer.add_scalar("train/loss_unsup_weighted", loss_unsup_weighted.detach().item(), global_step)
            writer.add_scalar("train/loss_total", loss_total.detach().item(), global_step)
            writer.add_scalar("train/psnr", psnr_labeled.detach().item(), global_step)
            writer.add_scalar("train/psnr_labeled", psnr_labeled.detach().item(), global_step)
            writer.add_scalar("train/pseudo_update_cnt", total_updated_cnt, global_step)
            writer.add_scalar("train/rampup_weight", rampup_w, global_step)

        if i + 1 < num_steps:
            labeled_batch_cpu, labeled_iter = base._next_batch_with_cycle(
                labeled_iter, labeled_loader, cycle_labeled, "labeled_loader"
            )
            unlabeled_batch_cpu, unlabeled_iter = base._next_batch_with_cycle(
                unlabeled_iter,
                unlabeled_loader,
                cycle_unlabeled,
                "unlabeled_loader",
            )
            prefetched_batch = _prefetch_train_batch_to_device(
                labeled_batch_cpu, unlabeled_batch_cpu, device, transfer_stream
            )

        global_step += 1

        if checkpoint_step_callback is not None:
            checkpoint_step_callback(global_step=global_step, epoch=epoch)
        if pseudo_log_step_callback is not None:
            pseudo_log_step_callback(global_step=global_step, epoch=epoch)
        if validation_step_callback is not None:
            validation_step_callback(global_step=global_step, epoch=epoch)

        if (
            use_real_eval
            and real_eval_datasets
            and int(real_eval_freq_steps) > 0
            and (int(global_step) % int(real_eval_freq_steps) == 0)
        ):
            if rank == 0:
                print(
                    f"\n[Stage {current_stage}] Step-based real eval at global_step={int(global_step)} "
                    f"(freq_steps={int(real_eval_freq_steps)})"
                )
            base.evaluate_real_unpaired(
                teacher_model,
                dino_model,
                cgcd_model.module if hasattr(cgcd_model, "module") else cgcd_model,
                real_eval_datasets,
                device,
                epoch,
                writer,
                rank,
                world_size,
                amp_dtype,
                max_eval_side=real_eval_max_side,
                allow_runtime_resize_retry=real_eval_allow_runtime_resize_retry,
                tb_step=int(global_step),
                step_label=f"Step {int(global_step)}",
                tb_prefix="RealEvalStep",
                log_prefix="RealEvalStep",
            )
            dist.barrier()

        steps_run += 1

    denom = max(1, steps_run)
    avg_loss_total = running_loss_total / denom
    avg_loss_sup = running_loss_sup / denom
    avg_loss_unsup = running_loss_unsup / denom
    avg_loss_unsup_weighted = running_loss_unsup_weighted / denom
    avg_psnr = running_psnr_labeled / denom

    if rank == 0:
        print(
            f"\n[Stage {current_stage}] Epoch {epoch} - "
            f"Update mode: {effective_update_mode} (target: {pseudo_update_mode}), CGCD score: anchor, "
            f"Total: {avg_loss_total:.4f}, Sup: {avg_loss_sup:.4f}, UnsupRaw: {avg_loss_unsup:.4f}, "
            f"UnsupW: {avg_loss_unsup_weighted:.4f}, PSNR: {avg_psnr:.2f} dB, Pseudo updated: {total_updated_cnt}"
        )

    return global_step, avg_loss_total, total_updated_cnt


def infer_mid_epoch_resume(saved_epoch, global_step, steps_per_epoch):
    saved_epoch = int(saved_epoch)
    global_step = int(global_step)
    steps_per_epoch = int(steps_per_epoch)
    if saved_epoch < 1 or global_step < 0 or steps_per_epoch <= 0:
        raise ValueError(
            f"Invalid resume state: epoch={saved_epoch}, global_step={global_step}, "
            f"steps_per_epoch={steps_per_epoch}"
        )
    step_in_epoch = global_step - (saved_epoch - 1) * steps_per_epoch
    if 0 < step_in_epoch < steps_per_epoch:
        return saved_epoch, step_in_epoch
    return saved_epoch + 1, 0


def train_stage(cfg):
    current_stage = cfg.incremental.stage
    rank, world_size, local_rank = base.setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    base.setup_runtime(cfg, rank)
    checkpoint_dir = os.path.join(cfg.train.save_dir, cfg.train.config_parent_dir, cfg.exp_name)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    use_bf16 = bool(getattr(cfg.train, "use_bf16", False))
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if rank == 0:
        tb_log_dir = resolve_tensorboard_log_dir(
            cfg.train.log_dir,
            cfg.exp_name,
            getattr(cfg.train, "resume", None),
            reuse_latest_resume_dir=bool(getattr(cfg.train, "reuse_latest_tensorboard_on_resume", False)),
        )
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"\n[Stage {current_stage}] Start - Device: {device}, World Size: {world_size}")
        print(f"AMP dtype: {'BF16' if use_bf16 else 'FP16'}")
        print(f"TensorBoard log dir: {tb_log_dir}")
    else:
        writer = None

    (
        labeled_loader,
        unlabeled_loader,
        val_datasets,
        real_eval_datasets,
        labeled_sampler,
        unlabeled_sampler,
    ) = build_dataloaders(cfg, rank, world_size)
    real_eval_step_datasets = build_real_eval_step_datasets(cfg, base.create_real_eval_datasets, rank=rank)
    paired_eval_enabled = bool(getattr(cfg.train, "paired_eval_enabled", False))
    paired_eval_freq = int(getattr(cfg.train, "paired_eval_freq", 1) or 1)
    paired_eval_num_workers = int(getattr(cfg.train, "paired_eval_num_workers", 2) or 0)
    paired_eval_datasets = []
    if paired_eval_enabled:
        paired_eval_datasets = create_synthetic_eval_datasets(
            getattr(cfg.train, "paired_eval_root"),
            list(getattr(cfg.train, "paired_eval_classes", ["test"])),
            input_subdir=str(getattr(cfg.train, "paired_eval_input_subdir", "blur")),
            gt_subdir=str(getattr(cfg.train, "paired_eval_gt_subdir", "sharp")),
            per_class=int(getattr(cfg.train, "paired_eval_per_class", 0) or 0),
            sample_seed=int(getattr(cfg.train, "paired_eval_seed", 42)),
            rank=rank,
        )
    pseudo_bank_ops = resolve_pseudo_bank_ops(
        cfg,
        local_ops={
            "initialize_pseudo_labels": initialize_pseudo_labels_semiuir,
            "copy_pseudo_labels": copy_pseudo_labels_semiuir,
            "initialize_zero_pseudo_labels": initialize_zero_pseudo_labels_semiuir,
            "get_reliable": get_reliable_semiuir,
        },
        rank=rank,
    )

    steps_per_epoch, step_anchor, labeled_steps, unlabeled_steps = compute_steps_per_epoch(
        cfg, labeled_loader, unlabeled_loader, rank=rank
    )

    total_epoch = int(getattr(cfg.train, "total_epoch", 0) or 0)
    if total_epoch > 0:
        cfg.train.epochs = total_epoch
        if rank == 0:
            print(
                f"[WRES-SemiUIR] Using train.total_epoch={cfg.train.epochs} "
                f"(step_anchor={step_anchor}, steps_per_epoch={steps_per_epoch})"
            )
    elif not hasattr(cfg.train, "epochs") or getattr(cfg.train, "epochs", None) is None:
        max_train_steps = int(getattr(cfg.train, "max_train_steps", 0) or 0)
        if max_train_steps <= 0:
            raise ValueError("train.epochs or train.max_train_steps must be set.")

        cfg.train.epochs = max(1, int(math.ceil(float(max_train_steps) / float(steps_per_epoch))))
        if rank == 0:
            print(
                f"[WRES-SemiUIR] Derived epochs from max_train_steps: epochs={cfg.train.epochs}, "
                f"steps_per_epoch={steps_per_epoch}, max_train_steps={max_train_steps}, step_anchor={step_anchor}"
            )
    use_validation = bool(getattr(cfg.train, "use_validation", False))
    use_real_eval = bool(getattr(cfg.train, "use_real_eval", False))
    val_freq = int(getattr(cfg.train, "val_freq", 1) or 1)
    raw_val_step_freq = getattr(cfg.train, "val_step_freq", None)
    if raw_val_step_freq is None:
        raw_val_step_freq = getattr(cfg.train, "val_freq_steps", 0)
    val_step_freq = int(raw_val_step_freq or 0)
    if val_step_freq < 0:
        raise ValueError(f"val_step_freq must be >= 0, got {val_step_freq}")
    real_eval_freq = int(getattr(cfg.train, "real_eval_freq", val_freq))
    real_eval_freq_steps = int(getattr(cfg.train, "real_eval_freq_steps", 0) or 0)
    real_eval_max_side = int(getattr(cfg.train, "real_eval_max_side", 0) or 0)
    real_eval_allow_runtime_resize_retry = bool(getattr(cfg.train, "real_eval_allow_runtime_resize_retry", True))
    if rank == 0 and use_real_eval and real_eval_freq_steps > 0:
        print(
            f"[WRES-SemiUIR] Step-based real eval enabled: freq_steps={real_eval_freq_steps} "
            f"(step_anchor={step_anchor}, steps_per_epoch={steps_per_epoch}, "
            f"approx {steps_per_epoch / float(real_eval_freq_steps):.2f} evals/epoch)"
        )
    if rank == 0 and int(getattr(cfg.train, "pseudo_log_freq_steps", 0) or 0) > 0:
        print(
            f"[WRES-SemiUIR] Step-based pseudo image log enabled: "
            f"freq_steps={int(getattr(cfg.train, 'pseudo_log_freq_steps', 0) or 0)}"
        )
    if rank == 0 and use_validation:
        print(f"[WRES-SemiUIR] Epoch-based validation enabled: freq={val_freq}")
        if val_step_freq > 0:
            print(f"[WRES-SemiUIR] Step-based validation enabled: freq={val_step_freq}")
        else:
            print("[WRES-SemiUIR] Step-based validation disabled")

    if rank == 0:
        student, teacher, dino, cgcd = base.build_models(cfg, device, rank)
    dist.barrier()
    if rank != 0:
        student, teacher, dino, cgcd = base.build_models(cfg, device, rank)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(cgcd.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    scaler = None if use_bf16 else GradScaler("cuda")
    musiq_warmup_epochs = int(getattr(cfg.train, "musiq_warmup_epochs", 0))
    if musiq_warmup_epochs < 0:
        musiq_warmup_epochs = cfg.train.epochs // 2

    need_fgresq = cfg.train.pseudo_update_mode == "fgresq"
    need_musiq = (cfg.train.pseudo_update_mode != "fgresq") or (musiq_warmup_epochs > 0)

    iqa_metrics = {"musiq": None, "fgresq": None}
    if need_musiq:
        iqa_metrics["musiq"] = pyiqa.create_metric("musiq", as_loss=True).cuda()
        if rank == 0:
            print("✓ MUSIQ metric initialized")
    if need_fgresq:
        iqa_metrics["fgresq"] = base.create_fgresq_metric(device=device)
        if rank == 0:
            print("✓ FGResQ metric initialized")
    if rank == 0 and musiq_warmup_epochs > 0:
        print(
            f"✓ MUSIQ warmup enabled: first {musiq_warmup_epochs} epochs use MUSIQ, then switch to {cfg.train.pseudo_update_mode}"
        )

    vgg_model = vgg16(pretrained=True).features[:16].to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_str = StructureLoss().to(device)
    loss_per = VGGPerceptualLoss(vgg_model).to(device)

    start_epoch, global_step, best_psnr, _ = base.load_checkpoints(
        student, teacher, cgcd, optimizer, scheduler, cfg, device, rank
    )
    resume_start_epoch_override = int(getattr(cfg.train, "resume_start_epoch_override", 0) or 0)
    if (
        resume_start_epoch_override <= 0
        and getattr(cfg.train, "resume", None)
        and bool(getattr(cfg.train, "resume_auto_mid_epoch", False))
    ):
        saved_epoch = start_epoch - 1
        inferred_epoch, inferred_skip = infer_mid_epoch_resume(saved_epoch, global_step, steps_per_epoch)
        if inferred_skip > 0:
            resume_start_epoch_override = inferred_epoch
            cfg.train.resume_skip_steps = inferred_skip
            cfg.train.resume_expected_steps_per_epoch = steps_per_epoch
            if rank == 0:
                print(
                    f"[WRES-SemiUIR][Resume] Auto mid-epoch state: epoch={inferred_epoch}, "
                    f"skip={inferred_skip}/{steps_per_epoch}, global_step={global_step}"
                )
    if resume_start_epoch_override > 0:
        if not getattr(cfg.train, "resume", None):
            raise ValueError("resume_start_epoch_override requires train.resume to be set.")
        if rank == 0:
            print(f"[WRES-SemiUIR][Resume] Override start_epoch: {start_epoch} -> {resume_start_epoch_override}")
        start_epoch = resume_start_epoch_override
        if bool(getattr(cfg.train, "resume_rewind_scheduler", True)):
            target_last_epoch = max(0, resume_start_epoch_override - 1)
            eta_min = float(getattr(scheduler, "eta_min", 0.0))
            t_max = float(getattr(scheduler, "T_max", max(1, int(getattr(cfg.train, "epochs", 1)))))
            lrs = []
            for param_group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                lr = eta_min + (float(base_lr) - eta_min) * (1.0 + math.cos(math.pi * target_last_epoch / t_max)) / 2.0
                param_group["lr"] = lr
                lrs.append(lr)
            scheduler.last_epoch = target_last_epoch
            scheduler._last_lr = lrs
            if rank == 0:
                lr_text = ", ".join(f"{lr:.8f}" for lr in lrs)
                print(f"[WRES-SemiUIR][Resume] Rewound scheduler to last_epoch={target_last_epoch}, lr={lr_text}")
    is_resume_run = bool(getattr(cfg.train, "resume", None))
    init_validate_on_resume = bool(getattr(cfg.train, "init_validate_on_resume", False))
    run_init_validate = bool(getattr(cfg.train, "init_validate", False)) and (
        init_validate_on_resume or not is_resume_run
    )
    cfg_init_pseudo = bool(getattr(cfg.train, "init_pseudo_label", False))
    cfg_copy_pseudo = bool(getattr(cfg.train, "copy_pseudo_label", False))
    skip_pseudo_init = bool(getattr(cfg.train, "skip_pseudo_init", False)) and (not is_resume_run)
    run_init_pseudo = cfg_init_pseudo and not is_resume_run
    run_init_copy = cfg_copy_pseudo and (not cfg_init_pseudo) and (not is_resume_run)
    run_init_zero = (not cfg_init_pseudo) and (not cfg_copy_pseudo) and (not is_resume_run)
    if skip_pseudo_init:
        if rank == 0:
            print("[InitPseudo] skip_pseudo_init enabled: skip python-side pseudo init/copy/zero loop.")
            if cfg_init_pseudo:
                print("[InitPseudo] init_pseudo_label is ignored because skip_pseudo_init is enabled.")
            if cfg_copy_pseudo:
                print("[InitPseudo] copy_pseudo_label is ignored because skip_pseudo_init is enabled.")
        run_init_pseudo = False
        run_init_copy = False
        run_init_zero = False
    elif rank == 0 and cfg_init_pseudo and cfg_copy_pseudo and not is_resume_run:
        print("[InitPseudo] copy_pseudo_label is ignored because init_pseudo_label is enabled.")
    if rank == 0 and is_resume_run:
        if getattr(cfg.train, "init_validate", False) and not init_validate_on_resume:
            print("[Resume] Skip initial validation at epoch 0 to keep TensorBoard scalar steps monotonic.")
        if cfg_init_pseudo:
            print("[Resume] Skip pseudo-label initialization on resume.")
        if cfg_copy_pseudo:
            print("[Resume] Skip pseudo-label copy initialization on resume.")

    for param in student.parameters():
        param.requires_grad = True

    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())

    student = DDP(student.to(device), device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    cgcd = DDP(cgcd.to(device), device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    def _run_paired_eval(epoch, global_step, step_label):
        if not paired_eval_enabled or not paired_eval_datasets:
            return
        summary = evaluate_synthetic_paired(
            teacher,
            dino,
            cgcd.module,
            paired_eval_datasets,
            device,
            rank,
            world_size,
            use_ddp=(world_size > 1),
            amp_dtype=amp_dtype,
            output_dir=None,
            save_images=False,
            num_workers=paired_eval_num_workers,
            model_tag="teacher",
        )
        if rank == 0 and writer is not None and summary is not None:
            average = summary["average"]
            writer.add_scalar("GoProTest/PSNR", average["psnr"], int(global_step))
            writer.add_scalar("GoProTest/SSIM", average["ssim"], int(global_step))
            writer.add_scalar("GoProTest/L1", average["l1"], int(global_step))
            writer.flush()
            print(
                f"[GoProTest] {step_label}: PSNR={average['psnr']:.4f}, "
                f"SSIM={average['ssim']:.4f}, L1={average['l1']:.6f}"
            )
        cgcd.train()
        dist.barrier()

    save_freq = int(getattr(cfg.train, "save_freq", 1) or 1)
    raw_save_step_freq = getattr(cfg.train, "save_step_freq", None)
    if raw_save_step_freq is None:
        raw_save_step_freq = getattr(cfg.train, "save_freq_steps", 0)
    save_step_freq = int(raw_save_step_freq or 0)
    if save_step_freq < 0:
        raise ValueError(f"save_step_freq must be >= 0, got {save_step_freq}")
    raw_pseudo_log_freq_steps = int(getattr(cfg.train, "pseudo_log_freq_steps", 0) or 0)
    if raw_pseudo_log_freq_steps < 0:
        raise ValueError(f"pseudo_log_freq_steps must be >= 0, got {raw_pseudo_log_freq_steps}")
    pseudo_log_freq_steps = raw_pseudo_log_freq_steps if raw_pseudo_log_freq_steps > 0 else save_step_freq
    max_train_steps = int(getattr(cfg.train, "max_train_steps", 0) or 0)

    def _build_checkpoint_state(epoch_value, global_step_value, is_best_value):
        return {
            "epoch": epoch_value,
            "stage": current_stage,
            "student_state_dict": student.module.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "cgcd_state_dict": cgcd.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step_value,
            "best_psnr": best_psnr,
            "is_best": is_best_value,
        }

    def _maybe_save_step_checkpoint(global_step, epoch):
        if rank != 0 or save_step_freq <= 0:
            return
        if int(global_step) <= 0 or (int(global_step) % int(save_step_freq) != 0):
            return
        state = _build_checkpoint_state(epoch_value=epoch, global_step_value=int(global_step), is_best_value=False)
        step_path = os.path.join(checkpoint_dir, f"stage{current_stage}_step_{int(global_step)}.pth")
        torch.save(state, step_path)
        latest_path = os.path.join(checkpoint_dir, f"stage{current_stage}_latest.pth")
        torch.save(state, latest_path)
        print(f"Saved checkpoint: {step_path}")

    def _maybe_log_pseudo_labels_step(global_step, epoch):
        if rank != 0 or writer is None or pseudo_log_freq_steps <= 0:
            return
        if int(global_step) <= 0 or (int(global_step) % int(pseudo_log_freq_steps) != 0):
            return
        log_pseudo_labels_wres_semiuir(
            cfg.train.pseudo_patches_dir,
            writer,
            epoch,
            rank,
            global_step=int(global_step),
            reference_patches_dir=getattr(cfg.train, "lq_patches_dir", getattr(cfg.train, "real_train_root", None)),
        )
        writer.flush()

    def _maybe_run_step_validation(global_step, epoch):
        nonlocal best_psnr
        if not use_validation or val_step_freq <= 0 or len(val_datasets) == 0:
            return
        if int(global_step) <= 0 or (int(global_step) % int(val_step_freq) != 0):
            return
        psnr_old, psnr_new, psnr_all = base.validate_stage(
            teacher,
            dino,
            cgcd.module,
            val_datasets,
            device,
            epoch,
            cfg,
            writer,
            rank,
            world_size,
            amp_dtype,
            tb_step=int(global_step),
            step_label=f"Step {int(global_step)}",
        )
        if rank == 0 and psnr_all > best_psnr:
            best_psnr = psnr_all
        cgcd.train()
        dist.barrier()

    if paired_eval_enabled and is_resume_run and bool(getattr(cfg.train, "paired_eval_on_resume", True)):
        _run_paired_eval(start_epoch, global_step, f"Resume Step {global_step}")

    if use_validation and len(val_datasets) > 0 and run_init_validate:
        psnr_old, psnr_new, psnr_all = base.validate_stage(
            teacher, dino, cgcd, val_datasets, device, 0, cfg, writer, rank, world_size, amp_dtype
        )
        if rank == 0:
            print(f"  Teacher validation: OLD={psnr_old:.2f}, NEW={psnr_new:.2f}, ALL={psnr_all:.2f}")
        dist.barrier()
    if use_real_eval and len(real_eval_datasets) > 0 and run_init_validate:
        base.evaluate_real_unpaired(
            teacher,
            dino,
            cgcd,
            real_eval_datasets,
            device,
            0,
            writer,
            rank,
            world_size,
            amp_dtype,
            max_eval_side=real_eval_max_side,
            allow_runtime_resize_retry=real_eval_allow_runtime_resize_retry,
            tb_step=0,
            step_label="Init Step 0",
            tb_prefix="RealEvalFull",
            log_prefix="RealEvalFull",
        )
        dist.barrier()

    if run_init_zero or run_init_copy or run_init_pseudo:
        init_batch_size = int(getattr(cfg.train, "init_pseudo_batch_size", cfg.train.batch_size))
        init_num_workers = int(getattr(cfg.train, "init_pseudo_num_workers", max(2, cfg.train.num_workers // 2)))
        init_pin_memory = bool(getattr(cfg.train, "init_pseudo_pin_memory", getattr(cfg.train, "pin_memory", True)))
        init_prefetch_factor = int(
            getattr(cfg.train, "init_pseudo_prefetch_factor", getattr(cfg.train, "prefetch_factor", 4))
        )
        init_persistent_workers = bool(
            getattr(cfg.train, "init_pseudo_persistent_workers", getattr(cfg.train, "persistent_workers", True))
        )
        rank_indices = list(range(rank, len(unlabeled_loader.dataset), world_size))
        init_subset = Subset(unlabeled_loader.dataset, rank_indices)
        init_loader_kwargs = {
            "batch_size": max(1, init_batch_size),
            "shuffle": False,
            "num_workers": max(0, init_num_workers),
            "pin_memory": init_pin_memory,
            "drop_last": False,
        }
        if init_num_workers > 0:
            init_loader_kwargs["prefetch_factor"] = max(1, init_prefetch_factor)
            init_loader_kwargs["persistent_workers"] = init_persistent_workers
        init_loader = DataLoader(init_subset, **init_loader_kwargs)
    else:
        init_batch_size = max(1, int(getattr(cfg.train, "init_pseudo_batch_size", cfg.train.batch_size)))
        init_subset = None
        init_loader = None

    if run_init_zero:
        if rank == 0:
            print(
                f"[InitZeroPseudo] Distributed zero-slot init enabled: world_size={world_size}, "
                f"per-rank samples~={len(init_subset)}, batch_size={init_batch_size}"
            )
        pseudo_bank_ops["initialize_zero_pseudo_labels"](init_loader, rank=rank, show_progress=(rank == 0))
        dist.barrier()

    if run_init_copy:
        if rank == 0:
            print(
                f"[CopyPseudo] Distributed copy enabled: world_size={world_size}, "
                f"per-rank samples~={len(init_subset)}, batch_size={init_batch_size}"
            )
        pseudo_bank_ops["copy_pseudo_labels"](init_loader, rank=rank, show_progress=(rank == 0))
        dist.barrier()

    if run_init_pseudo:
        if rank == 0:
            print(
                f"[InitPseudo] Distributed init enabled: world_size={world_size}, "
                f"per-rank samples~={len(init_subset)}, batch_size={init_batch_size}"
            )
        pseudo_bank_ops["initialize_pseudo_labels"](
            teacher,
            dino,
            cgcd.module,
            init_loader,
            device,
            dino_transform=base.transform_resize,
            amp_dtype=amp_dtype,
            rank=rank,
            show_progress=(rank == 0),
        )
        dist.barrier()

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        labeled_sampler.set_epoch(epoch)
        unlabeled_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(
                f"[Stage {current_stage}] Epoch {epoch}/{cfg.train.epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            print(f"{'='*60}")

        if use_validation and epoch % val_freq == 0 and len(val_datasets) > 0:
            psnr_old, psnr_new, psnr_all = base.validate_stage(
                teacher, dino, cgcd.module, val_datasets, device, epoch, cfg, writer, rank, world_size, amp_dtype
            )
            is_best = rank == 0 and psnr_all > best_psnr
            if is_best:
                best_psnr = psnr_all
        else:
            is_best = False

        resume_skip_steps_this_epoch = 0
        if resume_start_epoch_override > 0 and epoch == resume_start_epoch_override:
            resume_skip_steps_this_epoch = int(getattr(cfg.train, "resume_skip_steps", 0) or 0)

        global_step, train_loss, pseudo_update_cnt = train_one_epoch(
            student,
            teacher,
            dino,
            cgcd,
            iqa_metrics,
            labeled_loader,
            unlabeled_loader,
            optimizer,
            scaler,
            device,
            epoch,
            cfg,
            writer,
            global_step,
            rank,
            amp_dtype,
            pseudo_update_mode=cfg.train.pseudo_update_mode,
            musiq_warmup_epochs=musiq_warmup_epochs,
            loss_str=loss_str,
            loss_per=loss_per,
            checkpoint_step_callback=_maybe_save_step_checkpoint if save_step_freq > 0 else None,
            pseudo_log_step_callback=_maybe_log_pseudo_labels_step if pseudo_log_freq_steps > 0 else None,
            validation_step_callback=_maybe_run_step_validation if val_step_freq > 0 else None,
            reliable_pseudo_fn=pseudo_bank_ops["get_reliable"],
            use_real_eval=use_real_eval,
            real_eval_datasets=real_eval_step_datasets,
            real_eval_freq_steps=real_eval_freq_steps,
            real_eval_max_side=real_eval_max_side,
            real_eval_allow_runtime_resize_retry=real_eval_allow_runtime_resize_retry,
            world_size=world_size,
            resume_skip_steps=resume_skip_steps_this_epoch,
        )

        scheduler.step()

        if use_real_eval and real_eval_freq > 0 and epoch % real_eval_freq == 0 and len(real_eval_datasets) > 0:
            base.evaluate_real_unpaired(
                teacher,
                dino,
                cgcd.module,
                real_eval_datasets,
                device,
                epoch,
                writer,
                rank,
                world_size,
                amp_dtype,
                max_eval_side=real_eval_max_side,
                allow_runtime_resize_retry=real_eval_allow_runtime_resize_retry,
                tb_step=int(global_step),
                step_label=f"Epoch {epoch} / Step {int(global_step)}",
                tb_prefix="RealEvalFull",
                log_prefix="RealEvalFull",
            )
            dist.barrier()

        if paired_eval_enabled and paired_eval_freq > 0 and epoch % paired_eval_freq == 0:
            _run_paired_eval(epoch, global_step, f"Epoch {epoch} / Step {global_step}")

        if rank == 0 and writer is not None:
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Train/train_loss", train_loss, epoch)
            writer.add_scalar("Train/pseudo_update_cnt_epoch", pseudo_update_cnt, epoch)

        if pseudo_log_freq_steps <= 0:
            log_pseudo_labels_wres_semiuir(
                cfg.train.pseudo_patches_dir,
                writer,
                epoch,
                rank,
                global_step=global_step,
                reference_patches_dir=getattr(cfg.train, "lq_patches_dir", getattr(cfg.train, "real_train_root", None)),
            )
            if rank == 0 and writer is not None:
                writer.flush()

        if rank == 0:
            state = _build_checkpoint_state(epoch_value=epoch, global_step_value=global_step, is_best_value=is_best)
            if epoch % save_freq == 0 or epoch == cfg.train.epochs:
                save_path = os.path.join(checkpoint_dir, f"stage{current_stage}_epoch_{epoch}.pth")
                torch.save(state, save_path)
                print(f"Saved checkpoint: {save_path}")
            latest_path = os.path.join(checkpoint_dir, f"stage{current_stage}_latest.pth")
            torch.save(state, latest_path)
            if save_step_freq > 0:
                is_final_epoch = epoch == cfg.train.epochs
                is_final_step = max_train_steps > 0 and global_step >= max_train_steps
                if is_final_epoch or is_final_step:
                    save_path = os.path.join(checkpoint_dir, f"stage{current_stage}_step_{int(global_step)}.pth")
                    if not os.path.exists(save_path):
                        torch.save(state, save_path)
                        print(f"Saved checkpoint: {save_path}")

        dist.barrier()

        if max_train_steps > 0 and global_step >= max_train_steps:
            if rank == 0:
                print(f"[WRES-SemiUIR] Reached max_train_steps={max_train_steps}. Stop training.")
            break

    if rank == 0:
        print(f"\n[Stage {current_stage}] Training completed! Best PSNR: {best_psnr:.2f} dB")
        if writer is not None:
            writer.close()
    dist.destroy_process_group()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Stage N: Incremental OneRestore (Semi-UIR style)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Experiment name (미지정 시 config 파일명 + stage로 자동 생성)"
    )
    parser.add_argument("--stage", type=int, default=1, help="Current incremental stage")
    parser.add_argument("--inc_class_num", type=int, default=1, help="Number of NEW classes")
    parser.add_argument("--base_class_num", type=int, default=5, help="Number of OLD classes")
    parser.add_argument("--init_pseudo_label", action="store_true", help="Initialize pseudo labels with teacher")
    parser.add_argument(
        "--copy_pseudo_label",
        action="store_true",
        help="Materialize pseudo bank by copying unlabeled inputs into pseudo slots",
    )
    parser.add_argument(
        "--skip_pseudo_init",
        action="store_true",
        help="Skip python-side pseudo init/copy/zero loop (launcher prepopulates the bank externally)",
    )
    parser.add_argument("--init_validate", action="store_false")
    parser.add_argument(
        "--init_validate_on_resume",
        action="store_true",
        default=None,
        help="Run initial validation at epoch 0 even when resuming from checkpoint",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Start student from scratch while loading teacher/cgcd from previous-stage checkpoint",
    )
    parser.add_argument(
        "--pseudo_update_mode",
        type=str,
        default="musiq",
        choices=["musiq", "fgresq", "cgcd", "alternate", "musiq_then_cgcd"],
        help="Pseudo label 업데이트 전략: musiq(MUSIQ만), fgresq(FGResQ만), cgcd(CGCD만), alternate(번갈아), musiq_then_cgcd(MUSIQ 후 reject만 CGCD anchor fallback)",
    )
    parser.add_argument(
        "--musiq_warmup_epochs",
        type=int,
        default=0,
        help="초반 N epoch 동안 MUSIQ로 pseudo 업데이트 후 pseudo_update_mode로 전환 (-1이면 전체 epoch의 절반)",
    )
    parser.add_argument(
        "--cgcd_anchor_clear_weight",
        type=float,
        default=None,
        help="CGCD anchor score의 clear distance 가중치",
    )
    parser.add_argument(
        "--cgcd_anchor_neg_weight",
        type=float,
        default=None,
        help="CGCD anchor score의 negative distance 가중치",
    )
    parser.add_argument(
        "--cgcd_anchor_softmin_tau",
        type=float,
        default=None,
        help="CGCD anchor score의 softmin tau (0이면 hard min)",
    )
    parser.add_argument(
        "--pseudo_update_margin",
        type=float,
        default=None,
        help="pseudo 업데이트 조건 margin: teacher > max(student, reference) + margin",
    )
    parser.add_argument(
        "--pseudo_update_cgcd_fallback_margin",
        type=float,
        default=None,
        help="musiq_then_cgcd fallback에서 teacher > max(student, reference) + margin 조건에 쓰는 CGCD margin",
    )
    parser.add_argument(
        "--cgcd_score_mode",
        type=str,
        default=None,
        help="CGCD score mode override (config train.cgcd_score_mode)",
    )
    args = parser.parse_args()
    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    if args.exp_name is None:
        args.exp_name = f"{config_basename}_stage{args.stage}"
        print(f"[INFO] Auto-generated exp_name: {args.exp_name}")

    auto_pseudo_dir = os.path.join("../outputs/pseudo_labels", args.exp_name)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cfg = dict2namespace(config)
    if not hasattr(cfg, "incremental") or cfg.incremental is None:
        cfg.incremental = argparse.Namespace()
    cfg.exp_name = args.exp_name
    cfg.incremental.stage = args.stage
    cfg.incremental.inc_class_num = args.inc_class_num
    cfg.incremental.base_class_num = args.base_class_num
    cfg.train.init_pseudo_label = args.init_pseudo_label
    cfg.train.copy_pseudo_label = args.copy_pseudo_label
    cfg.train.skip_pseudo_init = args.skip_pseudo_init
    cfg.train.resume = args.resume
    cfg.train.init_validate = args.init_validate
    if args.init_validate_on_resume is not None:
        cfg.train.init_validate_on_resume = args.init_validate_on_resume
    cfg.train.scratch = args.scratch
    cfg.train.pseudo_update_mode = args.pseudo_update_mode
    cfg.train.musiq_warmup_epochs = args.musiq_warmup_epochs
    if args.cgcd_anchor_clear_weight is not None:
        cfg.train.cgcd_anchor_clear_weight = args.cgcd_anchor_clear_weight
    if args.cgcd_anchor_neg_weight is not None:
        cfg.train.cgcd_anchor_neg_weight = args.cgcd_anchor_neg_weight
    if args.cgcd_anchor_softmin_tau is not None:
        cfg.train.cgcd_anchor_softmin_tau = args.cgcd_anchor_softmin_tau
    if args.pseudo_update_margin is not None:
        cfg.train.pseudo_update_margin = args.pseudo_update_margin
    if args.pseudo_update_cgcd_fallback_margin is not None:
        cfg.train.pseudo_update_cgcd_fallback_margin = args.pseudo_update_cgcd_fallback_margin
    if args.cgcd_score_mode is not None:
        cfg.train.cgcd_score_mode = args.cgcd_score_mode
    if not getattr(cfg.train, "pseudo_patches_dir", ""):
        cfg.train.pseudo_patches_dir = auto_pseudo_dir
        print(f"[INFO] Auto-generated pseudo_patches_dir: {cfg.train.pseudo_patches_dir}")

    checkpoint_dir_template, prev_stage, base_checkpoint_path, config_parent_dir = base.precompute_checkpoint_path(
        cfg, args, config_basename
    )
    cfg.train.checkpoint_dir_template = checkpoint_dir_template
    cfg.train.prev_stage = prev_stage
    cfg.train.base_checkpoint_path = base_checkpoint_path
    cfg.train.config_parent_dir = config_parent_dir

    train_stage(cfg)


if __name__ == "__main__":
    main()
