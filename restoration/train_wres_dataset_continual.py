"""OneRestore + CGCD training and evaluation utilities."""

import os
import re
import sys
import math
import warnings

warnings.filterwarnings("ignore")
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import make_grid

import pyiqa

import cgcd.models as cgcd_models
import cgcd.models_margins as cgcd_models_margins
import cgcd.models_cov as cgcd_models_cov

CGCD_ARCH_REGISTRY = {
    "soft": "CGCDSignalModuleSoft",
    "softmargin": "CGCDSignalModuleSoftMargin",
    "hard": "CGCDSignalModuleHard",
    "hard_le": "CGCDSignalModuleStaticHardLeanableEmbedding",
    "staticsoft": "CGCDSignalModuleStaticSoft",
    "statichard": "CGCDSignalModuleStaticHard",
    "prompt": "CGCDSignalModulePrompt",
    "learnable": "CGCDSignalModuleLearnable",
    "adain": "CGCDSignalModuleControlNet",
    "mmdit": "CGCDSignalModuleMMdit",
    "adain_mmdit": "CGCDSignalModuleMMdit",
}

from models.OneRestore import OneRestore
import datasets_wres_continual as datasets_wres
from datasets_wres_continual import (
    create_labeled_dataset,
    create_real_eval_datasets,
    create_stage1_unlabeled_dataset,
    create_stage1_val_datasets,
    init_from_config,
)
from utils_lib.utils import dict2namespace
from utils_lib.helper import (
    create_fgresq_metric,
    freeze_teachers_parameters,
    get_cgcd_score,
    get_cgcd_score_ver2,
    get_fgresq_score,
    get_current_consistency_weight,
    get_musiq_score,
    get_reliable,
    initialize_pseudo_labels,
    copy_pseudo_labels,
    setup_ddp,
    setup_runtime,
)
from metrics import pt_psnr, pt_ssim
from feature_extractor.sl_finetuned_model import load_finetuned_model_from_checkpoint
from utils_lib.utils_incremental import print_incremental_info
from utils_lib.utils_dataset_wres import get_train_val_data_for_stage


def configure_torch_extensions_dir():
    """
    Configure per-rank torch extension build directory.
    Use BASE_EXT_DIR to avoid all ranks writing to a single extension cache dir.
    Also auto-fix TORCH_EXTENSIONS_DIR ending with "_<num>" to match LOCAL_RANK.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    base_ext_dir = os.environ.get("BASE_EXT_DIR")
    if base_ext_dir:
        os.environ["TORCH_EXTENSIONS_DIR"] = f"{base_ext_dir}_{local_rank}"
        return

    current_ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not current_ext_dir:
        return

    if re.search(r"_\d+$", current_ext_dir):
        prefix = current_ext_dir.rsplit("_", 1)[0]
        os.environ["TORCH_EXTENSIONS_DIR"] = f"{prefix}_{local_rank}"


configure_torch_extensions_dir()


# stage0 학습 코드(train_onerestore_soft_256.py)와 동일한 transform
transform_resize = v2.Compose(
    [
        v2.Resize([224, 224]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Keep a fixed pseudo-image subset per class so TensorBoard sample slots remain comparable over time.
_PSEUDO_LOG_FIXED_SAMPLES = {}


def get_cfg_cgcd_value(cfg, key, default):
    cgcd_cfg = getattr(cfg, "cgcd", None)
    if cgcd_cfg is not None and hasattr(cgcd_cfg, key):
        return getattr(cgcd_cfg, key)
    train_cfg = getattr(cfg, "train", None)
    if train_cfg is not None and hasattr(train_cfg, key):
        return getattr(train_cfg, key)
    return default


@torch.no_grad()
def update_teacher_ema_all_params(teacher_model, student_model, alpha=0.996, global_step=0):
    """
    Teacher EMA 업데이트 (모든 파라미터 대상).
    helper.update_teacher_ema()는 .lora_params만 업데이트하므로 wo_lora용으로 별도 구현.
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    teacher = teacher_model.module if hasattr(teacher_model, "module") else teacher_model
    student = student_model.module if hasattr(student_model, "module") else student_model

    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.mul_(alpha).add_(s_param, alpha=1 - alpha)


# @torch.no_grad()
# def update_teacher_cosine_ema_all_params(teacher_model, student_model, global_step=0, total_steps=40000):
#     base_alpha = 0.99   # 초반: Student의 변화를 1%씩 적극적으로 흡수 (팍팍 변함)
#     max_alpha = 0.999   # 후반: 0.1%만 흡수하여 Teacher 붕괴 방지 (무겁게 고정)

#     # Cosine 수식을 이용해 현재 스텝에 맞는 alpha 계산
#     if global_step >= total_steps:
#         current_alpha = max_alpha
#     else:
#         cosine_val = math.cos(math.pi * global_step / total_steps)
#         current_alpha = max_alpha - (max_alpha - base_alpha) * (cosine_val + 1) / 2

#     # 모델 파라미터 업데이트
#     student = student_model.module if hasattr(student_model, "module") else student_model
#     for t_param, s_param in zip(teacher_model.parameters(), student.parameters()):
#         t_param.data.mul_(current_alpha).add_(s_param.data, alpha=1 - current_alpha)


@torch.no_grad()
def log_pseudo_labels_wres(pseudo_patches_dir, writer, epoch, rank, global_step=None, reference_patches_dir=None):
    """Log pseudo-label samples for current NEW classes using a stable filename reference list."""
    if rank != 0 or writer is None:
        return

    new_classes = datasets_wres.NEW_CLASSES
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

        reference_images = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(".png")])
        if not reference_images:
            continue

        cache_key = (os.path.abspath(reference_dir), deg_name)
        fixed_samples = _PSEUDO_LOG_FIXED_SAMPLES.get(cache_key)
        if fixed_samples is None:
            num_samples = min(20, len(reference_images))
            if num_samples == 1:
                fixed_samples = [reference_images[0]]
            else:
                step = (len(reference_images) - 1) / float(num_samples - 1)
                indices = [int(round(i * step)) for i in range(num_samples)]
                fixed_samples = [reference_images[idx] for idx in indices]
            _PSEUDO_LOG_FIXED_SAMPLES[cache_key] = fixed_samples

        selected_images = fixed_samples
        image_tensors = []
        missing_images = []
        unreadable_images = []
        for image_name in selected_images:
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


def _next_batch_with_cycle(loader_iter, loader_obj, should_cycle, loader_name):
    """Fetch next batch and optionally restart iterator when exhausted."""
    try:
        batch = next(loader_iter)
    except StopIteration:
        if not should_cycle:
            raise RuntimeError(
                f"{loader_name} exhausted before epoch end. "
                "Set train.step_anchor to 'min' or enable cycling via 'labeled'/'unlabeled'/'max'."
            )
        loader_iter = iter(loader_obj)
        batch = next(loader_iter)
    return batch, loader_iter


def _prefetch_train_batch_to_device(labeled_batch, unlabeled_batch, device, transfer_stream):
    """Move labeled/unlabeled CPU batches to GPU on the transfer stream."""
    hq_labeled_cpu, lq_labeled_cpu, gt_deg_label_cpu = labeled_batch
    lq_unlabeled_cpu, pseudo_list_cpu, pseudo_names_cpu = unlabeled_batch
    with torch.cuda.stream(transfer_stream):
        hq_labeled_gpu = hq_labeled_cpu.to(device, non_blocking=True)
        lq_labeled_gpu = lq_labeled_cpu.to(device, non_blocking=True)
        gt_deg_label_gpu = gt_deg_label_cpu.to(device, non_blocking=True)
        lq_unlabeled_gpu = lq_unlabeled_cpu.to(device, non_blocking=True)
        pseudo_list_gpu = pseudo_list_cpu.to(device, non_blocking=True)
    return hq_labeled_gpu, lq_labeled_gpu, gt_deg_label_gpu, lq_unlabeled_gpu, pseudo_list_gpu, pseudo_names_cpu


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
    init_from_config(config_dict)
    if rank == 0:
        print_incremental_info(config_dict)

    train_data_old, _, val_data_all = get_train_val_data_for_stage(config_dict)

    labeled_dataset = create_labeled_dataset(train_data_old, cfg.train.patch_size)

    unlabeled_common_kwargs = dict(
        real_train_root=getattr(cfg.train, "real_train_root", cfg.train.data_root_train),
        lq_patches_dir=cfg.train.lq_patches_dir,
        pseudo_patches_dir=cfg.train.pseudo_patches_dir,
        patch_size=cfg.train.patch_size,
        patch_stride=getattr(cfg.train, "patch_stride", None),
        force_patchify=bool(getattr(cfg.train, "force_patchify", False)),
        max_patches_per_image=int(getattr(cfg.train, "max_patches_per_image", 0) or 0),
        seed=cfg.train.seed,
        clean_invalid_patches=bool(getattr(cfg.train, "verify_patch_cache", False)),
        show_progress=bool(getattr(cfg.train, "show_patchify_progress", True)),
        skip_cache_if_exists=bool(getattr(cfg.train, "skip_patchify_if_exists", True)),
    )

    # Avoid race conditions: only rank0 builds/writes patch cache, others reuse after barrier.
    if rank == 0:
        unlabeled_dataset = create_stage1_unlabeled_dataset(
            **unlabeled_common_kwargs,
            build_cache=True,
        )
    dist.barrier()
    if rank != 0:
        unlabeled_dataset = create_stage1_unlabeled_dataset(
            **unlabeled_common_kwargs,
            build_cache=False,
        )

    use_validation = bool(getattr(cfg.train, "use_validation", False))
    if use_validation:
        val_datasets = create_stage1_val_datasets(val_data_all)
    else:
        val_datasets = []
        if rank == 0:
            print("[WRES-Continual] Validation disabled (train.use_validation=False)")

    use_real_eval = bool(getattr(cfg.train, "use_real_eval", False))
    if use_real_eval:
        real_eval_datasets = create_real_eval_datasets(
            getattr(cfg.train, "real_eval_root", ""),
            list(getattr(cfg.train, "real_eval_sets", [])),
        )
    else:
        real_eval_datasets = []
        if rank == 0:
            print("[WRES-Continual] Real eval disabled (train.use_real_eval=False)")

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
            "[WRES-Continual] Train loader config: "
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


def build_models(cfg, device, rank):
    # 1. DINO (Frozen) - config에서 model_name 읽기
    dino_model = load_finetuned_model_from_checkpoint(
        checkpoint_dir=cfg.cgcd.dino_checkpoint,
        num_classes=cfg.cgcd.nclasses,
        model_name=cfg.cgcd.model_name,
        device=device,
    )
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False

    # 2. CGCD - CGCD_ARCH_REGISTRY로 동적 선택
    cgcd_arch = getattr(cfg.cgcd, "arch", "adain")
    cgcd_class_name = CGCD_ARCH_REGISTRY[cgcd_arch]
    CGCDClass = (
        getattr(cgcd_models, cgcd_class_name, None)
        or getattr(cgcd_models_margins, cgcd_class_name, None)
        or getattr(cgcd_models_cov, cgcd_class_name, None)
    )
    if CGCDClass is None:
        raise ValueError(f"Unsupported cgcd arch '{cgcd_arch}'")
    if rank == 0:
        print(f"CGCD arch: {cgcd_arch} -> {cgcd_class_name}")

    cgcd_model = CGCDClass(
        saved_models_dir=cfg.cgcd.saved_vcgcd_models_dir,
        output_dim=cfg.cgcd.embd_dim,
        pca_path=cfg.cgcd.pca_path,
        scaler_path=getattr(cfg.cgcd, "scaler_path", None),
        stage=cfg.incremental.stage,
    ).to(device)

    # 3. Student & Teacher (OneRestore) - LoRA 없이 직접 fine-tuning
    student_model = OneRestore(channel=cfg.model.width).to(device)
    teacher_model = OneRestore(channel=cfg.model.width).to(device)

    for param in student_model.parameters():
        param.requires_grad = True
    freeze_teachers_parameters(teacher_model)

    if rank == 0:
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in student_model.parameters())
        print(f"\n{'='*60}")
        print(f"OneRestore Student Model (NO LoRA):")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio:      {100 * trainable_params / total_params:.2f}%")
        print(f"{'='*60}\n")

    return student_model, teacher_model, dino_model, cgcd_model


def load_checkpoints(student_model, teacher_model, cgcd_model, optimizer, scheduler, cfg, device, rank):
    start_epoch, global_step, best_psnr = 1, 0, 0.0
    current_stage = cfg.incremental.stage
    scratch_mode = getattr(cfg.train, "scratch", False)
    checkpoint_loaded = False

    # 1. Resume (우선순위 최상)
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        if rank == 0:
            print(f"==> Resuming from checkpoint: {cfg.train.resume}")
        checkpoint = torch.load(cfg.train.resume, map_location=device)

        student_model.load_state_dict(checkpoint["student_state_dict"])
        teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
        cgcd_model.load_state_dict(checkpoint["cgcd_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_psnr = checkpoint.get("best_psnr", 0.0)
        checkpoint_loaded = True

        if rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}, global_step {global_step}")

        return start_epoch, global_step, best_psnr, checkpoint_loaded

    # 2. 이전 Stage에서 로드 (main()에서 미리 계산한 경로 사용)
    if current_stage >= 1:
        prev_stage = getattr(cfg.train, "prev_stage", None)
        base_path = getattr(cfg.train, "base_checkpoint_path", None)

        if prev_stage is None:
            raise ValueError("cfg.train.prev_stage is not set. Call precompute_checkpoint_path() in main() first.")
        if not base_path:
            raise ValueError(
                "cfg.train.base_checkpoint_path is not set. Call precompute_checkpoint_path() in main() first."
            )

        if os.path.exists(base_path):
            from utils_lib.utils import load_teacher_wo_lora

            if rank == 0:
                print(f"==> Loading Stage {prev_stage} checkpoint from: {base_path}")
                if scratch_mode:
                    print("==> Scratch mode enabled: student is randomly initialized, teacher loads checkpoint")

            load_teacher_wo_lora(
                checkpoint_path=base_path,
                student_model=None if scratch_mode else student_model,
                teacher_model=teacher_model,
                cgcd_model=cgcd_model,
            )
            checkpoint_loaded = True

            if rank == 0:
                if scratch_mode:
                    print(f"Checkpoint loading completed (Stage {prev_stage}, scratch student)")
                else:
                    print(f"Checkpoint loading completed (Stage {prev_stage})")
        else:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"[WARNING] Checkpoint not found at {base_path}")
                if current_stage == 1:
                    print(f"[WARNING] Stage 1 should load from Stage 0 checkpoint!")
                print(f"[WARNING] Starting from scratch")
                print(f"{'='*60}\n")
    else:
        if rank == 0:
            print(f"==> Stage {current_stage} - Training from scratch")

    return start_epoch, global_step, best_psnr, checkpoint_loaded


def precompute_checkpoint_path(cfg, args, config_basename):

    config_parent_dir = os.path.basename(os.path.dirname(args.config))
    checkpoint_dir_template = getattr(cfg.train, "checkpoint_dir_template", "")
    if checkpoint_dir_template == "auto" or checkpoint_dir_template == "":

        base_checkpoint_dir = "../outputs/checkpoints"
        stage_template_name = f"{config_basename}_stage{{stage}}"

        if config_parent_dir and config_parent_dir != "configs":
            checkpoint_dir_template = os.path.join(base_checkpoint_dir, config_parent_dir, stage_template_name)
        else:
            checkpoint_dir_template = os.path.join(base_checkpoint_dir, stage_template_name)

        print(f"[INFO] Auto-generated config_parent_dir: {config_parent_dir}")
        print(f"[INFO] Auto-generated stage_template_name: {stage_template_name}")
        print(f"[INFO] Auto-generated checkpoint_dir_template: {checkpoint_dir_template}")

    current_stage = cfg.incremental.stage
    if current_stage >= 1:
        prev_stage = 0 if current_stage == 1 else current_stage - 1
        stage0_ckpt = getattr(cfg.train, "stage0_checkpoint", None)
        previous_stage_ckpt = str(getattr(cfg.train, "previous_stage_checkpoint", "") or "").strip()
        checkpoint_dir = None
        checkpoint_name = None

        # Priority rule:
        # Explicit previous-stage checkpoint -> use it first.
        # Stage 1 + explicit stage0_checkpoint in YAML -> use it first.
        # Otherwise -> use precomputed stage(N-1) checkpoint path.
        if previous_stage_ckpt:
            base_path = previous_stage_ckpt
            print(f"[INFO] Using cfg.train.previous_stage_checkpoint: {base_path}")
        elif current_stage == 1 and stage0_ckpt:
            base_path = stage0_ckpt
            print(f"[INFO] Stage 1: using cfg.train.stage0_checkpoint with higher priority: {base_path}")
        else:
            checkpoint_dir = checkpoint_dir_template.format(stage=prev_stage)
            checkpoint_name = cfg.train.checkpoint_name.format(stage=prev_stage)
            base_path = os.path.join(checkpoint_dir, checkpoint_name)
            print(f"[INFO] Precomputed base checkpoint_dir: {checkpoint_dir}")
            print(f"[INFO] Precomputed base checkpoint_name: {checkpoint_name}")

        print(f"[INFO] Precomputed base checkpoint path: {base_path}")

    else:
        prev_stage = -1
        base_path = None

    return checkpoint_dir_template, prev_stage, base_path, config_parent_dir


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
    cgcd_score_mode="clear",
    musiq_warmup_epochs=0,
):
    current_stage = cfg.incremental.stage

    teacher_model.eval()
    freeze_teachers_parameters(teacher_model)

    student_model.train()
    cgcd_model.train()

    criterion_l1 = nn.L1Loss()
    criterion_ce = nn.CrossEntropyLoss()

    total_updated_cnt = 0
    running_loss_sup = 0.0
    running_loss_unsup = 0.0
    running_loss_total = 0.0
    running_psnr_labeled = 0.0

    transfer_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    labeled_steps = len(labeled_loader)
    unlabeled_steps = len(unlabeled_loader)
    step_anchor = str(getattr(cfg.train, "step_anchor", "unlabeled")).lower()
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

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    if rank == 0:
        print(
            "[WRES-Continual] Step scheduler: "
            f"anchor={step_anchor}, steps={num_steps}, "
            f"labeled_steps={labeled_steps}, unlabeled_steps={unlabeled_steps}, "
            f"cycle_labeled={cycle_labeled}, cycle_unlabeled={cycle_unlabeled}"
        )

    use_tqdm = rank == 0
    pbar = (
        tqdm(range(num_steps), desc=f"Epoch {epoch}", dynamic_ncols=True, mininterval=0.5)
        if use_tqdm
        else range(num_steps)
    )

    rampup_w = get_current_consistency_weight(epoch, cfg.train.consistency_weight, cfg.train.rampup_epoch)
    effective_update_mode = "musiq" if epoch <= musiq_warmup_epochs else pseudo_update_mode
    cgcd_contrastive_pos_weight = float(get_cfg_cgcd_value(cfg, "cgcd_contrastive_pos_weight", 1.0))
    cgcd_contrastive_neg_weight = float(get_cfg_cgcd_value(cfg, "cgcd_contrastive_neg_weight", 1.0))
    cgcd_contrastive_tau = float(get_cfg_cgcd_value(cfg, "cgcd_contrastive_tau", 1.0))
    cgcd_contrastive_score_temp = float(get_cfg_cgcd_value(cfg, "cgcd_contrastive_score_temp", 1.0))
    cgcd_mahalanobis_temp = float(get_cfg_cgcd_value(cfg, "cgcd_mahalanobis_temp", 20.0))
    cgcd_mahalanobis_pca_top_k = get_cfg_cgcd_value(cfg, "cgcd_mahalanobis_pca_top_k", 32)
    if cgcd_mahalanobis_pca_top_k is not None:
        cgcd_mahalanobis_pca_top_k = int(cgcd_mahalanobis_pca_top_k)
    pseudo_update_margin = float(get_cfg_cgcd_value(cfg, "pseudo_update_margin", 0.0))

    labeled_batch_cpu, labeled_iter = _next_batch_with_cycle(
        labeled_iter, labeled_loader, cycle_labeled, "labeled_loader"
    )
    unlabeled_batch_cpu, unlabeled_iter = _next_batch_with_cycle(
        unlabeled_iter,
        unlabeled_loader,
        cycle_unlabeled,
        "unlabeled_loader",
    )
    prefetched_batch = _prefetch_train_batch_to_device(labeled_batch_cpu, unlabeled_batch_cpu, device, transfer_stream)

    for i in pbar:
        compute_stream.wait_stream(transfer_stream)
        hq_labeled, lq_labeled, gt_deg_label, lq_unlabeled, pseudo_list, pseudo_names = prefetched_batch

        # DINO feature (frozen)
        with torch.no_grad():
            with autocast("cuda", dtype=amp_dtype):
                # Single DINO forward for both labeled/unlabeled batches to reduce per-step overhead.
                batch_labeled = lq_labeled.shape[0]
                lq_merged = torch.cat([lq_labeled, lq_unlabeled], dim=0)
                dino_feat_merged = dino_model(transform_resize(lq_merged)).pooler_output
                dino_feat_labeled = dino_feat_merged[:batch_labeled]
                dino_feat_unlabeled = dino_feat_merged[batch_labeled:]

        with autocast("cuda", dtype=amp_dtype):
            # A. Labeled Supervised Loss
            embedding_labeled, pseudo_class_labeled = cgcd_model(dino_feat_labeled)
            restored_labeled = student_model(lq_labeled, embedding_labeled)

            loss_l1_labeled = cfg.train.l1_weight * criterion_l1(restored_labeled, hq_labeled)
            loss_ce_labeled = cfg.train.ce_weight * criterion_ce(pseudo_class_labeled, gt_deg_label)
            loss_sup = loss_l1_labeled + loss_ce_labeled

            # B. Unlabeled Loss
            embedding_unlabeled, _ = cgcd_model(dino_feat_unlabeled)
            student_output_unlabeled = student_model(lq_unlabeled, embedding_unlabeled)

            with torch.no_grad():
                teacher_output_unlabeled = teacher_model(lq_unlabeled, embedding_unlabeled)
                if effective_update_mode == "fgresq":
                    iqa_metric = iqa_metrics.get("fgresq")
                    if iqa_metric is None:
                        raise ValueError("FGResQ metric is required but not initialized.")
                    score_reference = get_fgresq_score(iqa_metric, pseudo_list)
                elif effective_update_mode == "cgcd":
                    iqa_metric = iqa_metrics.get("musiq")
                    score_reference = None
                else:
                    iqa_metric = iqa_metrics.get("musiq")
                    if iqa_metric is None:
                        raise ValueError("MUSIQ metric is required but not initialized.")
                    score_reference = get_musiq_score(iqa_metric, pseudo_list)

                orig2classifier = datasets_wres.ORIG2CLASSIFIER
                if orig2classifier is None:
                    raise ValueError("ORIG2CLASSIFIER is not initialized. Call init_from_config() first.")

                # cgcd_model이 DDP일 수 있으므로 .module 접근
                cgcd_inner = cgcd_model.module if hasattr(cgcd_model, "module") else cgcd_model

                reliable_pseudo_labels, updated_cnt, update_mode = get_reliable(
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
                    dino_transform=transform_resize,
                    epoch=epoch,
                    orig2classifier=orig2classifier,
                    cgcd_score_mode=cgcd_score_mode,
                    amp_dtype=amp_dtype,
                    contrastive_pos_weight=cgcd_contrastive_pos_weight,
                    contrastive_neg_weight=cgcd_contrastive_neg_weight,
                    contrastive_tau=cgcd_contrastive_tau,
                    contrastive_score_temp=cgcd_contrastive_score_temp,
                    mahalanobis_temp=cgcd_mahalanobis_temp,
                    mahalanobis_pca_top_k=cgcd_mahalanobis_pca_top_k,
                    update_margin=pseudo_update_margin,
                )
                total_updated_cnt += updated_cnt

            loss_unsup = criterion_l1(student_output_unlabeled, reliable_pseudo_labels)

            # C. Total Loss
            loss_total = loss_sup + (rampup_w * loss_unsup)

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

        # Teacher EMA (모든 파라미터)
        with torch.no_grad():
            update_teacher_ema_all_params(teacher_model, student_model, alpha=0.996, global_step=global_step)

        with torch.no_grad():
            psnr_labeled = pt_psnr(hq_labeled, restored_labeled).mean()

        running_loss_sup += loss_sup.detach().item()
        running_loss_unsup += loss_unsup.detach().item()
        running_loss_total += loss_total.detach().item()
        running_psnr_labeled += psnr_labeled.detach().item()

        if use_tqdm and (i + 1) % 10 == 0:
            pbar.set_description(
                f"[Stage {current_stage}] Epoch {epoch}/{cfg.train.epochs} | "
                f"Update mode: {effective_update_mode} (target: {pseudo_update_mode}), "
                f"Score mode: {cgcd_score_mode} | "
                f"Sup: {running_loss_sup/(i+1):.4f}, Unsup: {running_loss_unsup/(i+1):.4f}, "
                f"PSNR: {running_psnr_labeled/(i+1):.2f}, "
                f"updated: {total_updated_cnt}"
            )

        if rank == 0 and (i + 1) % cfg.train.print_freq == 0 and writer is not None:
            writer.add_scalar("train/loss_sup", loss_sup.detach().item(), global_step)
            writer.add_scalar("train/loss_unsup", loss_unsup.detach().item(), global_step)
            writer.add_scalar("train/loss_total", loss_total.detach().item(), global_step)
            writer.add_scalar("train/psnr", psnr_labeled.detach().item(), global_step)
            writer.add_scalar("train/psnr_labeled", psnr_labeled.detach().item(), global_step)
            writer.add_scalar("train/pseudo_update_cnt", total_updated_cnt, global_step)

        if i + 1 < num_steps:
            labeled_batch_cpu, labeled_iter = _next_batch_with_cycle(
                labeled_iter, labeled_loader, cycle_labeled, "labeled_loader"
            )
            unlabeled_batch_cpu, unlabeled_iter = _next_batch_with_cycle(
                unlabeled_iter,
                unlabeled_loader,
                cycle_unlabeled,
                "unlabeled_loader",
            )
            next_prefetched_batch = _prefetch_train_batch_to_device(
                labeled_batch_cpu, unlabeled_batch_cpu, device, transfer_stream
            )
            prefetched_batch = next_prefetched_batch

        global_step += 1

    num_batches = num_steps
    avg_loss_total = running_loss_total / num_batches
    avg_loss_sup = running_loss_sup / num_batches
    avg_loss_unsup = running_loss_unsup / num_batches
    avg_psnr = running_psnr_labeled / num_batches

    if rank == 0:
        print(
            f"\n[Stage {current_stage}] Epoch {epoch} - "
            f"Update mode: {effective_update_mode} (target: {pseudo_update_mode}), Score mode: {cgcd_score_mode}, "
            f"Total: {avg_loss_total:.4f}, Sup: {avg_loss_sup:.4f}, Unsup: {avg_loss_unsup:.4f}, "
            f"PSNR: {avg_psnr:.2f} dB, Pseudo updated: {total_updated_cnt}"
        )

    return global_step, avg_loss_total, total_updated_cnt


@torch.no_grad()
def validate_stage(
    model,
    dino_model,
    cgcd_model,
    val_datasets,
    device,
    epoch,
    cfg,
    writer,
    rank,
    world_size,
    amp_dtype=torch.float16,
    tb_step=None,
    step_label=None,
):
    model.eval()
    dino_model.eval()
    cgcd_model.eval()

    musiq_metric = pyiqa.create_metric("musiq", as_loss=False, device=device)
    lpips_metric = pyiqa.create_metric("lpips", as_loss=False, device=device)

    new_classes = datasets_wres.NEW_CLASSES or []
    if datasets_wres.NEW_CLASSES is None and rank == 0:
        print("[WRES-Continual] Warning: NEW_CLASSES not initialized, treating all validation classes as OLD.")

    deg_types_per_gpu = []
    all_deg_names = [deg_name for _, deg_name in val_datasets]

    for i, (val_dataset, deg_name) in enumerate(val_datasets):
        if i % world_size == rank:
            deg_types_per_gpu.append((val_dataset, deg_name))

    current_stage = cfg.incremental.stage
    metric_step = int(tb_step) if tb_step is not None else int(epoch)
    display_step = str(step_label) if step_label is not None else f"Epoch {epoch}"
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"[Stage {current_stage}] Validation at {display_step}")
        print(f"{'=' * 60}")

    local_results = {}

    for val_dataset, deg_name in deg_types_per_gpu:
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, pin_memory=True
        )

        psnr_list, ssim_list, musiq_list, lpips_list = [], [], [], []
        class_type = "NEW" if deg_name in new_classes else "OLD"
        is_new_class = deg_name in new_classes

        iterator = tqdm(
            val_loader, desc=f"[GPU {rank}][{class_type}] {deg_name:<15}", leave=False, position=rank, ncols=100
        )

        for batch in iterator:
            hq_image = batch[0].to(device, non_blocking=True)
            lq_image = batch[1].to(device, non_blocking=True)

            with autocast("cuda", dtype=amp_dtype):
                dino_feature = dino_model(transform_resize(lq_image)).pooler_output
                cgcd_embd, _ = cgcd_model(dino_feature)
                restored = model(lq_image, cgcd_embd)

            restored_clamped = torch.clamp(restored, 0, 1)
            psnr_list.append(pt_psnr(hq_image, restored_clamped))
            ssim_list.append(pt_ssim(hq_image, restored_clamped))

            if is_new_class:
                musiq_list.append(musiq_metric(restored_clamped))
                lpips_list.append(lpips_metric(restored_clamped, hq_image))

        all_psnr = torch.cat(psnr_list, dim=0).mean().item()
        all_ssim = torch.cat(ssim_list, dim=0).mean().item()
        all_musiq = torch.cat(musiq_list, dim=0).mean().item() if is_new_class and musiq_list else 0.0
        all_lpips = torch.cat(lpips_list, dim=0).mean().item() if is_new_class and lpips_list else 0.0

        local_results[deg_name] = {
            "psnr": all_psnr,
            "ssim": all_ssim,
            "musiq": all_musiq,
            "lpips": all_lpips,
            "count": len(psnr_list),
            "is_new": is_new_class,
        }

        if is_new_class:
            print(
                f"[GPU {rank}][{class_type}] {deg_name:20s} - PSNR: {all_psnr:.4f}, SSIM: {all_ssim:.4f}, MUSIQ: {all_musiq:.4f}, LPIPS: {all_lpips:.4f}"
            )
        else:
            print(f"[GPU {rank}][{class_type}] {deg_name:20s} - PSNR: {all_psnr:.4f}, SSIM: {all_ssim:.4f}")

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        results = {}
        for rd in gathered_results:
            if rd:
                results.update(rd)

        old_psnr_sum, old_ssim_sum, old_count = 0.0, 0.0, 0
        new_psnr_sum, new_ssim_sum, new_musiq_sum, new_lpips_sum, new_count = 0.0, 0.0, 0.0, 0.0, 0
        all_psnr_sum, all_ssim_sum, all_count = 0.0, 0.0, 0

        print(f"\n{'=' * 80}")
        print(f"[Stage {current_stage}] Validation Results at {display_step}")
        print(f"{'=' * 80}")

        for deg_name in all_deg_names:
            if deg_name not in results:
                continue
            r = results[deg_name]
            class_type = "NEW" if r["is_new"] else "OLD"

            if r["is_new"]:
                print(
                    f"[{class_type}] {deg_name:20s} - PSNR: {r['psnr']:.4f}, SSIM: {r['ssim']:.4f}, MUSIQ: {r['musiq']:.4f}, LPIPS: {r['lpips']:.4f}"
                )
            else:
                print(f"[{class_type}] {deg_name:20s} - PSNR: {r['psnr']:.4f}, SSIM: {r['ssim']:.4f}")

            if writer:
                writer.add_scalar(f"Val_{deg_name}/PSNR", r["psnr"], metric_step)
                writer.add_scalar(f"Val_{deg_name}/SSIM", r["ssim"], metric_step)
                if r["is_new"]:
                    writer.add_scalar(f"Val_{deg_name}/MUSIQ", r["musiq"], metric_step)
                    writer.add_scalar(f"Val_{deg_name}/LPIPS", r["lpips"], metric_step)

            all_psnr_sum += r["psnr"]
            all_ssim_sum += r["ssim"]
            all_count += 1
            if r["is_new"]:
                new_psnr_sum += r["psnr"]
                new_ssim_sum += r["ssim"]
                new_musiq_sum += r["musiq"]
                new_lpips_sum += r["lpips"]
                new_count += 1
            else:
                old_psnr_sum += r["psnr"]
                old_ssim_sum += r["ssim"]
                old_count += 1

        avg_psnr_old = old_psnr_sum / old_count if old_count > 0 else 0.0
        avg_ssim_old = old_ssim_sum / old_count if old_count > 0 else 0.0
        avg_psnr_new = new_psnr_sum / new_count if new_count > 0 else 0.0
        avg_ssim_new = new_ssim_sum / new_count if new_count > 0 else 0.0
        avg_musiq_new = new_musiq_sum / new_count if new_count > 0 else 0.0
        avg_lpips_new = new_lpips_sum / new_count if new_count > 0 else 0.0
        avg_psnr_all = all_psnr_sum / all_count if all_count > 0 else 0.0
        avg_ssim_all = all_ssim_sum / all_count if all_count > 0 else 0.0

        print(f"{'-' * 80}")
        print(f"{'OLD Classes Avg':20s} - PSNR: {avg_psnr_old:.4f} dB, SSIM: {avg_ssim_old:.4f}")
        if new_count > 0:
            print(
                f"{'NEW Classes Avg':20s} - PSNR: {avg_psnr_new:.4f} dB, SSIM: {avg_ssim_new:.4f}, MUSIQ: {avg_musiq_new:.4f}, LPIPS: {avg_lpips_new:.4f}"
            )
        print(f"{'ALL Classes Avg':20s} - PSNR: {avg_psnr_all:.4f} dB, SSIM: {avg_ssim_all:.4f}")
        print(f"{'=' * 80}\n")

        if writer:
            writer.add_scalar("Val_OLD/Avg_PSNR", avg_psnr_old, metric_step)
            writer.add_scalar("Val_OLD/Avg_SSIM", avg_ssim_old, metric_step)
            if new_count > 0:
                writer.add_scalar("Val_NEW/Avg_PSNR", avg_psnr_new, metric_step)
                writer.add_scalar("Val_NEW/Avg_SSIM", avg_ssim_new, metric_step)
                writer.add_scalar("Val_NEW/Avg_MUSIQ", avg_musiq_new, metric_step)
                writer.add_scalar("Val_NEW/Avg_LPIPS", avg_lpips_new, metric_step)
            writer.add_scalar("Val_ALL/Avg_PSNR", avg_psnr_all, metric_step)
            writer.add_scalar("Val_ALL/Avg_SSIM", avg_ssim_all, metric_step)

        return avg_psnr_old, avg_psnr_new, avg_psnr_all
    else:
        return 0.0, 0.0, 0.0


@torch.no_grad()
def evaluate_real_unpaired(
    model,
    dino_model,
    cgcd_model,
    real_eval_datasets,
    device,
    epoch,
    writer,
    rank,
    world_size,
    amp_dtype=torch.float16,
    max_eval_side=1280,
    allow_runtime_resize_retry=True,
    include_cgcd_metric=False,
    cgcd_score_mode="clear",
    cgcd_contrastive_pos_weight=1.0,
    cgcd_contrastive_neg_weight=1.0,
    cgcd_contrastive_tau=1.0,
    cgcd_contrastive_score_temp=1.0,
    cgcd_mahalanobis_temp=20.0,
    cgcd_mahalanobis_pca_top_k=32,
    tb_step=None,
    step_label=None,
    tb_prefix="RealEval",
    log_prefix="RealEval",
):
    if not real_eval_datasets:
        return 0.0

    tb_prefix = str(tb_prefix or "RealEval").strip().rstrip("/") or "RealEval"
    log_prefix = str(log_prefix or "RealEval").strip() or "RealEval"

    model.eval()
    eval_model = model.module if hasattr(model, "module") else model
    if hasattr(eval_model, "set_runtime_resize_fallback"):
        eval_model.set_runtime_resize_fallback(True)
    dino_model.eval()
    cgcd_model.eval()
    metric_specs = [
        ("musiq", "MUSIQ", "MUSIQ"),
        ("clipiqa", "CLIP-IQA", "CLIPIQA"),
    ]
    iqa_metrics = {}
    active_metric_specs = []
    for metric_name, display_name, tb_name in metric_specs:
        try:
            iqa_metrics[metric_name] = pyiqa.create_metric(metric_name, as_loss=False, device=device)
            active_metric_specs.append((metric_name, display_name, tb_name, "pyiqa"))
        except Exception as e:
            if rank == 0:
                print(f"[{log_prefix}][Warning] Skip {display_name} ({metric_name}) - {e}")

    cgcd_mode_name = str(cgcd_score_mode).strip() or "clear"
    cgcd_mode_key = cgcd_mode_name.lower().replace("-", "_")
    if include_cgcd_metric:
        cgcd_tb_suffix = re.sub(r"[^0-9A-Za-z_]+", "_", cgcd_mode_key).strip("_") or "clear"
        active_metric_specs.append(("cgcd", f"CGCD-{cgcd_mode_name}", f"CGCD_{cgcd_tb_suffix}", "cgcd"))

    if len(active_metric_specs) == 0:
        raise RuntimeError("No real-eval metric could be initialized. Check pyiqa and CGCD settings.")

    clear_idx = 0
    if include_cgcd_metric:
        try:
            clear_idx = int(datasets_wres.ORIG2CLASSIFIER[0])
        except Exception:
            clear_idx = 0

    all_set_names = [set_name for _, set_name in real_eval_datasets]
    local_results = {}
    resize_retry_warned = False
    for eval_dataset, set_name in real_eval_datasets:
        total_count = len(eval_dataset)
        local_indices = list(range(rank, total_count, world_size))
        eval_subset = Subset(eval_dataset, local_indices)
        loader = DataLoader(
            eval_subset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            pin_memory=True,
        )

        metric_sums = {metric_name: 0.0 for metric_name, _, _, _ in active_metric_specs}
        image_count = 0
        use_tqdm = rank == 0 and sys.stderr.isatty()
        iterator = (
            tqdm(
                loader,
                desc=f"[GPU {rank}][{log_prefix}] {set_name:<18} ({len(local_indices)}/{total_count})",
                leave=False,
                ncols=100,
            )
            if use_tqdm
            else loader
        )
        for batch in iterator:
            lq_image = batch[0].to(device, non_blocking=True)
            lq_path = batch[1][0] if isinstance(batch, (list, tuple)) and len(batch) > 1 else ""
            image_count += int(lq_image.shape[0])
            if max_eval_side and max_eval_side > 0:
                _, _, h, w = lq_image.shape
                longest = max(h, w)
                if longest > max_eval_side:
                    scale = float(max_eval_side) / float(longest)
                    new_h = max(16, int((h * scale) // 16 * 16))
                    new_w = max(16, int((w * scale) // 16 * 16))
                    lq_image = F.interpolate(lq_image, size=(new_h, new_w), mode="bilinear", align_corners=False)

            restored = None
            cur_image = lq_image
            sample_name = os.path.basename(lq_path) if lq_path else f"{set_name}_idx{image_count-1}"
            os.environ["ONERESTORE_DEBUG_SAMPLE"] = sample_name
            retry_iters = 6 if allow_runtime_resize_retry else 1
            try:
                for _ in range(retry_iters):
                    try:
                        with autocast("cuda", dtype=amp_dtype):
                            dino_feature = dino_model(transform_resize(cur_image)).pooler_output
                            cgcd_embd, _ = cgcd_model(dino_feature)
                            restored = model(cur_image, cgcd_embd)
                        break
                    except RuntimeError as e:
                        msg = str(e)
                        recoverable = (
                            "Provided interpolation parameters can not be handled" in msg
                            or "Too much shared memory required" in msg
                            or "upsample_bilinear2d_aa" in msg
                        )
                        if (not allow_runtime_resize_retry) or (not recoverable):
                            raise
                        _, _, h, w = cur_image.shape
                        if min(h, w) <= 256:
                            raise
                        new_h = max(16, int((h * 0.85) // 16 * 16))
                        new_w = max(16, int((w * 0.85) // 16 * 16))
                        if new_h == h and new_w == w:
                            new_h = max(16, h - 16)
                            new_w = max(16, w - 16)
                        if not resize_retry_warned:
                            print(
                                f"[GPU {rank}][{log_prefix}][Warning] Resize retry due to shared-memory limit: "
                                f"{h}x{w} -> {new_h}x{new_w}"
                            )
                            resize_retry_warned = True
                        cur_image = F.interpolate(cur_image, size=(new_h, new_w), mode="bilinear", align_corners=False)
            finally:
                os.environ.pop("ONERESTORE_DEBUG_SAMPLE", None)
            if restored is None:
                raise RuntimeError("RealEval inference failed after resize retries.")

            restored = torch.clamp(restored, 0, 1)
            cgcd_value = None
            if include_cgcd_metric:
                cgcd_score_fn = (
                    get_cgcd_score_ver2
                    if cgcd_mode_key in {"mahalanobis_margin", "mahalanobis_pca"}
                    else get_cgcd_score
                )
                cgcd_kwargs = {
                    "mode": cgcd_mode_key,
                    "amp_dtype": amp_dtype,
                    "contrastive_pos_weight": cgcd_contrastive_pos_weight,
                    "contrastive_neg_weight": cgcd_contrastive_neg_weight,
                    "contrastive_tau": cgcd_contrastive_tau,
                    "contrastive_score_temp": cgcd_contrastive_score_temp,
                }
                if cgcd_score_fn is get_cgcd_score_ver2:
                    cgcd_kwargs["mahalanobis_temp"] = cgcd_mahalanobis_temp
                    cgcd_kwargs["mahalanobis_pca_top_k"] = cgcd_mahalanobis_pca_top_k
                cgcd_value = cgcd_score_fn(
                    restored,
                    dino_model,
                    cgcd_model,
                    transform_resize,
                    clear_idx,
                    **cgcd_kwargs,
                )
            for metric_name, _, _, metric_kind in active_metric_specs:
                if metric_kind == "cgcd":
                    metric_value = cgcd_value
                else:
                    metric_value = iqa_metrics[metric_name](restored)
                metric_sums[metric_name] += float(metric_value.squeeze().item())

        set_result = {"count": image_count, "total_count": total_count}
        for metric_name, _, _, _ in active_metric_specs:
            set_result[f"{metric_name}_sum"] = metric_sums[metric_name]
            set_result[metric_name] = metric_sums[metric_name] / image_count if image_count > 0 else 0.0
        local_results[set_name] = set_result

        metric_line = ", ".join(
            f"{display_name}: {set_result[metric_name]:.4f}" for metric_name, display_name, _, _ in active_metric_specs
        )
        print(
            f"[GPU {rank}][{log_prefix}] {set_name:20s} - {metric_line} "
            f"({set_result['count']}/{set_result['total_count']} imgs)"
        )

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, local_results)

    if rank != 0:
        if hasattr(eval_model, "set_runtime_resize_fallback"):
            eval_model.set_runtime_resize_fallback(False)
        return 0.0

    merged = {
        set_name: {"count": 0, "total_count": 0, **{f"{m}_sum": 0.0 for m, _, _, _ in active_metric_specs}}
        for set_name in all_set_names
    }
    for rd in gathered_results:
        if not rd:
            continue
        for set_name, result in rd.items():
            if set_name not in merged:
                merged[set_name] = {
                    "count": 0,
                    "total_count": 0,
                    **{f"{m}_sum": 0.0 for m, _, _, _ in active_metric_specs},
                }
            merged[set_name]["count"] += int(result.get("count", 0))
            merged[set_name]["total_count"] = max(
                int(merged[set_name].get("total_count", 0)),
                int(result.get("total_count", 0)),
            )
            for metric_name, _, _, _ in active_metric_specs:
                merged[set_name][f"{metric_name}_sum"] += float(result.get(f"{metric_name}_sum", 0.0))

    metric_title = " / ".join(display_name for _, display_name, _, _ in active_metric_specs)
    metric_step = int(tb_step) if tb_step is not None else int(epoch)
    display_step = str(step_label) if step_label is not None else f"Epoch {epoch}"
    print(f"\n{'=' * 70}")
    print(f"[{log_prefix}] {display_step} Metrics ({metric_title})")
    print(f"{'=' * 70}")

    metric_sums = {metric_name: 0.0 for metric_name, _, _, _ in active_metric_specs}
    valid_count = 0
    for set_name in all_set_names:
        if set_name not in merged:
            continue
        merged_result = merged[set_name]
        count = int(merged_result.get("count", 0))
        total_count = int(merged_result.get("total_count", 0))
        if count <= 0:
            continue
        result = {"count": count}
        for metric_name, _, _, _ in active_metric_specs:
            result[metric_name] = merged_result[f"{metric_name}_sum"] / float(count)
        metric_line = ", ".join(
            f"{display_name}: {result[metric_name]:.4f}" for metric_name, display_name, _, _ in active_metric_specs
        )
        print(f"{set_name:24s} {metric_line} ({count}/{total_count} imgs)")
        if writer is not None:
            for metric_name, _, tb_name, _ in active_metric_specs:
                writer.add_scalar(f"{tb_prefix}_{set_name}/{tb_name}", result[metric_name], metric_step)
        for metric_name, _, _, _ in active_metric_specs:
            metric_sums[metric_name] += result[metric_name]
        valid_count += 1

    avg_metrics = {
        metric_name: (metric_sums[metric_name] / valid_count if valid_count > 0 else 0.0)
        for metric_name, _, _, _ in active_metric_specs
    }
    print(f"{'-' * 70}")
    avg_line = ", ".join(
        f"{display_name}: {avg_metrics[metric_name]:.4f}" for metric_name, display_name, _, _ in active_metric_specs
    )
    print(f"{(log_prefix + ' Avg'):24s} {avg_line}")
    print(f"{'=' * 70}\n")
    if writer is not None:
        for metric_name, _, tb_name, _ in active_metric_specs:
            writer.add_scalar(f"{tb_prefix}/Avg_{tb_name}", avg_metrics[metric_name], metric_step)
    if hasattr(eval_model, "set_runtime_resize_fallback"):
        eval_model.set_runtime_resize_fallback(False)
    return avg_metrics.get("musiq", next(iter(avg_metrics.values()), 0.0))


def train_stage(cfg):
    current_stage = cfg.incremental.stage
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    setup_runtime(cfg, rank)
    checkpoint_dir = os.path.join(cfg.train.save_dir, cfg.train.config_parent_dir, cfg.exp_name)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    use_bf16 = getattr(cfg.train, "use_bf16", False)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.train.log_dir, cfg.exp_name))
        print(f"\n[Stage {current_stage}] Start - Device: {device}, World Size: {world_size}")
        print(f"AMP dtype: {'BF16' if use_bf16 else 'FP16'}")
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
    use_validation = bool(getattr(cfg.train, "use_validation", False))
    use_real_eval = bool(getattr(cfg.train, "use_real_eval", False))
    real_eval_freq = int(getattr(cfg.train, "real_eval_freq", cfg.train.val_freq))
    real_eval_max_side = int(getattr(cfg.train, "real_eval_max_side", 0) or 0)
    real_eval_allow_runtime_resize_retry = bool(getattr(cfg.train, "real_eval_allow_runtime_resize_retry", True))

    # Rank 0 먼저 빌드 (CUDA extension 컴파일 동기화)
    if rank == 0:
        student, teacher, dino, cgcd = build_models(cfg, device, rank)
    dist.barrier()
    if rank != 0:
        student, teacher, dino, cgcd = build_models(cfg, device, rank)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(cgcd.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda") if not use_bf16 else None
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
        iqa_metrics["fgresq"] = create_fgresq_metric(device=device)
        if rank == 0:
            print("✓ FGResQ metric initialized")
    if rank == 0 and musiq_warmup_epochs > 0:
        print(
            f"✓ MUSIQ warmup enabled: first {musiq_warmup_epochs} epochs use MUSIQ, then switch to {cfg.train.pseudo_update_mode}"
        )

    start_epoch, global_step, best_psnr, _ = load_checkpoints(
        student, teacher, cgcd, optimizer, scheduler, cfg, device, rank
    )
    is_resume_run = start_epoch > 1
    run_init_validate = bool(getattr(cfg.train, "init_validate", False)) and not is_resume_run
    cfg_init_pseudo = bool(getattr(cfg.train, "init_pseudo_label", False))
    cfg_copy_pseudo = bool(getattr(cfg.train, "copy_pseudo_label", False))
    run_init_pseudo = cfg_init_pseudo and not is_resume_run
    run_init_copy = cfg_copy_pseudo and (not cfg_init_pseudo) and (not is_resume_run)
    if rank == 0 and cfg_init_pseudo and cfg_copy_pseudo and not is_resume_run:
        print("[InitPseudo] copy_pseudo_label is ignored because init_pseudo_label is enabled.")
    if rank == 0 and is_resume_run:
        if getattr(cfg.train, "init_validate", False):
            print("[Resume] Skip initial validation at epoch 0 to keep TensorBoard scalar steps monotonic.")
        if cfg_init_pseudo:
            print("[Resume] Skip pseudo-label initialization on resume.")
        if cfg_copy_pseudo:
            print("[Resume] Skip pseudo-label copy initialization on resume.")

    # 체크포인트 로드 후 student gradient 재활성화
    for param in student.parameters():
        param.requires_grad = True

    # Epoch 동기화
    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())

    # Broadcast-heavy sync can stall when one rank is delayed by data I/O.
    # These models do not require per-step buffer broadcast for correctness here.
    student = DDP(student.to(device), device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    cgcd = DDP(cgcd.to(device), device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    # 초기 validation
    if use_validation and len(val_datasets) > 0 and run_init_validate:
        psnr_old, psnr_new, psnr_all = validate_stage(
            teacher, dino, cgcd, val_datasets, device, 0, cfg, writer, rank, world_size, amp_dtype
        )
        if rank == 0:
            print(f"  Teacher validation: OLD={psnr_old:.2f}, NEW={psnr_new:.2f}, ALL={psnr_all:.2f}")
        dist.barrier()
    if use_real_eval and len(real_eval_datasets) > 0 and run_init_validate:
        evaluate_real_unpaired(
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
        )
        dist.barrier()

    # Pseudo label 초기 복사 (teacher init보다 우선되지 않음)
    if run_init_copy:
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

        init_loader = DataLoader(
            init_subset,
            **init_loader_kwargs,
        )

        if rank == 0:
            print(
                f"[CopyPseudo] Distributed copy enabled: world_size={world_size}, "
                f"per-rank samples≈{len(init_subset)}, batch_size={max(1, init_batch_size)}"
            )

        copy_pseudo_labels(
            init_loader,
            rank=rank,
            show_progress=(rank == 0),
        )
        dist.barrier()

    # Pseudo label 초기화 (Rank 0만)
    if run_init_pseudo:
        init_batch_size = int(getattr(cfg.train, "init_pseudo_batch_size", cfg.train.batch_size))
        init_num_workers = int(getattr(cfg.train, "init_pseudo_num_workers", max(2, cfg.train.num_workers // 2)))
        init_pin_memory = bool(getattr(cfg.train, "init_pseudo_pin_memory", getattr(cfg.train, "pin_memory", True)))
        init_prefetch_factor = int(
            getattr(cfg.train, "init_pseudo_prefetch_factor", getattr(cfg.train, "prefetch_factor", 4))
        )
        init_persistent_workers = bool(
            getattr(cfg.train, "init_pseudo_persistent_workers", getattr(cfg.train, "persistent_workers", True))
        )

        # Split unlabeled dataset across ranks without overlap.
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

        init_loader = DataLoader(
            init_subset,
            **init_loader_kwargs,
        )

        if rank == 0:
            print(
                f"[InitPseudo] Distributed init enabled: world_size={world_size}, "
                f"per-rank samples≈{len(init_subset)}, batch_size={max(1, init_batch_size)}"
            )

        initialize_pseudo_labels(
            teacher,
            dino,
            cgcd.module,
            init_loader,
            device,
            dino_transform=transform_resize,
            amp_dtype=amp_dtype,
            use_class_routing=False,
            rank=rank,
            show_progress=(rank == 0),
        )
        dist.barrier()

    # 학습 루프
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        labeled_sampler.set_epoch(epoch)
        unlabeled_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(
                f"[Stage {current_stage}] Epoch {epoch}/{cfg.train.epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            print(f"{'='*60}")

        # Validation 먼저
        if use_validation and epoch % cfg.train.val_freq == 0 and len(val_datasets) > 0:
            psnr_old, psnr_new, psnr_all = validate_stage(
                teacher, dino, cgcd.module, val_datasets, device, epoch, cfg, writer, rank, world_size, amp_dtype
            )
            is_best = rank == 0 and psnr_all > best_psnr
            if is_best:
                best_psnr = psnr_all
        else:
            is_best = False

        if use_real_eval and real_eval_freq > 0 and epoch % real_eval_freq == 0 and len(real_eval_datasets) > 0:
            evaluate_real_unpaired(
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
            )
            dist.barrier()

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
            cgcd_score_mode=str(get_cfg_cgcd_value(cfg, "cgcd_score_mode", "clear")),
            musiq_warmup_epochs=musiq_warmup_epochs,
        )

        scheduler.step()

        if rank == 0 and writer is not None:
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Train/train_loss", train_loss, epoch)
            writer.add_scalar("Train/pseudo_update_cnt_epoch", pseudo_update_cnt, epoch)

        log_pseudo_labels_wres(
            cfg.train.pseudo_patches_dir,
            writer,
            epoch,
            rank,
            global_step=global_step,
            reference_patches_dir=getattr(cfg.train, "lq_patches_dir", None),
        )
        if rank == 0 and writer is not None:
            writer.flush()

        if rank == 0:
            state = {
                "epoch": epoch,
                "stage": current_stage,
                "student_state_dict": student.module.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "cgcd_state_dict": cgcd.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "best_psnr": best_psnr,
                "is_best": is_best,
            }
            if epoch % cfg.train.save_freq == 0 or epoch == cfg.train.epochs:
                save_path = os.path.join(checkpoint_dir, f"stage{current_stage}_epoch_{epoch}.pth")
                torch.save(state, save_path)
                print(f"Saved checkpoint: {save_path}")
            latest_path = os.path.join(checkpoint_dir, f"stage{current_stage}_latest.pth")
            torch.save(state, latest_path)

        dist.barrier()

    if rank == 0:
        print(f"\n[Stage {current_stage}] Training completed! Best PSNR: {best_psnr:.2f} dB")
        if writer is not None:
            writer.close()
    dist.destroy_process_group()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Stage N: Incremental OneRestore (wo LoRA)")
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
    parser.add_argument("--init_validate", action="store_false")
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
        choices=["musiq", "fgresq", "cgcd", "alternate"],
        help="Pseudo label 업데이트 전략: musiq(MUSIQ만), fgresq(FGResQ만), cgcd(CGCD만), alternate(번갈아)",
    )
    parser.add_argument(
        "--cgcd_score_mode",
        type=str,
        default=None,
        choices=[
            "clear",
            "neg_distance",
            "contrastive",
            "anchor",
            "mahalanobis_margin",
            "mahalanobis_pca",
        ],
        help="CGCD 점수 방식 (미지정 시 config의 cgcd_score_mode 사용)",
    )
    parser.add_argument(
        "--musiq_warmup_epochs",
        type=int,
        default=0,
        help="초반 N epoch 동안 MUSIQ로 pseudo 업데이트 후 pseudo_update_mode로 전환 (-1이면 전체 epoch의 절반)",
    )
    parser.add_argument(
        "--cgcd_contrastive_pos_weight",
        type=float,
        default=None,
        help="contrastive 모드에서 clear(logit) 가중치",
    )
    parser.add_argument(
        "--cgcd_contrastive_neg_weight",
        type=float,
        default=None,
        help="contrastive 모드에서 negative 집계(logit) 가중치",
    )
    parser.add_argument(
        "--cgcd_contrastive_tau",
        type=float,
        default=None,
        help="contrastive 모드에서 negative logsumexp 온도(tau)",
    )
    parser.add_argument(
        "--cgcd_contrastive_score_temp",
        type=float,
        default=None,
        help="contrastive 모드 최종 sigmoid temperature",
    )
    parser.add_argument(
        "--pseudo_update_margin",
        type=float,
        default=None,
        help="pseudo 업데이트 조건 margin: teacher > max(student, reference) + margin",
    )
    args = parser.parse_args()
    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    # exp_name 자동 생성 (미지정 시 config 파일명 + stage)
    if args.exp_name is None:
        args.exp_name = f"{config_basename}_stage{args.stage}"
        print(f"[INFO] Auto-generated exp_name: {args.exp_name}")

    # pseudo_patches_dir가 비어 있으면 실험명 기반으로 자동 생성
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
    cfg.train.resume = args.resume
    cfg.train.init_validate = args.init_validate
    cfg.train.scratch = args.scratch
    cfg.train.pseudo_update_mode = args.pseudo_update_mode
    cfg.train.musiq_warmup_epochs = args.musiq_warmup_epochs
    if args.cgcd_score_mode is not None:
        cfg.train.cgcd_score_mode = args.cgcd_score_mode
    if args.cgcd_contrastive_pos_weight is not None:
        cfg.train.cgcd_contrastive_pos_weight = args.cgcd_contrastive_pos_weight
    if args.cgcd_contrastive_neg_weight is not None:
        cfg.train.cgcd_contrastive_neg_weight = args.cgcd_contrastive_neg_weight
    if args.cgcd_contrastive_tau is not None:
        cfg.train.cgcd_contrastive_tau = args.cgcd_contrastive_tau
    if args.cgcd_contrastive_score_temp is not None:
        cfg.train.cgcd_contrastive_score_temp = args.cgcd_contrastive_score_temp
    if args.pseudo_update_margin is not None:
        cfg.train.pseudo_update_margin = args.pseudo_update_margin
    if not getattr(cfg.train, "pseudo_patches_dir", ""):
        cfg.train.pseudo_patches_dir = auto_pseudo_dir
        print(f"[INFO] Auto-generated pseudo_patches_dir: {cfg.train.pseudo_patches_dir}")

    checkpoint_dir_template, prev_stage, base_checkpoint_path, config_parent_dir = precompute_checkpoint_path(
        cfg, args, config_basename
    )
    cfg.train.checkpoint_dir_template = checkpoint_dir_template
    cfg.train.prev_stage = prev_stage
    cfg.train.base_checkpoint_path = base_checkpoint_path
    cfg.train.config_parent_dir = config_parent_dir

    train_stage(cfg)


if __name__ == "__main__":
    main()
