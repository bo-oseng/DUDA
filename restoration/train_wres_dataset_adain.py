import os
import sys
import argparse
import yaml
import json
import random
import math
import time
import numpy as np
from tqdm import tqdm
from glob import glob
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

import pyiqa

import cgcd.models as cgcd_base_models
import cgcd.models_cov as cgcd_models

CGCD_ARCH_REGISTRY = {
    "soft": "CGCDSignalModuleSoft",
    "soft_no_le": "CGCDSignalModuleStaticSoftFixedEmbedding",
    "adain": "CGCDSignalModuleControlNet",
    "adain_wo_zero_init": "CGCDSignalModuleControlNetNoZeroInit",
    "mmdit": "CGCDSignalModuleMMdit",
    "adain_mmdit": "CGCDSignalModuleMMdit",
    "hard": "CGCDSignalModuleHard",
    "hard_le": "CGCDSignalModuleStaticHardLeanableEmbedding",
    "hard_le_finst": "CGCDSignalModuleStaticHardLeanableEmbeddingInst",
    "hard_le_soft_finst": "CGCDSignalModuleStaticHardLeanableEmbeddingSoftInst",
    "hard_top1_no_le": "CGCDSignalModuleStaticHardFixedEmbedding",
}

CGCD_ARCH_MODULE_REGISTRY = {
    "soft": cgcd_base_models,
    "soft_no_le": cgcd_base_models,
    "hard": cgcd_base_models,
    "hard_le": cgcd_base_models,
    "hard_le_finst": cgcd_base_models,
    "hard_le_soft_finst": cgcd_base_models,
    "hard_top1_no_le": cgcd_base_models,
}

from models.OneRestore import OneRestore
from datasets_wres import create_train_datasets, create_val_datasets, init_from_config
from datasets_wres_continual import create_real_eval_datasets
from utils_lib.utils import seed_everything, dict2namespace, count_params, save_checkpoint, load_checkpoint
from utils_lib.utils_incremental import print_incremental_info
from utils_lib.utils_dataset_wres import get_train_val_data_for_stage
from metrics import pt_psnr, pt_ssim
from feature_extractor.sl_finetuned_model import load_finetuned_model_from_checkpoint

train_transform = v2.Compose(
    [
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


val_transform = v2.Compose(
    [
        v2.CenterCrop(224),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

real_eval_transform = v2.Compose(
    [
        v2.Resize([224, 224]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))

    # Get rank and world size from environment variables set by torchrun
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set device for this process
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


class DistributedWeightedSampler(Sampler):
    """DDP-compatible weighted sampler for class-balanced sampling."""

    def __init__(self, weights, num_replicas, rank, replacement=True, num_samples=None, seed=42):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas-1}], got {rank}")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.numel() == 0:
            raise ValueError("weights must not be empty")

        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        if num_samples is None:
            num_samples = int(math.ceil(len(self.weights) / float(self.num_replicas)))
        self.num_samples = max(1, int(num_samples))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        sampled = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = sampled[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _extract_labels_from_dataset(dataset):
    """Extract per-sample class labels from ConcatDataset/Subset of TrainLabeled datasets."""
    if isinstance(dataset, Subset):
        base_labels = _extract_labels_from_dataset(dataset.dataset)
        return base_labels[np.asarray(dataset.indices, dtype=np.int64)]

    if isinstance(dataset, ConcatDataset):
        labels = []
        for sub_dataset in dataset.datasets:
            if not hasattr(sub_dataset, "deg_class"):
                raise AttributeError("Expected sub-dataset to have 'deg_class' for class-balanced sampling.")
            labels.extend([int(sub_dataset.deg_class)] * len(sub_dataset))
        return np.asarray(labels, dtype=np.int64)

    if hasattr(dataset, "deg_class"):
        return np.full(len(dataset), int(dataset.deg_class), dtype=np.int64)

    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset)}")


def _apply_per_class_cap(dataset, labels, max_samples_per_class, seed):
    if max_samples_per_class is None or max_samples_per_class <= 0:
        return dataset, labels, None

    rng = np.random.default_rng(seed)
    selected_indices = []
    before = {}
    after = {}

    for cls_id in sorted(np.unique(labels).tolist()):
        cls_indices = np.where(labels == cls_id)[0]
        before[int(cls_id)] = int(len(cls_indices))

        if len(cls_indices) > max_samples_per_class:
            cls_indices = rng.choice(cls_indices, size=max_samples_per_class, replace=False)
            cls_indices = np.sort(cls_indices)

        after[int(cls_id)] = int(len(cls_indices))
        selected_indices.extend(cls_indices.tolist())

    selected_indices = np.asarray(sorted(selected_indices), dtype=np.int64)
    capped_dataset = Subset(dataset, selected_indices.tolist())
    capped_labels = labels[selected_indices]

    stats = {
        "original_total": int(len(labels)),
        "capped_total": int(len(capped_labels)),
        "before": before,
        "after": after,
        "max_samples_per_class": int(max_samples_per_class),
    }
    return capped_dataset, capped_labels, stats


def _build_class_balanced_weights(labels):
    class_counts = {}
    for cls_id in labels.tolist():
        class_counts[int(cls_id)] = class_counts.get(int(cls_id), 0) + 1

    weights = np.asarray([1.0 / class_counts[int(cls_id)] for cls_id in labels.tolist()], dtype=np.float64)
    return weights, class_counts


def train_one_epoch(
    model,
    dino_model,
    cgcd_model,
    train_loader,
    optimizer,
    scaler,
    device,
    epoch,
    cfg,
    writer,
    global_step,
    rank,
    amp_dtype=torch.float16,
):
    model.train()
    cgcd_model.train()  # Trainable
    dino_model.eval()  # Always frozen

    criterion_l1 = nn.L1Loss()
    criterion_ce = nn.CrossEntropyLoss()

    # [STREAM OPTIMIZATION] Create dedicated CUDA streams
    transfer_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}")
    else:
        pbar = train_loader

    # Use tensors for accumulation to avoid GPU sync every iteration
    running_loss = 0.0
    running_l1 = 0.0
    running_ce = 0.0
    running_psnr = 0.0

    # Debug accumulators for first epoch (all ranks accumulate, then all_reduce)
    if epoch == 1:
        debug_num_classes = 12
        debug_per_class_correct = torch.zeros(debug_num_classes, device=device)
        debug_per_class_total = torch.zeros(debug_num_classes, device=device)
        debug_ce_sum = torch.zeros(1, device=device)
        debug_ce_count = 0

    for i, batch in enumerate(pbar):
        with torch.cuda.stream(transfer_stream):
            hq_image = batch[0].to(device, non_blocking=True)
            lq_image = batch[1].to(device, non_blocking=True)
            deg_class = batch[2].to(device, non_blocking=True)  # GT label for CE loss (OLD classes only)
        compute_stream.wait_stream(transfer_stream)

        # Extract features using DINO (frozen)
        with torch.no_grad():
            dino_feature = dino_model(train_transform(lq_image)).pooler_output

        # [AMP] Mixed precision forward pass
        with autocast("cuda", dtype=amp_dtype):
            # CGCD forward: outputs embedding + logits (trainable)
            cgcd_embd, logits = cgcd_model(dino_feature)

            restored = model(lq_image, cgcd_embd)

            # tmp0 = hq_image[-1]
            # tmp1 = lq_image[-1]
            # tmp2 = restored[-1].detach().clamp(0.0, 1.0).cpu()

            loss_l1 = criterion_l1(restored, hq_image)

            # CE loss for monitoring only (logits are frozen, no gradient propagation)
            with torch.no_grad():
                loss_ce = criterion_ce(logits, deg_class)

            loss = cfg.train.l1_weight * loss_l1  # Only L1 loss for training

        # Debug: Accumulate classification stats for entire first epoch (all ranks)
        if epoch == 1:
            with torch.no_grad():
                pred_classes = logits.argmax(dim=1)
                debug_ce_sum += loss_ce.detach()
                debug_ce_count += 1

                for cls_id in deg_class.unique().tolist():
                    mask = deg_class == cls_id
                    debug_per_class_correct[cls_id] += (pred_classes[mask] == cls_id).sum()
                    debug_per_class_total[cls_id] += mask.sum()

                # Print static info only on rank 0 first batch
                if rank == 0 and i == 0:
                    from datasets_wres import ORIG2CLASSIFIER, DEG_MAP

                    print(f"\n[DEBUG] ORIG2CLASSIFIER: {ORIG2CLASSIFIER}")
                    print(f"[DEBUG] DEG_MAP: {DEG_MAP}")
                    print(f"[DEBUG] Logits shape: {logits.shape}")

        optimizer.zero_grad(set_to_none=True)

        # Use GradScaler only for FP16, BF16 doesn't need it
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(cgcd_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(cgcd_model.parameters(), max_norm=1.0)
            optimizer.step()

        # Compute PSNR
        with torch.no_grad():
            psnr = pt_psnr(hq_image, restored).mean()

        # Accumulate metrics
        running_loss += loss.detach().item()
        running_l1 += loss_l1.detach().item()
        running_ce += loss_ce.detach().item()
        running_psnr += psnr.detach().item()

        # Update progress bar (less frequent to reduce overhead) - only on rank 0
        if rank == 0 and (global_step + 1) % 10 == 0:
            avg_loss = running_loss / (i + 1)
            avg_psnr = running_psnr / (i + 1)
            pbar.set_description(
                f"[Stage 0] Epoch {epoch}/{cfg.train.epochs} - loss: {avg_loss:.4f}, "
                f"l1: {running_l1/(i+1):.4f}, ce: {running_ce/(i+1):.4f}, psnr: {avg_psnr:.2f}"
            )

        # TensorBoard logging - only on rank 0
        if rank == 0 and (i + 1) % cfg.train.print_freq == 0:
            writer.add_scalar("Train/Loss", loss.detach().item(), global_step)
            writer.add_scalar("Train/L1_Loss", loss_l1.detach().item(), global_step)
            writer.add_scalar("Train/CE_Loss", loss_ce.detach().item(), global_step)
            writer.add_scalar("Train/PSNR", psnr.detach().item(), global_step)

        global_step += 1

    # Epoch stats
    num_batches = len(pbar)
    avg_loss = running_loss / num_batches
    avg_l1 = running_l1 / num_batches
    avg_ce = running_ce / num_batches
    avg_psnr = running_psnr / num_batches

    if rank == 0:
        print(
            f"\n[Stage 0] Epoch {epoch} - Loss: {avg_loss:.4f}, L1: {avg_l1:.4f}, CE: {avg_ce:.4f}, PSNR: {avg_psnr:.2f} dB"
        )

    # Debug: All-reduce classification stats across GPUs, then print on rank 0
    if epoch == 1:
        dist.all_reduce(debug_per_class_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(debug_per_class_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(debug_ce_sum, op=dist.ReduceOp.SUM)

        if rank == 0:
            from datasets_wres import ORIG2CLASSIFIER, DEG_MAP

            # Reverse mapping: classifier_id → orig_id
            classifier2orig = {v: k for k, v in ORIG2CLASSIFIER.items()}
            # Reverse DEG_MAP: orig_id → name
            orig2name = {v: k for k, v in DEG_MAP.items()}

            total_correct = debug_per_class_correct.sum().item()
            total_samples = debug_per_class_total.sum().item()
            overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
            # CE sum is across all ranks; each rank had debug_ce_count batches
            total_ce_count = debug_ce_count * dist.get_world_size()
            avg_ce_epoch = debug_ce_sum.item() / total_ce_count if total_ce_count > 0 else 0.0

            print(f"\n{'=' * 70}")
            print(
                f"[DEBUG] Epoch 1 Full Classification Summary ({int(total_samples)} samples, {total_ce_count} batches, {dist.get_world_size()} GPUs)"
            )
            print(f"{'=' * 70}")
            print(f"  Overall Accuracy: {overall_acc:.4f} ({int(total_correct)}/{int(total_samples)})")
            print(f"  Average CE Loss:  {avg_ce_epoch:.4f}")
            print(f"\n  Per-class breakdown (cls_id = classifier space):")
            for cls_id in range(debug_num_classes):
                cls_total = int(debug_per_class_total[cls_id].item())
                if cls_total == 0:
                    continue
                cls_correct = int(debug_per_class_correct[cls_id].item())
                cls_acc = cls_correct / cls_total
                orig_id = classifier2orig.get(cls_id, cls_id)
                cls_name = orig2name.get(orig_id, f"class_{orig_id}")
                print(
                    f"    [cls={cls_id}] {cls_name:20s} (orig_id={orig_id}) - Acc: {cls_acc:.4f} ({cls_correct}/{cls_total})"
                )
            print(f"{'=' * 70}\n")

    return global_step, avg_loss


@torch.no_grad()
def validate_multi_gpu(
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
    val_metrics=None,
):
    model.eval()
    dino_model.eval()
    cgcd_model.eval()

    # Use pre-initialized metrics if provided (much faster than creating every time)
    if val_metrics is None:
        musiq_metric = pyiqa.create_metric("musiq", as_loss=False, device=device)
        lpips_metric = pyiqa.create_metric("lpips", as_loss=False, device=device)
    else:
        musiq_metric = val_metrics["musiq"]
        lpips_metric = val_metrics["lpips"]

    # Get NEW_CLASSES from dataset module (initialized from config)
    from datasets_wres import NEW_CLASSES

    if NEW_CLASSES is None:
        if rank == 0:
            print("[Warning] NEW_CLASSES not initialized, treating all classes as OLD")
        NEW_CLASSES = []

    all_deg_names = [deg_name for _, deg_name in val_datasets]

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"[Stage 0] Validation at Epoch {epoch}")
        print(f"Multi-GPU Validation: {world_size} GPUs (each set is sharded across ranks)")
        print(f"{'=' * 60}")

    # NOTE:
    # Previous implementation assigned whole degradation types to a single rank and then all_gather'ed once.
    # With highly imbalanced set sizes, fast ranks waited too long and hit NCCL timeout.
    # We now shard each validation set across all ranks (strided split), then all_reduce set-wise stats.
    results = {} if rank == 0 else None

    for val_dataset, deg_name in val_datasets:
        local_indices = list(range(rank, len(val_dataset), world_size))
        local_subset = Subset(val_dataset, local_indices)

        val_loader = DataLoader(
            local_subset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            pin_memory=True,
        )

        class_type = "NEW" if deg_name in NEW_CLASSES else "OLD"
        is_new_class = deg_name in NEW_CLASSES

        if rank == 0:
            iterator = tqdm(
                val_loader,
                desc=f"[Stage 0][VAL][{class_type}] {deg_name:<15}",
                leave=False,
                ncols=100,
            )
        else:
            iterator = val_loader

        local_psnr_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_ssim_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_musiq_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_lpips_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_count = torch.zeros(1, device=device, dtype=torch.float64)

        for batch in iterator:
            hq_image = batch[0].to(device, non_blocking=True)
            lq_image = batch[1].to(device, non_blocking=True)

            dino_feature = dino_model(val_transform(lq_image)).pooler_output

            with autocast("cuda", dtype=amp_dtype):
                cgcd_embd, logits = cgcd_model(dino_feature)
                restored = model(lq_image, cgcd_embd)

            restored_clamped = torch.clamp(restored, 0, 1)

            local_psnr_sum += pt_psnr(hq_image, restored_clamped).mean().double()
            local_ssim_sum += pt_ssim(hq_image, restored_clamped).mean().double()
            if is_new_class:
                local_musiq_sum += musiq_metric(restored_clamped).mean().double()
                local_lpips_sum += lpips_metric(restored_clamped, hq_image).mean().double()
            local_count += float(hq_image.shape[0])

        local_stats = torch.stack(
            [local_psnr_sum[0], local_ssim_sum[0], local_musiq_sum[0], local_lpips_sum[0], local_count[0]]
        )
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        global_count = int(local_stats[4].item())
        all_psnr = (local_stats[0] / local_stats[4]).item() if global_count > 0 else 0.0
        all_ssim = (local_stats[1] / local_stats[4]).item() if global_count > 0 else 0.0
        all_musiq = (local_stats[2] / local_stats[4]).item() if (is_new_class and global_count > 0) else 0.0
        all_lpips = (local_stats[3] / local_stats[4]).item() if (is_new_class and global_count > 0) else 0.0

        if rank == 0:
            results[deg_name] = {
                "psnr": all_psnr,
                "ssim": all_ssim,
                "musiq": all_musiq,
                "lpips": all_lpips,
                "count": global_count,
                "is_new": is_new_class,
            }

    if rank == 0:
        # OLD/NEW/ALL로 분리해서 평균 계산
        old_psnr_sum = 0.0
        old_ssim_sum = 0.0
        old_count = 0

        new_psnr_sum = 0.0
        new_ssim_sum = 0.0
        new_musiq_sum = 0.0
        new_lpips_sum = 0.0
        new_count = 0

        all_psnr_sum = 0.0
        all_ssim_sum = 0.0
        all_count = 0

        print(f"\n{'=' * 80}")
        print(f"[Stage 0] Validation Results at Epoch {epoch}")
        print(f"{'=' * 80}")

        for deg_name in all_deg_names:
            if deg_name in results:
                psnr = results[deg_name]["psnr"]
                ssim = results[deg_name]["ssim"]
                musiq = results[deg_name]["musiq"]
                lpips = results[deg_name]["lpips"]
                count = results[deg_name]["count"]
                is_new = results[deg_name]["is_new"]

                class_type = "NEW" if is_new else "OLD"

                if is_new:
                    print(
                        f"[{class_type}] {deg_name:20s} - "
                        f"PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}, "
                        f"MUSIQ: {musiq:.4f}, LPIPS: {lpips:.4f}"
                    )
                else:
                    print(f"[{class_type}] {deg_name:20s} - " f"PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")

                # TensorBoard 로깅 (개별 클래스)
                if writer:
                    writer.add_scalar(f"Val_{deg_name}/PSNR", psnr, epoch)
                    writer.add_scalar(f"Val_{deg_name}/SSIM", ssim, epoch)
                    if is_new:
                        writer.add_scalar(f"Val_{deg_name}/MUSIQ", musiq, epoch)
                        writer.add_scalar(f"Val_{deg_name}/LPIPS", lpips, epoch)

                # ALL 집계
                all_psnr_sum += psnr
                all_ssim_sum += ssim
                all_count += 1

                # OLD/NEW 분리 집계
                if is_new:
                    new_psnr_sum += psnr
                    new_ssim_sum += ssim
                    new_musiq_sum += musiq
                    new_lpips_sum += lpips
                    new_count += 1
                else:
                    old_psnr_sum += psnr
                    old_ssim_sum += ssim
                    old_count += 1

        # 평균 계산
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
                f"{'NEW Classes Avg':20s} - PSNR: {avg_psnr_new:.4f} dB, SSIM: {avg_ssim_new:.4f}, "
                f"MUSIQ: {avg_musiq_new:.4f}, LPIPS: {avg_lpips_new:.4f}"
            )
        print(f"{'ALL Classes Avg':20s} - PSNR: {avg_psnr_all:.4f} dB, SSIM: {avg_ssim_all:.4f}")
        print(f"{'=' * 80}\n")

        # TensorBoard 로깅 (집계)
        if writer:
            writer.add_scalar("Val_OLD/Avg_PSNR", avg_psnr_old, epoch)
            writer.add_scalar("Val_OLD/Avg_SSIM", avg_ssim_old, epoch)
            if new_count > 0:
                writer.add_scalar("Val_NEW/Avg_PSNR", avg_psnr_new, epoch)
                writer.add_scalar("Val_NEW/Avg_SSIM", avg_ssim_new, epoch)
                writer.add_scalar("Val_NEW/Avg_MUSIQ", avg_musiq_new, epoch)
                writer.add_scalar("Val_NEW/Avg_LPIPS", avg_lpips_new, epoch)
            writer.add_scalar("Val_ALL/Avg_PSNR", avg_psnr_all, epoch)
            writer.add_scalar("Val_ALL/Avg_SSIM", avg_ssim_all, epoch)

        return avg_psnr_old, avg_psnr_new, avg_psnr_all
    else:
        return 0.0, 0.0, 0.0


def _parse_real_eval_sets(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = [s.strip() for s in raw_value.split(",")]
    elif isinstance(raw_value, (list, tuple)):
        values = [str(v).strip() for v in raw_value]
    else:
        values = [str(raw_value).strip()]
    return [v for v in values if v]


@torch.no_grad()
def validate_real_unpaired_multi_gpu(
    model,
    dino_model,
    cgcd_model,
    real_eval_datasets,
    device,
    epoch,
    cfg,
    writer,
    rank,
    world_size,
    amp_dtype=torch.float16,
    timeout_seconds=3600,
    max_eval_side=1280,
    num_workers=2,
    real_val_metrics=None,
):
    if len(real_eval_datasets) == 0:
        return 0.0, False

    model.eval()
    dino_model.eval()
    cgcd_model.eval()

    if real_val_metrics is None:
        musiq_metric = pyiqa.create_metric("musiq", as_loss=False, device=device)
    else:
        musiq_metric = real_val_metrics["musiq"]

    timeout_seconds = int(timeout_seconds or 0)
    start_ts = time.monotonic()
    timed_out = False
    all_set_names = [set_name for _, set_name in real_eval_datasets]
    results = {} if rank == 0 else None

    if rank == 0:
        timeout_msg = f"{timeout_seconds}s" if timeout_seconds > 0 else "disabled"
        print(f"\n{'=' * 80}")
        print(f"[Stage 0][RealEval] Epoch {epoch} - datasets={len(real_eval_datasets)}, timeout={timeout_msg}")
        print(f"{'=' * 80}")

    for eval_dataset, set_name in real_eval_datasets:
        if timeout_seconds > 0:
            timeout_tensor = torch.tensor(
                [1 if (time.monotonic() - start_ts) >= timeout_seconds else 0], device=device, dtype=torch.int32
            )
            dist.all_reduce(timeout_tensor, op=dist.ReduceOp.MAX)
            if int(timeout_tensor.item()) > 0:
                timed_out = True
                if rank == 0:
                    print(f"[Stage 0][RealEval] Timeout reached before set '{set_name}', skip remaining sets.")
                break

        local_indices = list(range(rank, len(eval_dataset), world_size))
        eval_subset = Subset(eval_dataset, local_indices)
        print(
            f"[Stage 0][RealEval][Rank {rank}] {set_name}: assigned {len(local_indices)}/{len(eval_dataset)} images",
            flush=True,
        )
        eval_loader = DataLoader(
            eval_subset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )

        if rank == 0:
            iterator = tqdm(
                eval_loader,
                desc=f"[Stage 0][RealEval] {set_name:<18} ({len(local_indices)}/{len(eval_dataset)})",
                leave=False,
                ncols=110,
            )
        else:
            iterator = eval_loader

        local_musiq_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_count = torch.zeros(1, device=device, dtype=torch.float64)

        for batch in iterator:
            lq_image = batch[0].to(device, non_blocking=True)

            if max_eval_side and max_eval_side > 0:
                _, _, h, w = lq_image.shape
                longest = max(h, w)
                if longest > max_eval_side:
                    scale = float(max_eval_side) / float(longest)
                    new_h = max(16, int((h * scale) // 16 * 16))
                    new_w = max(16, int((w * scale) // 16 * 16))
                    lq_image = F.interpolate(lq_image, size=(new_h, new_w), mode="bilinear", align_corners=False)

            with autocast("cuda", dtype=amp_dtype):
                dino_feature = dino_model(real_eval_transform(lq_image)).pooler_output
                cgcd_embd, _ = cgcd_model(dino_feature)
                restored = model(lq_image, cgcd_embd)

            restored = torch.clamp(restored, 0, 1)
            local_musiq_sum += musiq_metric(restored).mean().double()
            local_count += float(restored.shape[0])

        print(
            f"[Stage 0][RealEval][Rank {rank}] {set_name}: completed {int(local_count.item())} images",
            flush=True,
        )

        local_stats = torch.stack([local_musiq_sum[0], local_count[0]])
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        total_count = int(local_stats[1].item())
        set_musiq = (local_stats[0] / local_stats[1]).item() if total_count > 0 else 0.0
        per_rank_counts = [int(local_count.item())]
        if world_size > 1:
            per_rank_counts = [0 for _ in range(world_size)]
            dist.all_gather_object(per_rank_counts, int(local_count.item()))
        if rank == 0:
            print(
                f"[Stage 0][RealEval] {set_name} per-rank counts: {per_rank_counts} "
                f"(sum={sum(per_rank_counts)}/{len(eval_dataset)})"
            )
            results[set_name] = {
                "musiq": set_musiq,
                "count": total_count,
            }
            if writer:
                writer.add_scalar(f"RealVal_{set_name}/MUSIQ", set_musiq, epoch)

    timeout_sync = torch.tensor([1 if timed_out else 0], device=device, dtype=torch.int32)
    dist.all_reduce(timeout_sync, op=dist.ReduceOp.MAX)
    timed_out = bool(int(timeout_sync.item()))

    if rank == 0:
        musiq_sum = 0.0
        valid_count = 0

        print(f"\n{'-' * 80}")
        print(f"[Stage 0][RealEval] Per-set MUSIQ @ Epoch {epoch}")
        print(f"{'-' * 80}")
        for set_name in all_set_names:
            if set_name not in results:
                continue
            row = results[set_name]
            print(f"{set_name:24s} MUSIQ: {row['musiq']:.4f} ({row['count']} imgs)")
            musiq_sum += row["musiq"]
            valid_count += 1

        avg_musiq = musiq_sum / valid_count if valid_count > 0 else 0.0
        print(f"{'-' * 80}")
        print(f"{'RealEval Avg':24s} MUSIQ: {avg_musiq:.4f}")
        if timed_out:
            print(f"[Stage 0][RealEval] Stopped by timeout ({timeout_seconds}s).")
        print(f"{'=' * 80}\n")

        if writer:
            writer.add_scalar("RealVal/Avg_MUSIQ", avg_musiq, epoch)

        return avg_musiq, timed_out

    return 0.0, timed_out


def train_ddp(cfg):
    # Setup DDP (torchrun sets environment variables automatically)
    rank, world_size, local_rank = setup_ddp()

    device = torch.device(f"cuda:{local_rank}")

    # A100 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Use BF16 or FP16 based on config (A100 prefers BF16)
    use_bf16 = getattr(cfg.train, "use_bf16", False)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if rank == 0:
        print(f"\n{'='*50}")
        print("A100 Optimizations Enabled:")
        print(f"  TF32 (matmul):       {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  TF32 (cudnn):        {torch.backends.cudnn.allow_tf32}")
        print(f"  cudnn.benchmark:     {torch.backends.cudnn.benchmark}")
        print(f"  cudnn.deterministic: {torch.backends.cudnn.deterministic}")
        print(f"  AMP dtype:           {'BF16' if use_bf16 else 'FP16'}")
        print(f"{'='*50}\n")
        print(f"Using {world_size} GPUs for training")
        print(f"Process rank {rank}, local_rank {local_rank}, device: {device}")

    seed_everything(cfg.train.seed + rank)

    if rank == 0:
        os.makedirs(os.path.join(cfg.train.save_dir, cfg.exp_name), exist_ok=True)
        os.makedirs(os.path.join(cfg.train.log_dir, cfg.exp_name), exist_ok=True)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.train.log_dir, cfg.exp_name))
    else:
        writer = None

    # Initialize dataset configuration from config file
    # Convert namespace to dict for utils_incremental
    config_dict = {
        "deg_map": vars(cfg.deg_map) if hasattr(cfg, "deg_map") else {},
        "incremental": {
            "stage": cfg.incremental.stage if hasattr(cfg, "incremental") else 0,
            "inc_class_num": cfg.incremental.inc_class_num if hasattr(cfg, "incremental") else 0,
            "base_class_num": cfg.incremental.base_class_num if hasattr(cfg, "incremental") else 12,
        },
        "cgcd": {
            "class_order": cfg.cgcd.class_order if hasattr(cfg.cgcd, "class_order") else list(range(12)),
            "class_mappings": cfg.cgcd.class_mappings if hasattr(cfg.cgcd, "class_mappings") else None,
        },
        "train": {
            "data_root_train": (
                cfg.train.data_root_train if hasattr(cfg.train, "data_root_train") else cfg.train.train_data
            ),
            "data_root_val": cfg.train.data_root_val if hasattr(cfg.train, "data_root_val") else cfg.train.val_data,
            "all_classes": cfg.train.all_classes if hasattr(cfg.train, "all_classes") else [],
        },
    }

    # Initialize dataset configuration (all ranks need this)
    init_from_config(config_dict)

    if rank == 0:
        print("\n" + "=" * 50)
        print("Dataset Configuration Initialized")
        print("=" * 50)

        # Debug: Print ORIG2CLASSIFIER to verify it's loaded
        from datasets_wres import ORIG2CLASSIFIER

        print(f"\n[DEBUG] ORIG2CLASSIFIER after init:")
        print(f"  {ORIG2CLASSIFIER}\n")

        if hasattr(cfg, "incremental") and cfg.incremental.stage > 0:
            print_incremental_info(config_dict)

    use_validation = bool(getattr(cfg.train, "use_validation", True))
    use_class_balanced_sampler = bool(getattr(cfg.train, "use_class_balanced_sampler", False))
    max_samples_per_class = int(getattr(cfg.train, "max_samples_per_class", 0) or 0)
    use_real_eval = bool(getattr(cfg.train, "use_real_eval", False))
    real_eval_precheck_before_train = bool(getattr(cfg.train, "real_eval_precheck_before_train", True))
    real_eval_freq = int(getattr(cfg.train, "real_eval_freq", 100) or 100)
    real_eval_timeout_seconds = int(getattr(cfg.train, "real_eval_timeout_seconds", 3600) or 0)
    real_eval_max_side = int(getattr(cfg.train, "real_eval_max_side", 1280) or 0)
    real_eval_num_workers = int(getattr(cfg.train, "real_eval_num_workers", 2) or 2)
    real_eval_root = str(getattr(cfg.train, "real_eval_root", "") or "").strip()
    real_eval_sets = _parse_real_eval_sets(getattr(cfg.train, "real_eval_sets", []))

    # Generate train/val data automatically from config (all ranks need this)
    train_data_old, train_data_new, val_data_all = get_train_val_data_for_stage(config_dict)

    if rank == 0:
        print("\n" + "=" * 50)
        print("[Stage 0] Creating Training Dataset (OLD classes only)...")
        print("=" * 50)

    train_dataset = create_train_datasets(train_data_old, patch_size=224)

    train_labels = _extract_labels_from_dataset(train_dataset)
    if max_samples_per_class > 0:
        train_dataset, train_labels, cap_stats = _apply_per_class_cap(
            train_dataset,
            train_labels,
            max_samples_per_class=max_samples_per_class,
            seed=cfg.train.seed,
        )
        if rank == 0 and cap_stats is not None:
            print("\n" + "=" * 50)
            print("[Stage 0] Per-class cap applied")
            print("=" * 50)
            print(f"  max_samples_per_class: {cap_stats['max_samples_per_class']}")
            print(f"  total: {cap_stats['original_total']} -> {cap_stats['capped_total']}")
            print(f"  class counts (before): {cap_stats['before']}")
            print(f"  class counts (after):  {cap_stats['after']}")

    val_datasets = []
    if use_validation:
        if rank == 0:
            print("\n" + "=" * 50)
            print("[Stage 0] Creating Validation Datasets (ALL classes for zero-shot on NEW)...")
            print("=" * 50)

        # Stage 0 validation: ALL classes (including NEW for zero-shot evaluation)
        from datasets_wres import create_stage0_val_datasets

        val_datasets = create_stage0_val_datasets(val_data_all)
    elif rank == 0:
        print("\n" + "=" * 50)
        print("[Stage 0] Validation disabled (train.use_validation=False)")
        print("=" * 50)

    real_eval_datasets = []
    if use_real_eval:
        if not real_eval_root:
            if rank == 0:
                print("\n" + "=" * 50)
                print("[Stage 0][RealEval] Disabled: train.real_eval_root is empty")
                print("=" * 50)
        elif len(real_eval_sets) == 0:
            if rank == 0:
                print("\n" + "=" * 50)
                print("[Stage 0][RealEval] Disabled: train.real_eval_sets is empty")
                print("=" * 50)
        else:
            if rank == 0:
                print("\n" + "=" * 50)
                print("[Stage 0] Creating Real Validation Datasets (unpaired IQA)...")
                print("=" * 50)
                print(f"  root: {real_eval_root}")
                print(f"  sets: {real_eval_sets}")
                print(f"  freq: every {real_eval_freq} epochs")
                print(f"  timeout: {real_eval_timeout_seconds} sec")
            real_eval_datasets = create_real_eval_datasets(real_eval_root, real_eval_sets)
            if rank == 0:
                print(f"  loaded real eval sets: {len(real_eval_datasets)}")

    if use_class_balanced_sampler:
        weights, class_counts = _build_class_balanced_weights(train_labels)
        train_sampler = DistributedWeightedSampler(
            weights=weights,
            num_replicas=world_size,
            rank=rank,
            replacement=True,
            num_samples=int(math.ceil(len(train_dataset) / float(world_size))),
            seed=cfg.train.seed,
        )
        if rank == 0:
            print("\n" + "=" * 50)
            print("[Stage 0] Class-balanced sampler enabled")
            print("=" * 50)
            print(f"  class counts: {class_counts}")
    else:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.train.seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,  # Use DistributedSampler instead of shuffle
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,  # Each worker prefetches 2 batches
        persistent_workers=True,  # Keep workers alive between epochs
    )

    if rank == 0:
        print(f"\nTrain batches per GPU: {len(train_loader)}")
        print(f"Total train batches: {len(train_loader) * world_size}")

    if rank == 0:
        print("\n" + "=" * 50)
        print("Creating Models...")
        print("=" * 50)

    dino_model = load_finetuned_model_from_checkpoint(
        checkpoint_dir=cfg.cgcd.dino_checkpoint,
        num_classes=cfg.cgcd.nclasses,
        model_name=cfg.cgcd.model_name,
        device=device,
    )
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False

    # CGCD model (trainable) - dynamically selected by cfg.cgcd.arch
    cgcd_arch = getattr(cfg.cgcd, "arch", "hard")
    if cgcd_arch not in CGCD_ARCH_REGISTRY:
        available_arches = ", ".join(sorted(CGCD_ARCH_REGISTRY))
        raise ValueError(f"Unsupported cgcd arch '{cgcd_arch}'. Available: {available_arches}")
    cgcd_class_name = CGCD_ARCH_REGISTRY[cgcd_arch]
    cgcd_module = CGCD_ARCH_MODULE_REGISTRY.get(cgcd_arch, cgcd_models)
    CGCDClass = getattr(cgcd_module, cgcd_class_name, None)
    if CGCDClass is None:
        raise ValueError(f"Unsupported cgcd arch '{cgcd_arch}' for adain trainer")
    if rank == 0:
        print(f"CGCD arch: {cgcd_arch} -> {cgcd_class_name}")
    cgcd_model = CGCDClass(
        saved_models_dir=cfg.cgcd.saved_vcgcd_models_dir,
        stage=0,
        pca_path=cfg.cgcd.pca_path,
        output_dim=cfg.cgcd.embd_dim,  # Output dimension for OneRestore (324)
    ).to(device)

    # OneRestore model (trainable)
    model = OneRestore(channel=cfg.model.width).to(device)
    if hasattr(model, "set_runtime_resize_fallback"):
        model.set_runtime_resize_fallback(True)
        if rank == 0:
            print("[Stage 0] Enabled OneRestore runtime resize fallback for large real-eval images")

    # Optional torch.compile for PyTorch 2.0+
    use_compile = getattr(cfg.train, "use_compile", False)
    if use_compile:
        if rank == 0:
            print(f"Using torch.compile() for model optimization...")
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    cgcd_model = DDP(cgcd_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if rank == 0:
        print(f"DINO Model: {count_params(dino_model):,} parameters (frozen)")
        print(f"CGCD Model (with projection): {count_params(cgcd_model):,} parameters (trainable)")
        print(f"OneRestore: {count_params(model):,} parameters")
        print(f"Total trainable: {count_params(cgcd_model) + count_params(model):,} parameters")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cgcd_model.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    save_latest = bool(getattr(cfg.train, "save_latest", False))

    # GradScaler only needed for FP16, not for BF16
    scaler = GradScaler("cuda") if not use_bf16 else None

    start_epoch = 1
    global_step = 0
    resume_path = getattr(cfg.train, "resume", None)
    if resume_path and os.path.exists(resume_path):  # 모든 rank에서 로드
        start_epoch, global_step = load_checkpoint(
            resume_path,
            model.module,
            cgcd_model.module,
            optimizer,
            scheduler,
        )
        if rank == 0:
            print(f"👍resume checkpoint from {resume_path}")

    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = start_epoch_tensor.item()

    if rank == 0:
        print("\n" + "=" * 50)
        print("Starting Training...")
        print("=" * 50)

    best_psnr = 0.0

    if use_real_eval and real_eval_precheck_before_train and len(real_eval_datasets) > 0:
        if rank == 0:
            print("\n" + "=" * 80)
            print("[Stage 0][RealEval Precheck] Running one real-world validation before epoch start...")
            print("=" * 80)
        precheck_musiq, precheck_timed_out = validate_real_unpaired_multi_gpu(
            model.module,
            dino_model,
            cgcd_model,
            real_eval_datasets,
            device,
            epoch=0,
            cfg=cfg,
            writer=writer,
            rank=rank,
            world_size=world_size,
            amp_dtype=amp_dtype,
            timeout_seconds=real_eval_timeout_seconds,
            max_eval_side=real_eval_max_side,
            num_workers=real_eval_num_workers,
            real_val_metrics=None,
        )
        if rank == 0:
            print(
                f"[Stage 0][RealEval Precheck] Avg MUSIQ: {precheck_musiq:.4f}, "
                f"timeout_triggered={precheck_timed_out}"
            )

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{cfg.train.epochs}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*50}")

        global_step, train_loss = train_one_epoch(
            model,
            dino_model,
            cgcd_model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            cfg,
            writer,
            global_step,
            rank,
            amp_dtype,
        )

        scheduler.step()

        if rank == 0:
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Train/Loss", train_loss, epoch)

        if epoch % cfg.train.val_freq == 0 and len(val_datasets) > 0:
            # Validation metrics are created inside validate_multi_gpu when needed
            # This saves GPU memory during training (metrics only loaded during validation)
            avg_psnr_old, avg_psnr_new, avg_psnr_all = validate_multi_gpu(
                model.module,
                dino_model,
                cgcd_model,
                val_datasets,
                device,
                epoch,
                cfg,
                writer,
                rank,
                world_size,
                amp_dtype,
                val_metrics=None,  # Metrics created on-demand inside function
            )

            if rank == 0:
                print(
                    f"[Stage 0 Summary] OLD: {avg_psnr_old:.2f} dB, "
                    f"NEW (zero-shot): {avg_psnr_new:.2f} dB, "
                    f"ALL: {avg_psnr_all:.2f} dB"
                )

                # Best model selected based on OLD classes only (NEW is zero-shot)
                is_best = avg_psnr_old > best_psnr
                if is_best:
                    best_psnr = avg_psnr_old
                    print(f"[Stage 0] New best PSNR (OLD): {best_psnr:.2f} dB")
            else:
                is_best = False
        else:
            is_best = False

        if use_real_eval and len(real_eval_datasets) > 0 and (epoch % real_eval_freq == 0):
            avg_real_musiq, real_timed_out = validate_real_unpaired_multi_gpu(
                model.module,
                dino_model,
                cgcd_model,
                real_eval_datasets,
                device,
                epoch,
                cfg,
                writer,
                rank,
                world_size,
                amp_dtype=amp_dtype,
                timeout_seconds=real_eval_timeout_seconds,
                max_eval_side=real_eval_max_side,
                num_workers=real_eval_num_workers,
                real_val_metrics=None,
            )
            if rank == 0:
                print(
                    f"[Stage 0][RealEval Summary] Avg MUSIQ: {avg_real_musiq:.4f}, "
                    f"timeout_triggered={real_timed_out}"
                )

        if rank == 0:
            if epoch % cfg.train.save_freq == 0 or epoch == cfg.train.epochs:
                save_path = os.path.join(cfg.train.save_dir, cfg.exp_name, f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(model.module, cgcd_model.module, optimizer, scheduler, epoch, save_path, is_best)

            if save_latest:
                latest_path = os.path.join(cfg.train.save_dir, cfg.exp_name, "checkpoint_latest.pth")
                save_checkpoint(model.module, cgcd_model.module, optimizer, scheduler, epoch, latest_path, False)

        dist.barrier()

    if rank == 0:
        print("\n" + "=" * 50)
        print("[Stage 0] Training completed!")
        print(f"Best validation PSNR (OLD classes): {best_psnr:.2f} dB")
        print("=" * 50)
        writer.close()

    # Cleanup
    dist.destroy_process_group()


def main():
    """
    Main function - called by torchrun for each process.

    MULTI-GPU VALIDATION: All GPUs participate in validation with round-robin assignment.

    Usage:
        # 4 GPUs on single node with A100 optimizations + Multi-GPU Validation
        torchrun --nproc_per_node=4 train_wcgcd_onerestore_old.py --config configs/train_wcgcd_onerestore.yml --exp_name exp_onerestore

        # Specific GPUs
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_wcgcd_onerestore_old.py --config configs/train_wcgcd_onerestore.yml --exp_name exp_onerestore
    """

    # A100 memory allocator optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        description="Train OneRestore with CGCD using DDP, A100 optimizations, and Multi-GPU Validation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/onerestore/000_onerestore_wres_stage0.yml",
        help="Path to config file",
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--use_compile", action="store_true", help="Initialize pseudo labels with teacher before training"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cfg = dict2namespace(config)
    cfg.exp_name = args.exp_name
    cfg.use_compile = args.use_compile

    train_ddp(cfg)


if __name__ == "__main__":
    main()
