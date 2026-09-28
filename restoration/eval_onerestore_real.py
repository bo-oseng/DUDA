"""OneRestore + CGCD training and evaluation utilities."""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path

import builtins
import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import save_image

import pyiqa
import pandas as pd

import cgcd.models as cgcd_models
import cgcd.models_margins as cgcd_models_margins
import cgcd.models_cov as cgcd_models_cov
import cgcd.models_cov_perturb as cgcd_models_cov_perturb
from models.OneRestore import OneRestore
import datasets_wres_continual as datasets_module
from datasets_wres_continual import (
    RefDegImage,
    create_real_eval_datasets,
    create_stage1_val_datasets,
    init_from_config,
)
from feature_extractor.sl_finetuned_model import load_finetuned_model_from_checkpoint
from utils_lib.utils import count_params, dict2namespace, seed_everything
from utils_lib.helper import create_fgresq_metric
from utils_lib.utils_dataset_wres import get_train_val_data_for_stage
from metrics import pt_psnr, pt_ssim


CGCD_ARCH_REGISTRY = {
    "soft": "CGCDSignalModuleSoft",
    "soft_no_le": "CGCDSignalModuleStaticSoftFixedEmbedding",
    "softmargin": "CGCDSignalModuleSoftMargin",
    "hard": "CGCDSignalModuleHard",
    "staticsoft": "CGCDSignalModuleStaticSoft",
    "statichard": "CGCDSignalModuleStaticHard",
    "hard_le": "CGCDSignalModuleStaticHardLeanableEmbedding",
    "hard_le_finst": "CGCDSignalModuleStaticHardLeanableEmbeddingInst",
    "hard_le_soft_finst": "CGCDSignalModuleStaticHardLeanableEmbeddingSoftInst",
    "hard_top1_no_le": "CGCDSignalModuleStaticHardFixedEmbedding",
    "prompt": "CGCDSignalModulePrompt",
    "learnable": "CGCDSignalModuleLearnable",
    "adain": "CGCDSignalModuleControlNet",
    "adain_wo_zero_init": "CGCDSignalModuleControlNetNoZeroInit",
    "mmdit": "CGCDSignalModuleMMdit",
    "adain_mmdit": "CGCDSignalModuleMMdit",
    "adain_weight_lamda": "CGCDSignalModuleWeightLamda",
    "adain_weight_lambda": "CGCDSignalModuleWeightLamda",
    "adain_weight_perturb": "CGCDSignalModuleControlNetWeightPerturb",
    "adain_residual_dropout": "CGCDSignalModuleControlNetResidualDropout",
    "adain_residual_channel_dropout": "CGCDSignalModuleControlNetResidualChannelDropout",
    "adain_residual_chaneel_dropout": "CGCDSignalModuleControlNetResidualChannelDropout",
    "adain_residual_uniformmul": "CGCDSignalModuleControlNetResidualUniformMul",
}


IQA_METRICS = [("musiq", "MUSIQ"), ("clipiqa", "CLIP-IQA"), ("liqe", "LIQE")]

CGCD_PRIMARY_MODES = ("clear", "contrastive")
CGCD_VER2_MODES = ("mahalanobis_margin", "mahalanobis_pca")
CGCD_ALL_MODES = CGCD_PRIMARY_MODES + CGCD_VER2_MODES

# eval에 쓰이는 이지미들은 224, 224로 줄여서 classfication 하는게 적절한지?
# 만약 다르게한다면?
transform_resize = v2.Compose(
    [
        v2.Resize([224, 224]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def extract_base_state(state_dict):
    base_state = {}
    for key, value in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        if key.startswith("base_model.model."):
            key = key.replace("base_model.model.", "")
        key = key.replace(".base_layer.", ".")
        base_state[key] = value
    return base_state


def _install_qalign_compat():
    # Q-Align's cached HF module references older transformers symbols as globals.
    try:
        from transformers.cache_utils import Cache
        from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    except Exception:
        return
    builtins.Cache = Cache
    builtins.BaseModelOutputWithPast = BaseModelOutputWithPast
    builtins.CausalLMOutputWithPast = CausalLMOutputWithPast


def build_iqa_metrics(device, rank, use_vlm_vis=False):
    metrics = {}
    available = []
    active_metrics = list(IQA_METRICS)
    if use_vlm_vis:
        active_metrics.append(("vlm_vis", "VLM-Vis"))

    for metric_name, display_name in active_metrics:
        try:
            if metric_name == "fgresq":
                metrics[metric_name] = create_fgresq_metric(device=device)
            elif metric_name == "vlm_vis":
                from WResVLM.tools.rate_scripts.llava.vlm_vis_metric import VLMVisMetric

                metrics[metric_name] = VLMVisMetric(device=device)
            else:
                if metric_name == "qalign":
                    _install_qalign_compat()
                metrics[metric_name] = pyiqa.create_metric(metric_name, as_loss=False, device=device)
            available.append((metric_name, display_name))
        except Exception as e:
            raise RuntimeError(f"Failed to load required metric {display_name}") from e
    if not available:
        raise RuntimeError("No IQA metrics were loaded. Check your pyiqa installation and checkpoints.")
    return metrics, available


def preprocess_for_metric(metric_name, image):
    """
    Apply metric-specific input constraints without affecting other metrics.
    image: [B, C, H, W] in [0, 1]
    """
    if metric_name == "liqe":
        _, _, h, w = image.shape
        short_side = min(h, w)
        if short_side < 224:
            scale = 224.0 / float(short_side)
            new_h = max(224, int(round(h * scale)))
            new_w = max(224, int(round(w * scale)))
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return image


def build_summary_table_rows(summary):
    metric_names = list(summary.get("metric_names", []))
    per_set = dict(summary.get("per_set", {}))
    rows = []
    for set_name, stats in per_set.items():
        row = {
            "model": summary.get("model"),
            "set": set_name,
            "count": stats.get("count"),
            "reused": stats.get("reused"),
            "inferred": stats.get("inferred"),
        }
        for metric_name in metric_names:
            row[metric_name] = stats.get(metric_name)
        rows.append(row)

    avg_row = {
        "model": summary.get("model"),
        "set": "AVG",
        "count": sum((row.get("count") or 0) for row in rows),
        "reused": sum((row.get("reused") or 0) for row in rows),
        "inferred": sum((row.get("inferred") or 0) for row in rows),
    }
    for metric_name in metric_names:
        avg_row[metric_name] = summary.get("average", {}).get(metric_name)
    rows.append(avg_row)
    return rows


def save_summary_tables(summary, json_path, prefix):
    metric_names = list(summary.get("metric_names", []))
    metric_labels = dict(summary.get("metric_labels", {}))
    rows = build_summary_table_rows(summary)
    if not rows:
        return

    csv_path = os.path.splitext(json_path)[0] + ".csv"
    fieldnames = ["model", "set", "count", "reused", "inferred"] + metric_names
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{prefix} Saved CSV summary: {csv_path}")

    long_csv_path = os.path.splitext(json_path)[0] + "_long.csv"
    with open(long_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "set", "metric_name", "metric_label", "value", "count", "reused", "inferred"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for metric_name in metric_names:
                writer.writerow(
                    {
                        "model": row["model"],
                        "set": row["set"],
                        "metric_name": metric_name,
                        "metric_label": metric_labels.get(metric_name, metric_name),
                        "value": row.get(metric_name),
                        "count": row.get("count"),
                        "reused": row.get("reused"),
                        "inferred": row.get("inferred"),
                    }
                )
    print(f"{prefix} Saved long CSV summary: {long_csv_path}")

    xlsx_path = os.path.splitext(json_path)[0] + ".xlsx"
    wide_df = pd.DataFrame(rows)
    long_rows = []
    for row in rows:
        for metric_name in metric_names:
            long_rows.append(
                {
                    "model": row["model"],
                    "set": row["set"],
                    "metric_name": metric_name,
                    "metric_label": metric_labels.get(metric_name, metric_name),
                    "value": row.get(metric_name),
                    "count": row.get("count"),
                    "reused": row.get("reused"),
                    "inferred": row.get("inferred"),
                }
            )
    long_df = pd.DataFrame(long_rows)
    labels_df = pd.DataFrame(
        [
            {"metric_name": metric_name, "metric_label": metric_labels.get(metric_name, metric_name)}
            for metric_name in metric_names
        ]
    )
    meta_df = pd.DataFrame(
        [
            {"key": "model", "value": summary.get("model")},
            {"key": "num_sets", "value": summary.get("num_sets")},
            {"key": "json_path", "value": json_path},
        ]
    )
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            wide_df.to_excel(writer, sheet_name="summary_wide", index=False)
            long_df.to_excel(writer, sheet_name="summary_long", index=False)
            labels_df.to_excel(writer, sheet_name="metric_labels", index=False)
            meta_df.to_excel(writer, sheet_name="meta", index=False)
        print(f"{prefix} Saved XLSX summary: {xlsx_path}")
    except Exception as e:
        print(f"{prefix} Failed to save XLSX summary ({xlsx_path}): {e}")


def save_per_sample_metric_csvs(sample_records, metric_names, output_dir, model_tag, prefix):
    if output_dir is None or not sample_records or not metric_names:
        return

    os.makedirs(output_dir, exist_ok=True)
    base_columns = ["set", "file"]
    ordered_columns = base_columns + list(metric_names)
    ordered_records = []
    for record in sample_records:
        row = {column: record.get(column) for column in ordered_columns}
        ordered_records.append(row)

    wide_df = pd.DataFrame(ordered_records, columns=ordered_columns)
    wide_path = os.path.join(output_dir, f"{model_tag}_saved_metrics_per_sample.csv")
    wide_df.to_csv(wide_path, index=False)
    print(f"{prefix} Saved per-sample CSV: {wide_path}")

    for metric_name in metric_names:
        metric_path = os.path.join(output_dir, f"{model_tag}_saved_{metric_name}_per_sample.csv")
        metric_df = wide_df[base_columns + [metric_name]]
        metric_df.to_csv(metric_path, index=False)
        print(f"{prefix} Saved per-sample metric CSV: {metric_path}")


def parse_csv_tokens(csv_str):
    if not csv_str:
        return None
    if isinstance(csv_str, (list, tuple)):
        values = [str(v).strip() for v in csv_str]
    else:
        values = [s.strip() for s in str(csv_str).split(",")]
    return [v for v in values if v]


def build_cgcd_modes(primary_mode, extra_modes):
    modes = [str(primary_mode).strip()]
    if extra_modes:
        for mode in extra_modes:
            mode = str(mode).strip()
            if mode and mode not in modes:
                modes.append(mode)
    return modes


@torch.no_grad()
def compute_cgcd_scores_multi(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    modes,
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
    mahalanobis_temp=20.0,
    mahalanobis_pca_top_k=32,
):
    def _manual_preprocess_for_mahalanobis(feat):
        feat_out = feat
        pca_layer = getattr(cgcd_model, "pca_layer", None)
        if pca_layer is not None:
            feat_out = pca_layer(feat_out)

        scaler_mean = getattr(cgcd_model, "scaler_mean", None)
        scaler_scale = getattr(cgcd_model, "scaler_scale", None)
        if (
            isinstance(scaler_mean, torch.Tensor)
            and isinstance(scaler_scale, torch.Tensor)
            and scaler_mean.numel() > 0
            and scaler_scale.numel() > 0
            and feat_out.shape[-1] == scaler_mean.shape[0]
            and feat_out.shape[-1] == scaler_scale.shape[0]
        ):
            feat_out = (feat_out - scaler_mean) / scaler_scale
        return feat_out

    def _build_inv_sqrt_covs(covs, top_k=None):
        covs = covs.float()
        evals, evecs = torch.linalg.eigh(covs)
        evals = evals.clamp(min=1e-8)
        inv_sqrt_evals = 1.0 / torch.sqrt(evals)

        if top_k is not None:
            dim = inv_sqrt_evals.shape[-1]
            k = int(max(1, min(int(top_k), dim)))
            mask_top = torch.zeros_like(inv_sqrt_evals)
            # eigh 오름차순 정렬이므로 뒤쪽이 큰 고유값(top-k).
            mask_top[:, -k:] = 1.0
            inv_sqrt_evals = inv_sqrt_evals * mask_top

        return torch.bmm(evecs, torch.bmm(torch.diag_embed(inv_sqrt_evals), evecs.mT))

    use_amp = amp_dtype is not None and image_list.is_cuda
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        dino_feat = dino_model(dino_transform(image_list)).pooler_output
        if hasattr(cgcd_model, "_preprocess_feature"):
            preprocess_fn = getattr(cgcd_model, "_preprocess_feature")
            try:
                processed_feat = preprocess_fn(
                    dino_feat, cgcd_model.pca_layer, cgcd_model.scaler_mean, cgcd_model.scaler_scale
                )
            except TypeError:
                processed_feat = preprocess_fn(dino_feat)
        else:
            processed_feat = _manual_preprocess_for_mahalanobis(dino_feat)
        _, logits = cgcd_model(dino_feat)

    logits = logits.float()
    probs = F.softmax(logits, dim=-1)
    n_classes = logits.shape[1]
    if clear_idx < 0 or clear_idx >= n_classes:
        raise ValueError(f"clear_idx out of range: {clear_idx}, num_classes={n_classes}")

    means_dim = int(cgcd_model.means.shape[-1]) if hasattr(cgcd_model, "means") else int(processed_feat.shape[-1])
    if int(processed_feat.shape[-1]) != means_dim:
        processed_feat = _manual_preprocess_for_mahalanobis(dino_feat)
    if int(processed_feat.shape[-1]) != means_dim:
        need_mahalanobis = any(m in ("mahalanobis_margin", "mahalanobis_pca") for m in modes)
        if need_mahalanobis:
            raise RuntimeError(
                "CGCD preprocessing dimension mismatch for mahalanobis modes: "
                f"processed_feat_dim={int(processed_feat.shape[-1])}, means_dim={means_dim}. "
                "Check cgcd pca/scaler config."
            )

    mask = torch.ones(n_classes, device=logits.device, dtype=torch.bool)
    mask[clear_idx] = False
    neg_logits = logits[:, mask]

    tau = max(float(contrastive_tau), 1e-6)
    score_temp = max(float(contrastive_score_temp), 1e-6)
    w_pos = float(contrastive_pos_weight)
    w_neg = float(contrastive_neg_weight)
    maha_temp = max(float(mahalanobis_temp), 1e-6)

    scores = {}
    dist_sq_all = None
    for mode in modes:
        if mode == "clear":
            scores[mode] = probs[:, clear_idx]
        elif mode == "contrastive":
            pos_logit = logits[:, clear_idx]
            if neg_logits.shape[1] == 0:
                neg_agg = torch.zeros_like(pos_logit)
            else:
                neg_agg = tau * torch.logsumexp(neg_logits / tau, dim=-1)
            margin_like = (w_pos * pos_logit) - (w_neg * neg_agg)
            scores[mode] = torch.sigmoid(margin_like / score_temp)
        elif mode == "mahalanobis_margin":
            if dist_sq_all is None:
                diffs = processed_feat.float().unsqueeze(1) - cgcd_model.means.float()
                if hasattr(cgcd_model, "inv_covs"):
                    dist_sq_all = torch.einsum("bci,cij,bcj->bc", diffs, cgcd_model.inv_covs.float(), diffs)
                elif hasattr(cgcd_model, "inv_sqrt_covs"):
                    whitened_diff = torch.einsum("cij,bcj->bci", cgcd_model.inv_sqrt_covs.float(), diffs)
                    dist_sq_all = torch.sum(whitened_diff * whitened_diff, dim=-1)
                elif hasattr(cgcd_model, "covs"):
                    inv_sqrt_covs = _build_inv_sqrt_covs(cgcd_model.covs, top_k=None)
                    whitened_diff = torch.einsum("cij,bcj->bci", inv_sqrt_covs, diffs)
                    dist_sq_all = torch.sum(whitened_diff * whitened_diff, dim=-1)

            dist_clear = dist_sq_all[:, clear_idx]
            dist_neg = dist_sq_all[:, mask]
            if dist_neg.shape[1] == 0:
                agg_dist_neg = torch.zeros_like(dist_clear)
            else:
                agg_dist_neg = -tau * torch.logsumexp(-dist_neg / tau, dim=-1)
            margin = agg_dist_neg - dist_clear
            scores[mode] = torch.sigmoid(margin / maha_temp)

    return scores


def get_cfg_cgcd_value(cfg, key, default):
    cgcd_cfg = getattr(cfg, "cgcd", None)
    if cgcd_cfg is not None and hasattr(cgcd_cfg, key):
        return getattr(cgcd_cfg, key)
    train_cfg = getattr(cfg, "train", None)
    if train_cfg is not None and hasattr(train_cfg, key):
        return getattr(train_cfg, key)
    return default


def load_saved_restored_image(path, device):
    image = read_image(path).float() / 255.0
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    return image.unsqueeze(0).to(device, non_blocking=True)


def build_wres_dataset_config(cfg):
    if hasattr(cfg, "deg_map"):
        deg_map = cfg.deg_map if isinstance(cfg.deg_map, dict) else vars(cfg.deg_map)
    else:
        deg_map = {}

    cgcd_cfg = {
        "class_order": cfg.cgcd.class_order if hasattr(cfg, "cgcd") and hasattr(cfg.cgcd, "class_order") else "",
    }
    class_mappings = getattr(cfg.cgcd, "class_mappings", None) if hasattr(cfg, "cgcd") else None
    if class_mappings:
        cgcd_cfg["class_mappings"] = class_mappings

    train_cfg = {
        "data_root_train": getattr(cfg.train, "data_root_train", ""),
        "data_root_val": getattr(cfg.train, "data_root_val", getattr(cfg.train, "data_root_train", "")),
        "all_classes": list(getattr(cfg.train, "all_classes", [])),
        "input_subdir": getattr(cfg.train, "input_subdir", "input"),
        "gt_subdir": getattr(cfg.train, "gt_subdir", "gt"),
    }

    return {
        "deg_map": deg_map,
        "incremental": {
            "stage": cfg.incremental.stage if hasattr(cfg, "incremental") else 0,
            "inc_class_num": cfg.incremental.inc_class_num if hasattr(cfg, "incremental") else 0,
            "base_class_num": cfg.incremental.base_class_num if hasattr(cfg, "incremental") else 12,
        },
        "cgcd": cgcd_cfg,
        "train": train_cfg,
    }


def create_labeled_val_datasets_from_cfg(cfg, rank):
    config_dict = build_wres_dataset_config(cfg)
    init_from_config(config_dict)
    _, _, val_data_all = get_train_val_data_for_stage(config_dict)
    val_datasets = create_stage1_val_datasets(val_data_all)
    if rank == 0:
        print(f"[LabeledEval] Loaded paired validation sets: {len(val_datasets)}")
    return val_datasets


def resolve_clear_idx(cfg, rank):
    clear_idx = 0
    class_mappings_path = getattr(cfg.cgcd, "class_mappings", None) if hasattr(cfg, "cgcd") else None
    if class_mappings_path and os.path.exists(class_mappings_path):
        try:
            with open(class_mappings_path, "r", encoding="utf-8") as f:
                class_mappings = json.load(f)
            clear_idx = int(class_mappings["orig2classifier"]["0"])
        except Exception as e:
            if rank == 0:
                print(f"[Warning] Failed to read clear_idx from class_mappings ({class_mappings_path}): {e}")
    elif rank == 0:
        print("[Warning] class_mappings not found. Fallback clear_idx=0.")
    return clear_idx


def load_models(cfg, checkpoint_path, device, rank):
    # DINO
    dino_model = load_finetuned_model_from_checkpoint(
        checkpoint_dir=cfg.cgcd.dino_checkpoint,
        num_classes=cfg.cgcd.nclasses,
        model_name=cfg.cgcd.model_name,
        device=device,
    )
    dino_model.eval()
    for p in dino_model.parameters():
        p.requires_grad = False

    # CGCD
    cgcd_arch = getattr(cfg.cgcd, "arch", "soft")
    if cgcd_arch not in CGCD_ARCH_REGISTRY:
        available_arches = ", ".join(sorted(CGCD_ARCH_REGISTRY))
        raise ValueError(f"Unsupported cgcd arch '{cgcd_arch}'. Available: {available_arches}")
    cgcd_class_name = CGCD_ARCH_REGISTRY[cgcd_arch]
    CGCDClass = (
        getattr(cgcd_models_cov_perturb, cgcd_class_name, None)
        or getattr(cgcd_models, cgcd_class_name, None)
        or getattr(cgcd_models_margins, cgcd_class_name, None)
        or getattr(cgcd_models_cov, cgcd_class_name, None)
    )
    if CGCDClass is None:
        raise ValueError(f"Unsupported cgcd arch '{cgcd_arch}' for eval")

    cgcd_kwargs = dict(
        saved_models_dir=cfg.cgcd.saved_vcgcd_models_dir,
        output_dim=cfg.cgcd.embd_dim,
        pca_path=cfg.cgcd.pca_path,
        scaler_path=getattr(cfg.cgcd, "scaler_path", None),
        stage=cfg.incremental.stage,
    )
    if cgcd_arch in {"adain_weight_lamda", "adain_weight_lambda"}:
        cgcd_kwargs["perturb_on_eval"] = bool(getattr(cfg.cgcd, "perturb_on_eval", False))
    elif cgcd_arch == "adain_weight_perturb":
        cgcd_kwargs["noise_std"] = float(getattr(cfg.cgcd, "noise_std", 0.01))
        cgcd_kwargs["perturb_on_eval"] = bool(getattr(cfg.cgcd, "perturb_on_eval", False))
    elif cgcd_arch in {"adain_residual_dropout", "adain_residual_channel_dropout", "adain_residual_chaneel_dropout"}:
        cgcd_kwargs["dropout_p"] = float(getattr(cfg.cgcd, "dropout_p", 0.4))
    elif cgcd_arch == "adain_residual_uniformmul":
        cgcd_kwargs["noise_range"] = float(getattr(cfg.cgcd, "noise_range", 0.3))
        cgcd_kwargs["perturb_on_eval"] = bool(getattr(cfg.cgcd, "perturb_on_eval", False))

    cgcd_model = CGCDClass(**cgcd_kwargs).to(device)
    cgcd_model.eval()

    student_model = OneRestore(channel=cfg.model.width).to(device)
    teacher_model = OneRestore(channel=cfg.model.width).to(device)
    student_model.eval()
    teacher_model.eval()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "student_state_dict" in checkpoint and "teacher_state_dict" in checkpoint:
        student_base = extract_base_state(checkpoint["student_state_dict"])
        teacher_base = extract_base_state(checkpoint["teacher_state_dict"])
    elif "teacher_state_dict" in checkpoint:
        teacher_base = extract_base_state(checkpoint["teacher_state_dict"])
        student_base = teacher_base
    elif "model_state_dict" in checkpoint:
        teacher_base = extract_base_state(checkpoint["model_state_dict"])
        student_base = teacher_base
    else:
        raise KeyError(f"Unsupported checkpoint format. Available keys: {list(checkpoint.keys())}")

    missing_s, unexpected_s = student_model.load_state_dict(student_base, strict=False)
    missing_t, unexpected_t = teacher_model.load_state_dict(teacher_base, strict=False)

    cgcd_state = checkpoint.get("cgcd_state_dict", checkpoint.get("cgcd_model_state_dict"))
    if cgcd_state is not None:
        try:
            cgcd_model.load_state_dict(cgcd_state, strict=True)
        except Exception:
            cgcd_model.load_state_dict(cgcd_state, strict=False)
            if rank == 0:
                print("[Warning] CGCD strict load failed. Loaded with strict=False.")
    elif rank == 0:
        print("[Warning] CGCD state dict not found in checkpoint.")

    epoch = int(checkpoint.get("epoch", 0))

    if rank == 0:
        print(f"CGCD arch: {cgcd_arch} -> {cgcd_class_name}")
        print(
            f"Loaded student weights ({len(student_base)} keys, missing={len(missing_s)}, unexpected={len(unexpected_s)})"
        )
        print(
            f"Loaded teacher weights ({len(teacher_base)} keys, missing={len(missing_t)}, unexpected={len(unexpected_t)})"
        )
        print(f"Loaded checkpoint epoch: {epoch}")
        print(f"DINO params (trainable): {count_params(dino_model):,}")
        print(f"CGCD params (trainable): {count_params(cgcd_model):,}")
        print(f"Student params (trainable): {count_params(student_model):,}")
        print(f"Teacher params (trainable): {count_params(teacher_model):,}")

    return student_model, teacher_model, dino_model, cgcd_model, epoch


@torch.no_grad()
def evaluate_labeled_paired(
    model,
    dino_model,
    cgcd_model,
    val_datasets,
    device,
    rank,
    world_size,
    use_ddp,
    amp_dtype=torch.float16,
    num_workers=2,
    output_dir=None,
    model_tag="teacher",
):
    if not val_datasets:
        if rank == 0:
            print(f"[LabeledEval][{model_tag}] No paired labeled validation datasets. Skip.")
        return None

    model.eval()
    dino_model.eval()
    cgcd_model.eval()

    deg_types_per_gpu = []
    all_deg_names = [deg_name for _, deg_name in val_datasets]
    for i, (dataset, deg_name) in enumerate(val_datasets):
        if i % world_size == rank:
            deg_types_per_gpu.append((dataset, deg_name))

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"[LabeledEval][{model_tag}] Metrics: PSNR")
        print(f"{'=' * 80}")

    local_results = {}
    for val_dataset, deg_name in syn_eval_datasets:
        local_indices = list(range(rank, len(val_dataset), world_size)) if use_ddp else list(range(len(val_dataset)))
        if not local_indices:
            continue
        local_dataset = Subset(val_dataset, local_indices)
        loader = DataLoader(
            local_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )

        psnr_scores = []
        iterator = tqdm(
            loader, desc=f"[GPU {rank}][LabeledEval] {deg_name:<15}", leave=False, position=rank, ncols=100
        )
        for batch in iterator:
            hq_image = batch[0].to(device, non_blocking=True)
            lq_image = batch[1].to(device, non_blocking=True)

            with autocast("cuda", dtype=amp_dtype):
                dino_feature = dino_model(transform_resize(lq_image)).pooler_output
                cgcd_embd, _ = cgcd_model(dino_feature)
                restored = model(lq_image, cgcd_embd)

            restored = torch.clamp(restored, 0, 1)
            psnr_scores.append(pt_psnr(hq_image, restored))

        set_psnr = torch.cat(psnr_scores, dim=0).mean().item() if psnr_scores else 0.0
        local_results[deg_name] = {"psnr": set_psnr, "count": len(psnr_scores)}
        print(
            f"[GPU {rank}][LabeledEval][{model_tag}] {deg_name:20s} - PSNR: {set_psnr:.4f} ({len(psnr_scores)} imgs)"
        )

    if use_ddp:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
    else:
        gathered_results = [local_results]

    if rank != 0:
        return None

    merged = {}
    for result_dict in gathered_results:
        if not result_dict:
            continue
        for deg_name, row in result_dict.items():
            merged_row = merged.setdefault(
                deg_name,
                {"count": 0, **{f"{metric_name}_sum": 0.0 for metric_name in metric_names}},
            )
            count = int(row["count"])
            merged_row["count"] += count
            for metric_name in metric_names:
                merged_row[f"{metric_name}_sum"] += float(row[metric_name]) * count
    for row in merged.values():
        count = int(row["count"])
        for metric_name in metric_names:
            row[metric_name] = row[f"{metric_name}_sum"] / count if count > 0 else 0.0

    print(f"\n{'=' * 80}")
    print(f"[LabeledEval][{model_tag}] Per-set Results")
    print(f"{'=' * 80}")

    psnr_sum = 0.0
    valid_count = 0
    for deg_name in all_deg_names:
        if deg_name not in merged:
            continue
        row = merged[deg_name]
        print(f"{deg_name:24s} PSNR: {row['psnr']:.4f} dB ({row['count']} imgs)")
        psnr_sum += row["psnr"]
        valid_count += 1

    avg_psnr = psnr_sum / valid_count if valid_count > 0 else 0.0
    print(f"{'-' * 80}")
    print(f"{'LabeledEval Avg':24s} PSNR: {avg_psnr:.4f} dB")
    print(f"{'=' * 80}\n")

    summary = {
        "model": model_tag,
        "metric_names": ["psnr"],
        "metric_labels": {"psnr": "PSNR"},
        "num_sets": valid_count,
        "per_set": merged,
        "average": {"psnr": avg_psnr},
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, f"{model_tag}_labeled_eval_metrics.json")
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[LabeledEval][{model_tag}] Saved summary: {result_path}")

    return summary


@torch.no_grad()
def evaluate_real_unpaired(
    model,
    dino_model,
    cgcd_model,
    real_eval_datasets,
    device,
    rank,
    world_size,
    use_ddp,
    amp_dtype=torch.float16,
    max_eval_side=0,
    output_dir=None,
    save_images=False,
    save_concat=False,
    reuse_saved_restored=False,
    num_workers=2,
    model_tag="teacher",
    clear_idx=0,
    cgcd_score_mode="clear",
    cgcd_extra_modes=None,
    cgcd_contrastive_pos_weight=1.0,
    cgcd_contrastive_neg_weight=1.0,
    cgcd_contrastive_tau=1.0,
    cgcd_contrastive_score_temp=1.0,
    cgcd_mahalanobis_temp=20.0,
    cgcd_mahalanobis_pca_top_k=32,
    use_vlm_vis=False,
    enable_cgcd_metrics=False,
):
    model.eval()
    if hasattr(model, "set_runtime_resize_fallback"):
        model.set_runtime_resize_fallback(True)
    dino_model.eval()
    cgcd_model.eval()

    metrics, available_metrics = build_iqa_metrics(device, rank, use_vlm_vis=use_vlm_vis)
    metric_names = [m for m, _ in available_metrics]
    metric_labels = {m: label for m, label in available_metrics}

    cgcd_metric_items = []
    cgcd_mode_by_key = {}
    if enable_cgcd_metrics:
        cgcd_modes = build_cgcd_modes(cgcd_score_mode, cgcd_extra_modes)
        for mode_idx, mode in enumerate(cgcd_modes):
            key = "cgcd" if mode_idx == 0 else f"cgcd_{mode}"
            cgcd_metric_items.append((key, mode))
            metric_names.append(key)
            metric_labels[key] = f"CGCD-{mode}"
        cgcd_mode_by_key = {k: m for k, m in cgcd_metric_items}

    all_set_names = [set_name for _, set_name in real_eval_datasets]

    if rank == 0:
        metric_title = ", ".join(metric_labels[m] for m in metric_names)
        print(f"\n{'=' * 80}")
        print(f"[RealEval][{model_tag}] Metrics: {metric_title}")
        print(f"{'=' * 80}")

    local_results = {}
    local_sample_records = {}
    for eval_dataset, set_name in real_eval_datasets:
        total_count = len(eval_dataset)
        if use_ddp:
            local_indices = list(range(rank, total_count, world_size))
        else:
            local_indices = list(range(total_count))

        eval_subset = Subset(eval_dataset, local_indices)
        loader = DataLoader(
            eval_subset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )

        metric_sums = {m: 0.0 for m in metric_names}
        sample_records = []
        reused_count = 0
        inferred_count = 0
        reuse_load_failed_warned = False
        iterator = tqdm(
            loader,
            desc=f"[GPU {rank}][RealEval] {set_name:<18} ({len(local_indices)}/{total_count})",
            leave=False,
            position=rank,
            ncols=100,
        )

        concat_save_dir = None
        restored_save_dir = None
        if save_images and output_dir is not None:
            if save_concat:
                concat_save_dir = os.path.join(output_dir, model_tag, set_name, "concat")
                restored_save_dir = os.path.join(output_dir, model_tag, set_name, "output")
                os.makedirs(concat_save_dir, exist_ok=True)
                os.makedirs(restored_save_dir, exist_ok=True)
            else:
                restored_save_dir = os.path.join(output_dir, model_tag, set_name, "output")
                os.makedirs(restored_save_dir, exist_ok=True)

        for local_pos, batch in enumerate(iterator):
            lq_image = batch[0].to(device, non_blocking=True)
            lq_path = batch[1][0]
            file_stem = os.path.splitext(os.path.basename(lq_path))[0]
            global_idx = local_indices[local_pos]

            restored_path = None
            if output_dir is not None:
                restored_path = os.path.join(output_dir, model_tag, set_name, "output", f"{file_stem}.png")

            use_saved_restored = (
                reuse_saved_restored
                and (not save_concat)
                and restored_path is not None
                and os.path.exists(restored_path)
            )
            if use_saved_restored:
                try:
                    restored = load_saved_restored_image(restored_path, device)
                    reused_count += 1
                except Exception as e:
                    if not reuse_load_failed_warned:
                        print(
                            f"[GPU {rank}][RealEval][Warning] Failed to read saved output image "
                            f"({restored_path}), fallback to inference: {e}"
                        )
                        reuse_load_failed_warned = True
                    use_saved_restored = False

            if not use_saved_restored:
                if max_eval_side and max_eval_side > 0:
                    _, _, h, w = lq_image.shape
                    longest = max(h, w)
                    if longest > max_eval_side:
                        scale = float(max_eval_side) / float(longest)
                        new_h = max(16, int((h * scale) // 16 * 16))
                        new_w = max(16, int((w * scale) // 16 * 16))
                        lq_image = F.interpolate(lq_image, size=(new_h, new_w), mode="bilinear", align_corners=False)

                os.environ["ONERESTORE_DEBUG_SAMPLE"] = os.path.basename(lq_path)
                try:
                    with autocast("cuda", dtype=amp_dtype):
                        dino_feature = dino_model(transform_resize(lq_image)).pooler_output
                        cgcd_embd, _ = cgcd_model(dino_feature)
                        restored = model(lq_image, cgcd_embd)
                finally:
                    os.environ.pop("ONERESTORE_DEBUG_SAMPLE", None)

                restored = torch.clamp(restored, 0, 1)
                inferred_count += 1

            sample_values = {}
            if cgcd_metric_items:
                cgcd_scores = compute_cgcd_scores_multi(
                    restored,
                    dino_model,
                    cgcd_model,
                    transform_resize,
                    clear_idx,
                    [mode for _, mode in cgcd_metric_items],
                    amp_dtype=amp_dtype,
                    contrastive_pos_weight=cgcd_contrastive_pos_weight,
                    contrastive_neg_weight=cgcd_contrastive_neg_weight,
                    contrastive_tau=cgcd_contrastive_tau,
                    contrastive_score_temp=cgcd_contrastive_score_temp,
                    mahalanobis_temp=cgcd_mahalanobis_temp,
                    mahalanobis_pca_top_k=cgcd_mahalanobis_pca_top_k,
                )
            else:
                cgcd_scores = {}

            for metric_name in metric_names:
                if metric_name in cgcd_mode_by_key:
                    cgcd_score = cgcd_scores[cgcd_mode_by_key[metric_name]]
                    score_value = float(cgcd_score.squeeze().item())
                else:
                    metric_input = preprocess_for_metric(metric_name, restored)
                    metric_value = metrics[metric_name](metric_input)
                    score_value = float(metric_value.squeeze().item())
                metric_sums[metric_name] += score_value
                sample_values[metric_name] = score_value

            file_name = os.path.basename(lq_path)
            line = f"{file_name} : " + ", ".join(f"{metric_labels[m]}: {sample_values[m]:.4f}" for m in metric_names)
            sample_record = {"set": set_name, "file": file_name}
            sample_record.update(sample_values)
            sample_records.append((int(global_idx), sample_record, line))

            if save_images and output_dir is not None:
                if save_concat:
                    concat_path = os.path.join(concat_save_dir, f"{file_stem}.png")
                    output_path = os.path.join(restored_save_dir, f"{file_stem}.png")
                    input_vis = torch.clamp(lq_image[0], 0, 1)
                    concat = torch.cat([input_vis, restored[0]], dim=2)
                    save_image(concat, concat_path)
                    save_image(restored[0], output_path)
                else:
                    restored_path_save = os.path.join(restored_save_dir, f"{file_stem}.png")
                    if not use_saved_restored or (not os.path.exists(restored_path_save)):
                        save_image(restored[0], restored_path_save)

        local_count = len(local_indices)
        set_result = {
            "count_total": total_count,
            "count_local": local_count,
            "metric_sums": metric_sums,
        }
        set_result["reused"] = reused_count
        set_result["inferred"] = inferred_count
        local_results[set_name] = set_result
        local_sample_records[set_name] = sample_records

        local_means = {m: (metric_sums[m] / local_count if local_count > 0 else 0.0) for m in metric_names}
        set_line = ", ".join(f"{metric_labels[m]}: {local_means[m]:.4f}" for m in metric_names)
        print(
            f"[GPU {rank}][RealEval][{model_tag}] {set_name:20s} - {set_line} "
            f"({local_count}/{total_count} imgs, reused={reused_count}, inferred={inferred_count})"
        )

    if use_ddp:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        gathered_sample_records = [None] * world_size
        dist.all_gather_object(gathered_sample_records, local_sample_records)
    else:
        gathered_results = [local_results]
        gathered_sample_records = [local_sample_records]

    if rank != 0:
        if hasattr(model, "set_runtime_resize_fallback"):
            model.set_runtime_resize_fallback(False)
        return None

    merged = {}
    merged_sample_records = {}
    merged_sample_lines = {}
    for set_name in all_set_names:
        count_total = 0
        count_local = 0
        reused_total = 0
        inferred_total = 0
        metric_sums_total = {m: 0.0 for m in metric_names}
        sample_items = []

        for result_dict in gathered_results:
            if not result_dict or set_name not in result_dict:
                continue
            row = result_dict[set_name]
            count_total = max(count_total, int(row.get("count_total", 0)))
            count_local += int(row.get("count_local", 0))
            reused_total += int(row.get("reused", 0))
            inferred_total += int(row.get("inferred", 0))
            row_metric_sums = row.get("metric_sums", {})
            for metric_name in metric_names:
                metric_sums_total[metric_name] += float(row_metric_sums.get(metric_name, 0.0))

        for record_dict in gathered_sample_records:
            if not record_dict or set_name not in record_dict:
                continue
            sample_items.extend(record_dict[set_name])

        if count_total <= 0:
            continue
        if count_local != count_total:
            print(
                f"[RealEval][{model_tag}][Warning] {set_name}: aggregated {count_local} "
                f"samples but expected {count_total}."
            )

        denom = count_local if count_local > 0 else 1
        set_result = {"count": count_total, "reused": reused_total, "inferred": inferred_total}
        for metric_name in metric_names:
            set_result[metric_name] = metric_sums_total[metric_name] / float(denom)
        merged[set_name] = set_result

        if sample_items:
            sample_items.sort(key=lambda x: x[0])
            merged_sample_records[set_name] = [record for _, record, _ in sample_items]
            merged_sample_lines[set_name] = [line for _, _, line in sample_items]
        else:
            merged_sample_records[set_name] = []
            merged_sample_lines[set_name] = []

    print(f"\n{'=' * 80}")
    print(f"[RealEval][{model_tag}] Per-set Results")
    print(f"{'=' * 80}")

    avg = {m: 0.0 for m in metric_names}
    valid_count = 0
    for set_name in all_set_names:
        if set_name not in merged:
            continue
        row = merged[set_name]
        line = ", ".join(f"{metric_labels[m]}: {row[m]:.4f}" for m in metric_names)
        print(
            f"{set_name:24s} {line} "
            f"({row['count']} imgs, reused={row.get('reused', 0)}, inferred={row.get('inferred', 0)})"
        )
        for metric_name in metric_names:
            avg[metric_name] += row[metric_name]
        valid_count += 1

    if valid_count > 0:
        for metric_name in metric_names:
            avg[metric_name] /= valid_count

    avg_line = ", ".join(f"{metric_labels[m]}: {avg[m]:.4f}" for m in metric_names)
    print(f"{'-' * 80}")
    print(f"{'RealEval Avg':24s} {avg_line}")
    print(f"{'=' * 80}\n")

    summary = {
        "model": model_tag,
        "metric_names": metric_names,
        "metric_labels": metric_labels,
        "num_sets": valid_count,
        "per_set": merged,
        "average": avg,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, f"{model_tag}_real_eval_metrics.json")
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[RealEval][{model_tag}] Saved summary: {result_path}")
        save_summary_tables(summary, result_path, prefix=f"[RealEval][{model_tag}]")

        txt_dir = os.path.join(output_dir, model_tag)
        os.makedirs(txt_dir, exist_ok=True)
        flat_sample_records = []
        for set_name in all_set_names:
            if set_name not in merged_sample_lines:
                continue
            safe_name = set_name.replace("/", "_")
            txt_path = os.path.join(txt_dir, f"{safe_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                if merged_sample_lines[set_name]:
                    f.write("\n".join(merged_sample_lines[set_name]) + "\n")
            print(f"[RealEval][{model_tag}] Saved per-sample metrics: {txt_path}")
            flat_sample_records.extend(merged_sample_records.get(set_name, []))

        save_per_sample_metric_csvs(
            flat_sample_records,
            metric_names,
            output_dir,
            model_tag,
            prefix=f"[RealEval][{model_tag}]",
        )

    if hasattr(model, "set_runtime_resize_fallback"):
        model.set_runtime_resize_fallback(False)
    return summary


def parse_set_names(set_names_str):
    return parse_csv_tokens(set_names_str)


def create_synthetic_eval_datasets(
    syn_eval_root,
    class_names,
    input_subdir="input",
    gt_subdir="gt",
    per_class=100,
    sample_seed=42,
    rank=0,
):
    datasets = []
    rng = np.random.default_rng(int(sample_seed))
    for class_name in class_names:
        class_dir = os.path.join(syn_eval_root, class_name)
        lq_dir = os.path.join(class_dir, input_subdir)
        hq_dir = os.path.join(class_dir, gt_subdir)
        if not os.path.isdir(lq_dir) or not os.path.isdir(hq_dir):
            if rank == 0:
                print(f"[SynEval][Warning] Skip class '{class_name}' (missing dirs): " f"input={lq_dir}, gt={hq_dir}")
            continue

        try:
            hq_img_paths, lq_img_paths = datasets_module._build_pairs(hq_dir, lq_dir, class_name)
        except Exception as e:
            if rank == 0:
                print(f"[SynEval][Warning] Skip class '{class_name}' (pairing failed): {e}")
            continue

        if per_class is not None and per_class > 0:
            n_total = len(hq_img_paths)
            n_pick = min(int(per_class), n_total)
            pick = np.sort(rng.choice(n_total, size=n_pick, replace=False))
            hq_img_paths = [hq_img_paths[i] for i in pick]
            lq_img_paths = [lq_img_paths[i] for i in pick]

        if len(hq_img_paths) == 0:
            if rank == 0:
                print(f"[SynEval][Warning] Skip class '{class_name}' (no paired images)")
            continue

        # Return both hq_path and lq_path so save file naming can follow input image.
        class SynPairEvalDataset(RefDegImage):
            def __getitem__(self, idx):
                hq_tensor, lq_tensor, hq_path = super().__getitem__(idx)
                lq_path = self.lq_paths[idx]
                return hq_tensor, lq_tensor, hq_path, lq_path

        dataset = SynPairEvalDataset(
            hq_img_paths, lq_img_paths, val=True, name=class_name, deg_name=class_name, deg_class=0
        )
        datasets.append((dataset, class_name))
        if rank == 0:
            print(f"[SynEval] Loaded class: {class_name} ({len(dataset)} images)")

    return datasets


@torch.no_grad()
def evaluate_synthetic_paired(
    model,
    dino_model,
    cgcd_model,
    syn_eval_datasets,
    device,
    rank,
    world_size,
    use_ddp,
    amp_dtype=torch.float16,
    output_dir=None,
    save_images=False,
    save_concat=False,
    num_workers=2,
    model_tag="teacher",
    clear_idx=0,
    cgcd_score_mode="contrastive",
    cgcd_extra_modes=None,
    cgcd_contrastive_pos_weight=1.0,
    cgcd_contrastive_neg_weight=1.0,
    cgcd_contrastive_tau=1.0,
    cgcd_contrastive_score_temp=1.0,
    cgcd_mahalanobis_temp=20.0,
    cgcd_mahalanobis_pca_top_k=32,
    enable_cgcd_metrics=False,
):
    if not syn_eval_datasets:
        if rank == 0:
            print(f"[SynEval][{model_tag}] No synthetic paired datasets. Skip.")
        return None

    model.eval()
    dino_model.eval()
    cgcd_model.eval()

    metric_names = ["psnr", "ssim", "l1"]
    metric_labels = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "l1": "L1",
    }
    cgcd_metric_items = []
    if enable_cgcd_metrics:
        cgcd_modes = build_cgcd_modes(cgcd_score_mode, cgcd_extra_modes)
        for mode_idx, mode in enumerate(cgcd_modes):
            key = "cgcd" if mode_idx == 0 else f"cgcd_{mode}"
            cgcd_metric_items.append((key, mode))
            metric_names.append(key)
            metric_labels[key] = f"CGCD-{mode}"

    all_deg_names = [deg_name for _, deg_name in syn_eval_datasets]

    if rank == 0:
        metric_title = ", ".join(metric_labels[m] for m in metric_names)
        print(f"\n{'=' * 80}")
        print(f"[SynEval][{model_tag}] Metrics: {metric_title}")
        print(f"{'=' * 80}")

    local_results = {}
    local_sample_records = {}
    for val_dataset, deg_name in syn_eval_datasets:
        local_indices = list(range(rank, len(val_dataset), world_size)) if use_ddp else list(range(len(val_dataset)))
        if not local_indices:
            continue
        local_dataset = Subset(val_dataset, local_indices)
        loader = DataLoader(
            local_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )

        metric_scores = {m: [] for m in metric_names}
        sample_records = []
        iterator = tqdm(loader, desc=f"[GPU {rank}][SynEval] {deg_name:<15}", leave=False, position=rank, ncols=100)

        concat_save_dir = None
        restored_save_dir = None
        if save_images and output_dir is not None:
            if save_concat:
                concat_save_dir = os.path.join(output_dir, model_tag, deg_name, "concat")
                restored_save_dir = os.path.join(output_dir, model_tag, deg_name, "output")
                os.makedirs(concat_save_dir, exist_ok=True)
                os.makedirs(restored_save_dir, exist_ok=True)
            else:
                restored_save_dir = os.path.join(output_dir, model_tag, deg_name, "output")
                os.makedirs(restored_save_dir, exist_ok=True)

        for batch in iterator:
            hq_image = batch[0].to(device, non_blocking=True)
            lq_image = batch[1].to(device, non_blocking=True)
            hq_path = batch[2][0]
            lq_path = batch[3][0]
            file_stem = os.path.splitext(os.path.basename(lq_path))[0]

            with autocast("cuda", dtype=amp_dtype):
                dino_feature = dino_model(transform_resize(lq_image)).pooler_output
                cgcd_embd, _ = cgcd_model(dino_feature)
                restored = model(lq_image, cgcd_embd)
            restored = torch.clamp(restored, 0, 1)

            restored_metric = restored.float()
            hq_metric = hq_image.float()
            psnr_score = pt_psnr(hq_metric, restored_metric)
            ssim_score = pt_ssim(hq_metric, restored_metric)
            l1_score = torch.mean(torch.abs(hq_metric - restored_metric), dim=(1, 2, 3), keepdim=True)
            if cgcd_metric_items:
                cgcd_scores = compute_cgcd_scores_multi(
                    restored,
                    dino_model,
                    cgcd_model,
                    transform_resize,
                    clear_idx,
                    [mode for _, mode in cgcd_metric_items],
                    amp_dtype=amp_dtype,
                    contrastive_pos_weight=cgcd_contrastive_pos_weight,
                    contrastive_neg_weight=cgcd_contrastive_neg_weight,
                    contrastive_tau=cgcd_contrastive_tau,
                    contrastive_score_temp=cgcd_contrastive_score_temp,
                    mahalanobis_temp=cgcd_mahalanobis_temp,
                    mahalanobis_pca_top_k=cgcd_mahalanobis_pca_top_k,
                )
            else:
                cgcd_scores = {}

            metric_scores["psnr"].append(psnr_score)
            metric_scores["ssim"].append(ssim_score)
            metric_scores["l1"].append(l1_score)
            for cgcd_key, cgcd_mode in cgcd_metric_items:
                metric_scores[cgcd_key].append(cgcd_scores[cgcd_mode])

            sample_values = {
                "psnr": float(psnr_score.squeeze().item()),
                "ssim": float(ssim_score.squeeze().item()),
                "l1": float(l1_score.squeeze().item()),
            }
            for cgcd_key, cgcd_mode in cgcd_metric_items:
                sample_values[cgcd_key] = float(cgcd_scores[cgcd_mode].squeeze().item())
            file_name = os.path.basename(lq_path)
            line = f"{file_name} : " + ", ".join(f"{metric_labels[m]}: {sample_values[m]:.4f}" for m in metric_names)
            sample_record = {"set": deg_name, "file": file_name}
            sample_record.update(sample_values)
            sample_records.append((sample_record, line))

            if save_images and output_dir is not None:
                if save_concat:
                    concat_path = os.path.join(concat_save_dir, f"{file_stem}.png")
                    output_path = os.path.join(restored_save_dir, f"{file_stem}.png")
                    concat = torch.cat([lq_image[0], restored[0], hq_image[0]], dim=2)
                    save_image(concat, concat_path)
                    save_image(restored[0], output_path)
                else:
                    restored_path_save = os.path.join(restored_save_dir, f"{file_stem}.png")
                    save_image(restored[0], restored_path_save)

        set_result = {"count": len(local_dataset)}
        for metric_name in metric_names:
            values = metric_scores[metric_name]
            set_result[metric_name] = torch.cat(values, dim=0).mean().item() if values else 0.0
        local_results[deg_name] = set_result
        local_sample_records[deg_name] = sample_records

        set_line = ", ".join(f"{metric_labels[m]}: {set_result[m]:.4f}" for m in metric_names)
        print(f"[GPU {rank}][SynEval][{model_tag}] {deg_name:20s} - {set_line} ({set_result['count']} imgs)")

    if use_ddp:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        gathered_sample_records = [None] * world_size
        dist.all_gather_object(gathered_sample_records, local_sample_records)
    else:
        gathered_results = [local_results]
        gathered_sample_records = [local_sample_records]

    if rank != 0:
        return None

    merged = {}
    for result_dict in gathered_results:
        if not result_dict:
            continue
        for deg_name, row in result_dict.items():
            merged_row = merged.setdefault(
                deg_name,
                {"count": 0, **{f"{metric_name}_sum": 0.0 for metric_name in metric_names}},
            )
            count = int(row["count"])
            merged_row["count"] += count
            for metric_name in metric_names:
                merged_row[f"{metric_name}_sum"] += float(row[metric_name]) * count
    for row in merged.values():
        count = int(row["count"])
        for metric_name in metric_names:
            row[metric_name] = row[f"{metric_name}_sum"] / count if count > 0 else 0.0
    merged_sample_records = {}
    merged_sample_lines = {}
    for record_dict in gathered_sample_records:
        if not record_dict:
            continue
        for deg_name, records in record_dict.items():
            merged_sample_records.setdefault(deg_name, []).extend(record for record, _ in records)
            merged_sample_lines.setdefault(deg_name, []).extend(line for _, line in records)

    print(f"\n{'=' * 80}")
    print(f"[SynEval][{model_tag}] Per-class Results")
    print(f"{'=' * 80}")

    avg = {m: 0.0 for m in metric_names}
    valid_count = 0
    for deg_name in all_deg_names:
        if deg_name not in merged:
            continue
        row = merged[deg_name]
        line = ", ".join(f"{metric_labels[m]}: {row[m]:.4f}" for m in metric_names)
        print(f"{deg_name:24s} {line} ({row['count']} imgs)")
        for metric_name in metric_names:
            avg[metric_name] += row[metric_name]
        valid_count += 1

    if valid_count > 0:
        for metric_name in metric_names:
            avg[metric_name] /= valid_count

    avg_line = ", ".join(f"{metric_labels[m]}: {avg[m]:.4f}" for m in metric_names)
    print(f"{'-' * 80}")
    print(f"{'SynEval Avg':24s} {avg_line}")
    print(f"{'=' * 80}\n")

    summary = {
        "model": model_tag,
        "metric_names": metric_names,
        "metric_labels": metric_labels,
        "num_sets": valid_count,
        "per_set": merged,
        "average": avg,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, f"{model_tag}_synthetic_eval_metrics.json")
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SynEval][{model_tag}] Saved summary: {result_path}")
        save_summary_tables(summary, result_path, prefix=f"[SynEval][{model_tag}]")

        txt_dir = os.path.join(output_dir, model_tag)
        os.makedirs(txt_dir, exist_ok=True)
        flat_sample_records = []
        for deg_name in all_deg_names:
            if deg_name not in merged_sample_lines:
                continue
            safe_name = deg_name.replace("/", "_")
            txt_path = os.path.join(txt_dir, f"{safe_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                if merged_sample_lines[deg_name]:
                    f.write("\n".join(merged_sample_lines[deg_name]) + "\n")
            print(f"[SynEval][{model_tag}] Saved per-sample metrics: {txt_path}")
            flat_sample_records.extend(merged_sample_records.get(deg_name, []))

        save_per_sample_metric_csvs(
            flat_sample_records,
            metric_names,
            output_dir,
            model_tag,
            prefix=f"[SynEval][{model_tag}]",
        )

    return summary


def run_eval(cfg, args):
    rank, world_size, local_rank, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    use_bf16 = bool(getattr(cfg.train, "use_bf16", False))
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    seed = int(getattr(cfg.train, "seed", 42))
    seed_everything(seed + rank)

    real_eval_root = args.real_eval_root or getattr(cfg.train, "real_eval_root", "")
    eval_set_names = parse_set_names(args.real_eval_sets)
    if eval_set_names is None:
        eval_set_names = list(getattr(cfg.train, "real_eval_sets", []))
    syn_eval_root = args.syn_eval_root
    syn_eval_class_names = parse_set_names(args.syn_eval_classes) or []
    syn_eval_per_class = int(args.syn_eval_per_class)
    syn_eval_seed = int(args.syn_eval_seed)
    syn_input_subdir = args.syn_input_subdir
    syn_gt_subdir = args.syn_gt_subdir

    max_eval_side = args.max_eval_side
    if max_eval_side is None:
        max_eval_side = int(getattr(cfg.train, "real_eval_max_side", 0) or 0)

    enable_cgcd_metrics = bool(getattr(args, "enable_cgcd_metrics", False))
    cgcd_score_mode = None
    cgcd_extra_modes = []
    cgcd_modes = []
    cgcd_contrastive_pos_weight = float(get_cfg_cgcd_value(cfg, "cgcd_contrastive_pos_weight", 1.0))
    cgcd_contrastive_neg_weight = float(get_cfg_cgcd_value(cfg, "cgcd_contrastive_neg_weight", 1.0))
    cgcd_contrastive_tau = float(get_cfg_cgcd_value(cfg, "cgcd_contrastive_tau", 1.0))
    cgcd_contrastive_score_temp = float(get_cfg_cgcd_value(cfg, "cgcd_contrastive_score_temp", 1.0))
    cgcd_mahalanobis_temp = float(get_cfg_cgcd_value(cfg, "cgcd_mahalanobis_temp", 20.0))
    cgcd_mahalanobis_pca_top_k = get_cfg_cgcd_value(cfg, "cgcd_mahalanobis_pca_top_k", 32)

    if enable_cgcd_metrics:
        cgcd_score_mode = args.cgcd_score_mode or str(get_cfg_cgcd_value(cfg, "cgcd_score_mode", "contrastive"))
        cgcd_extra_modes = parse_csv_tokens(args.cgcd_extra_modes)
        if cgcd_extra_modes is None:
            cgcd_extra_modes = parse_csv_tokens(get_cfg_cgcd_value(cfg, "cgcd_eval_extra_modes", "")) or []
        cgcd_modes = build_cgcd_modes(cgcd_score_mode, cgcd_extra_modes)
        invalid_modes = [m for m in cgcd_modes if m not in CGCD_ALL_MODES]
        if invalid_modes:
            valid_text = ", ".join(CGCD_ALL_MODES)
            raise ValueError(f"Unsupported CGCD mode(s): {invalid_modes}. Available: {valid_text}")
        if args.cgcd_contrastive_pos_weight is not None:
            cgcd_contrastive_pos_weight = args.cgcd_contrastive_pos_weight
        if args.cgcd_contrastive_neg_weight is not None:
            cgcd_contrastive_neg_weight = args.cgcd_contrastive_neg_weight
        if args.cgcd_contrastive_tau is not None:
            cgcd_contrastive_tau = args.cgcd_contrastive_tau
        if args.cgcd_contrastive_score_temp is not None:
            cgcd_contrastive_score_temp = args.cgcd_contrastive_score_temp
        if args.cgcd_mahalanobis_temp is not None:
            cgcd_mahalanobis_temp = args.cgcd_mahalanobis_temp
        if args.cgcd_mahalanobis_pca_top_k is not None:
            cgcd_mahalanobis_pca_top_k = args.cgcd_mahalanobis_pca_top_k

    if cgcd_mahalanobis_pca_top_k is not None:
        cgcd_mahalanobis_pca_top_k = int(cgcd_mahalanobis_pca_top_k)
    clear_idx = resolve_clear_idx(cfg, rank)

    if rank == 0:
        print(f"\n{'=' * 80}")
        print("Evaluation Mode - OneRestore Real IQA")
        print(f"{'=' * 80}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Eval model: {args.eval_model}")
        print(f"World size: {world_size}")
        print(f"AMP dtype: {'BF16' if use_bf16 else 'FP16'}")
        if hasattr(cfg, "incremental"):
            print(
                "incremental: "
                f"stage={cfg.incremental.stage}, "
                f"base_class_num={cfg.incremental.base_class_num}, "
                f"inc_class_num={cfg.incremental.inc_class_num}"
            )
        print(f"real_eval_root: {real_eval_root}")
        print(f"real_eval_sets: {eval_set_names}")
        print(f"syn_eval_only: {args.syn_eval_only}")
        if args.syn_eval_only:
            print(f"syn_eval_root: {syn_eval_root}")
            print(f"syn_eval_classes: {syn_eval_class_names}")
            print(f"syn_eval_per_class: {syn_eval_per_class}")
            print(f"syn_eval_seed: {syn_eval_seed}")
            print(f"syn_input_subdir: {syn_input_subdir}")
            print(f"syn_gt_subdir: {syn_gt_subdir}")
        print(f"max_eval_side: {max_eval_side}")
        print(f"enable_cgcd_metrics: {enable_cgcd_metrics}")
        if enable_cgcd_metrics:
            print(f"cgcd_score_mode: {cgcd_score_mode}")
            print(f"cgcd_modes(all): {cgcd_modes}")
            if any(m == "contrastive" for m in cgcd_modes):
                print(
                    "cgcd contrastive params: "
                    f"w_pos={cgcd_contrastive_pos_weight}, "
                    f"w_neg={cgcd_contrastive_neg_weight}, "
                    f"tau={cgcd_contrastive_tau}, "
                    f"score_temp={cgcd_contrastive_score_temp}"
                )
            if any(m in ("mahalanobis_margin", "mahalanobis_pca") for m in cgcd_modes):
                print(f"cgcd mahalanobis_temp: {cgcd_mahalanobis_temp}")
            if any(m == "mahalanobis_pca" for m in cgcd_modes):
                print(f"cgcd mahalanobis_pca_top_k: {cgcd_mahalanobis_pca_top_k}")
        print(f"clear_idx: {clear_idx}")
        print(f"reuse_saved_restored: {args.reuse_saved_restored and (not args.save_concat)}")
        if args.reuse_saved_restored and args.save_concat:
            print("[Warning] --reuse_saved_restored is ignored when --save_concat is enabled.")
        if args.output:
            print(f"Output: {args.output}")
        print(f"{'=' * 80}\n")

    labeled_val_datasets = []
    if (not args.labeled_eval) and (not args.syn_eval_only):
        labeled_val_datasets = create_labeled_val_datasets_from_cfg(cfg, rank)
        if rank == 0 and len(labeled_val_datasets) == 0:
            print("[LabeledEval] No paired labeled validation sets found. Skip labeled evaluation.")

    real_eval_datasets = []
    syn_eval_datasets = []
    if args.syn_eval_only:
        if not syn_eval_root:
            raise ValueError("syn_eval_root is empty. Pass --syn_eval_root when --syn_eval_only is enabled.")
        if not syn_eval_class_names:
            raise ValueError("syn_eval_classes is empty. Pass --syn_eval_classes when --syn_eval_only is enabled.")
        syn_eval_datasets = create_synthetic_eval_datasets(
            syn_eval_root=syn_eval_root,
            class_names=syn_eval_class_names,
            input_subdir=syn_input_subdir,
            gt_subdir=syn_gt_subdir,
            per_class=syn_eval_per_class,
            sample_seed=syn_eval_seed,
            rank=rank,
        )
        if len(syn_eval_datasets) == 0:
            raise ValueError("No synthetic eval datasets were created. Check root/classes/subdir names.")
    else:
        if not real_eval_root:
            raise ValueError("real_eval_root is empty. Set train.real_eval_root in config or pass --real_eval_root.")
        if not eval_set_names:
            raise ValueError("No real eval sets found. Set train.real_eval_sets or pass --real_eval_sets.")

        real_eval_datasets = create_real_eval_datasets(real_eval_root, eval_set_names)
        if len(real_eval_datasets) == 0:
            raise ValueError("No real eval datasets were created. Check paths and set names.")

    if args.save_image and not args.output:
        raise ValueError("--save_image requires --output.")
    if args.reuse_saved_restored and not args.output:
        raise ValueError("--reuse_saved_restored requires --output.")

    student_model, teacher_model, dino_model, cgcd_model, epoch = load_models(cfg, args.checkpoint, device, rank)
    if rank == 0:
        print(f"Checkpoint epoch: {epoch}")

    real_summaries = {}
    syn_summaries = {}
    labeled_summaries = {}
    if args.eval_model in ("teacher", "both"):
        if not args.syn_eval_only and (not args.labeled_eval) and len(labeled_val_datasets) > 0:
            labeled_summary = evaluate_labeled_paired(
                teacher_model,
                dino_model,
                cgcd_model,
                labeled_val_datasets,
                device,
                rank,
                world_size,
                use_ddp,
                amp_dtype=amp_dtype,
                num_workers=args.num_workers,
                output_dir=args.output,
                model_tag="teacher",
            )
            if rank == 0 and labeled_summary is not None:
                labeled_summaries["teacher"] = labeled_summary

        if args.syn_eval_only:
            summary = evaluate_synthetic_paired(
                teacher_model,
                dino_model,
                cgcd_model,
                syn_eval_datasets,
                device,
                rank,
                world_size,
                use_ddp,
                amp_dtype=amp_dtype,
                output_dir=args.output,
                save_images=args.save_image,
                save_concat=args.save_concat,
                num_workers=args.num_workers,
                model_tag="teacher",
                clear_idx=clear_idx,
                cgcd_score_mode=cgcd_score_mode,
                cgcd_extra_modes=cgcd_extra_modes,
                cgcd_contrastive_pos_weight=cgcd_contrastive_pos_weight,
                cgcd_contrastive_neg_weight=cgcd_contrastive_neg_weight,
                cgcd_contrastive_tau=cgcd_contrastive_tau,
                cgcd_contrastive_score_temp=cgcd_contrastive_score_temp,
                cgcd_mahalanobis_temp=cgcd_mahalanobis_temp,
                cgcd_mahalanobis_pca_top_k=cgcd_mahalanobis_pca_top_k,
                enable_cgcd_metrics=enable_cgcd_metrics,
            )
            if rank == 0 and summary is not None:
                syn_summaries["teacher"] = summary
        else:
            summary = evaluate_real_unpaired(
                teacher_model,
                dino_model,
                cgcd_model,
                real_eval_datasets,
                device,
                rank,
                world_size,
                use_ddp,
                amp_dtype=amp_dtype,
                max_eval_side=max_eval_side,
                output_dir=args.output,
                save_images=args.save_image,
                save_concat=args.save_concat,
                reuse_saved_restored=args.reuse_saved_restored,
                num_workers=args.num_workers,
                model_tag="teacher",
                clear_idx=clear_idx,
                cgcd_score_mode=cgcd_score_mode,
                cgcd_extra_modes=cgcd_extra_modes,
                cgcd_contrastive_pos_weight=cgcd_contrastive_pos_weight,
                cgcd_contrastive_neg_weight=cgcd_contrastive_neg_weight,
                cgcd_contrastive_tau=cgcd_contrastive_tau,
                cgcd_contrastive_score_temp=cgcd_contrastive_score_temp,
                cgcd_mahalanobis_temp=cgcd_mahalanobis_temp,
                cgcd_mahalanobis_pca_top_k=cgcd_mahalanobis_pca_top_k,
                use_vlm_vis=getattr(args, "use_vlm_vis", False),
                enable_cgcd_metrics=enable_cgcd_metrics,
            )
            if rank == 0 and summary is not None:
                real_summaries["teacher"] = summary

    if args.eval_model in ("student", "both"):
        if not args.syn_eval_only and (not args.labeled_eval) and len(labeled_val_datasets) > 0:
            labeled_summary = evaluate_labeled_paired(
                student_model,
                dino_model,
                cgcd_model,
                labeled_val_datasets,
                device,
                rank,
                world_size,
                use_ddp,
                amp_dtype=amp_dtype,
                num_workers=args.num_workers,
                output_dir=args.output,
                model_tag="student",
            )
            if rank == 0 and labeled_summary is not None:
                labeled_summaries["student"] = labeled_summary

        if args.syn_eval_only:
            summary = evaluate_synthetic_paired(
                student_model,
                dino_model,
                cgcd_model,
                syn_eval_datasets,
                device,
                rank,
                world_size,
                use_ddp,
                amp_dtype=amp_dtype,
                output_dir=args.output,
                save_images=args.save_image,
                save_concat=args.save_concat,
                num_workers=args.num_workers,
                model_tag="student",
                clear_idx=clear_idx,
                cgcd_score_mode=cgcd_score_mode,
                cgcd_extra_modes=cgcd_extra_modes,
                cgcd_contrastive_pos_weight=cgcd_contrastive_pos_weight,
                cgcd_contrastive_neg_weight=cgcd_contrastive_neg_weight,
                cgcd_contrastive_tau=cgcd_contrastive_tau,
                cgcd_contrastive_score_temp=cgcd_contrastive_score_temp,
                cgcd_mahalanobis_temp=cgcd_mahalanobis_temp,
                cgcd_mahalanobis_pca_top_k=cgcd_mahalanobis_pca_top_k,
                enable_cgcd_metrics=enable_cgcd_metrics,
            )
            if rank == 0 and summary is not None:
                syn_summaries["student"] = summary
        else:
            summary = evaluate_real_unpaired(
                student_model,
                dino_model,
                cgcd_model,
                real_eval_datasets,
                device,
                rank,
                world_size,
                use_ddp,
                amp_dtype=amp_dtype,
                max_eval_side=max_eval_side,
                output_dir=args.output,
                save_images=args.save_image,
                save_concat=args.save_concat,
                reuse_saved_restored=args.reuse_saved_restored,
                num_workers=args.num_workers,
                model_tag="student",
                clear_idx=clear_idx,
                cgcd_score_mode=cgcd_score_mode,
                cgcd_extra_modes=cgcd_extra_modes,
                cgcd_contrastive_pos_weight=cgcd_contrastive_pos_weight,
                cgcd_contrastive_neg_weight=cgcd_contrastive_neg_weight,
                cgcd_contrastive_tau=cgcd_contrastive_tau,
                cgcd_contrastive_score_temp=cgcd_contrastive_score_temp,
                cgcd_mahalanobis_temp=cgcd_mahalanobis_temp,
                cgcd_mahalanobis_pca_top_k=cgcd_mahalanobis_pca_top_k,
                use_vlm_vis=getattr(args, "use_vlm_vis", False),
                enable_cgcd_metrics=enable_cgcd_metrics,
            )
            if rank == 0 and summary is not None:
                real_summaries["student"] = summary

    if rank == 0 and (real_summaries or syn_summaries or labeled_summaries):
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}")
        for model_tag in ("teacher", "student"):
            parts = []
            if model_tag in labeled_summaries:
                avg_psnr = labeled_summaries[model_tag]["average"]["psnr"]
                parts.append(f"Labeled PSNR: {avg_psnr:.4f} dB")
            if model_tag in real_summaries:
                avg = real_summaries[model_tag]["average"]
                labels = real_summaries[model_tag]["metric_labels"]
                metric_line = ", ".join(
                    f"{labels[m]}: {avg[m]:.4f}" for m in real_summaries[model_tag]["metric_names"]
                )
                parts.append(f"Real {metric_line}")
            if model_tag in syn_summaries:
                avg = syn_summaries[model_tag]["average"]
                labels = syn_summaries[model_tag]["metric_labels"]
                metric_line = ", ".join(f"{labels[m]}: {avg[m]:.4f}" for m in syn_summaries[model_tag]["metric_names"])
                parts.append(f"Synthetic {metric_line}")
            if not parts:
                continue
            print(f"{model_tag:10s} " + " | ".join(parts))
        print(f"{'=' * 80}\n")

    if use_ddp:
        dist.destroy_process_group()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Evaluate OneRestore on real unpaired datasets with IQA metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--eval_model",
        type=str,
        default="teacher",
        choices=["teacher", "student", "both"],
        help="Which model to evaluate",
    )
    parser.add_argument("--output", type=str, default=None, help="Directory to save JSON summaries and output images")
    parser.add_argument("--save_image", action="store_true", help="Save output images")
    parser.add_argument("--save_concat", action="store_true", help="Save input|output concat images")
    parser.add_argument(
        "--use_vlm_vis", action="store_true", help="Enable VLM-Vis evaluation locally using LLaVA (requires 8GB+ VRAM)"
    )
    parser.add_argument(
        "--reuse_saved_restored",
        action="store_true",
        help="For non-concat mode, if output image already exists under --output, skip model forward and evaluate from that image.",
    )
    parser.add_argument("--real_eval_root", type=str, default=None, help="Override train.real_eval_root")
    parser.add_argument(
        "--real_eval_sets",
        type=str,
        default=None,
        help="Comma-separated set names (override train.real_eval_sets)",
    )
    parser.add_argument("--max_eval_side", type=int, default=0, help="Max side for evaluation images (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers per rank")
    parser.add_argument("--labeled_eval", action="store_false", help="Skip paired labeled validation evaluation")
    parser.add_argument(
        "--syn_eval_only",
        action="store_true",
        help="Evaluate synthetic paired sets only (skip real eval).",
    )
    parser.add_argument(
        "--syn_eval_root",
        type=str,
        default=None,
        help="Synthetic paired eval root: <root>/<class>/{input,gt}",
    )
    parser.add_argument(
        "--syn_eval_classes",
        type=str,
        default="OTS,Outdoor_Rain,RainDrop,Snow100k,SPA",
        help="Comma-separated synthetic classes to evaluate",
    )
    parser.add_argument(
        "--syn_eval_per_class",
        type=int,
        default=100,
        help="Max number of paired samples per synthetic class",
    )
    parser.add_argument(
        "--syn_eval_seed",
        type=int,
        default=42,
        help="Random seed for synthetic class-wise sampling",
    )
    parser.add_argument(
        "--syn_input_subdir",
        type=str,
        default="input",
        help="Input subdir name under each synthetic class folder",
    )
    parser.add_argument(
        "--syn_gt_subdir",
        type=str,
        default="gt",
        help="GT subdir name under each synthetic class folder",
    )
    parser.add_argument(
        "--enable_cgcd_metrics",
        action="store_true",
        help="Enable optional CGCD metric computation during eval (disabled by default)",
    )
    parser.add_argument(
        "--cgcd_score_mode",
        type=str,
        default=None,
        choices=list(CGCD_ALL_MODES),
        help="CGCD score mode for eval when --enable_cgcd_metrics is set",
    )
    parser.add_argument(
        "--cgcd_extra_modes",
        type=str,
        default=None,
        help=("Comma-separated extra CGCD modes to evaluate together. " f"Available: {','.join(CGCD_ALL_MODES)}"),
    )
    parser.add_argument("--cgcd_contrastive_pos_weight", type=float, default=None, help="CGCD contrastive pos weight")
    parser.add_argument("--cgcd_contrastive_neg_weight", type=float, default=None, help="CGCD contrastive neg weight")
    parser.add_argument("--cgcd_contrastive_tau", type=float, default=None, help="CGCD contrastive tau")
    parser.add_argument(
        "--cgcd_contrastive_score_temp",
        type=float,
        default=None,
        help="CGCD contrastive final score temperature",
    )
    parser.add_argument(
        "--cgcd_mahalanobis_temp",
        type=float,
        default=20,
        help="CGCD Mahalanobis score temperature (for mahalanobis_* modes)",
    )
    parser.add_argument(
        "--cgcd_mahalanobis_pca_top_k",
        type=int,
        default=32,
        help="CGCD mahalanobis_pca에서 사용할 top-k eigen components (None이면 전체 사용)",
    )
    parser.add_argument("--stage", type=int, default=None, help="Override incremental.stage in config")
    parser.add_argument("--base_class_num", type=int, default=None, help="Override incremental.base_class_num")
    parser.add_argument("--inc_class_num", type=int, default=None, help="Override incremental.inc_class_num")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path("./results") / Path(args.config).stem

    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    if args.save_concat:
        args.save_image = True

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.stage is not None:
        config["incremental"]["stage"] = args.stage
    if args.base_class_num is not None:
        config["incremental"]["base_class_num"] = args.base_class_num
    if args.inc_class_num is not None:
        config["incremental"]["inc_class_num"] = args.inc_class_num

    cfg = dict2namespace(config)
    run_eval(cfg, args)


if __name__ == "__main__":
    main()
