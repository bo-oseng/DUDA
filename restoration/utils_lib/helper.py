"""
Training helper functions for incremental learning with Mean Teacher.
Shared across different training scripts.
"""

import os
import random
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop
from datetime import timedelta
from tqdm import tqdm
from torchvision.utils import make_grid

from utils_lib.score import get_cgcd_anchor_score, is_cgcd_anchor_mode

_FGRESQ_METRIC_CACHE = {}


def _load_fgresq_module():
    """Load FGResQ module from local symlink/repo path."""
    helper_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(helper_dir)
    fgresq_root = os.path.join(project_root, "FGResQ")
    module_path = os.path.join(fgresq_root, "model", "FGResQ.py")

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"FGResQ module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("fgresq_external_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load FGResQ module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fgresq_root


class FGResQBatchMetric:
    """
    Batch metric wrapper for FGResQ.
    Expects images as [B, C, H, W] tensors in [0, 1] range.
    """

    def __init__(self, device):
        module, fgresq_root = _load_fgresq_module()
        model_path = os.path.join(fgresq_root, "weights", "FGResQ.pth")
        degradation_path = "FGResQ/weights/Degradation.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FGResQ weight not found: {model_path}")
        if not os.path.exists(degradation_path):
            raise FileNotFoundError(f"FGResQ degradation weight not found: {degradation_path}")

        original_hf_hub_download = module.hf_hub_download
        original_safe_check_import_utils = None
        original_safe_check_modeling_utils = None

        def _patched_hf_hub_download(repo_id, filename, *args, **kwargs):
            if filename == "weights/Degradation.pth":
                return degradation_path
            return original_hf_hub_download(repo_id=repo_id, filename=filename, *args, **kwargs)

        # transformers>=4.56 blocks torch<2.6 for torch.load even with weights_only=True.
        # FGResQ depends on CLIPVisionModel.from_pretrained() that can load .bin weights.
        def _unsafe_but_compatible_torch_load_check(*args, **kwargs):
            return None

        try:
            import transformers.modeling_utils as modeling_utils
            import transformers.utils.import_utils as import_utils

            if hasattr(import_utils, "check_torch_load_is_safe"):
                original_safe_check_import_utils = import_utils.check_torch_load_is_safe
                import_utils.check_torch_load_is_safe = _unsafe_but_compatible_torch_load_check
            if hasattr(modeling_utils, "check_torch_load_is_safe"):
                original_safe_check_modeling_utils = modeling_utils.check_torch_load_is_safe
                modeling_utils.check_torch_load_is_safe = _unsafe_but_compatible_torch_load_check
        except Exception:
            # If patching fails, keep original behavior and surface the downstream exception.
            pass

        module.hf_hub_download = _patched_hf_hub_download
        try:
            self.engine = module.FGResQ(model_path=model_path, device=device)
        finally:
            module.hf_hub_download = original_hf_hub_download
            try:
                import transformers.modeling_utils as modeling_utils
                import transformers.utils.import_utils as import_utils

                if original_safe_check_import_utils is not None:
                    import_utils.check_torch_load_is_safe = original_safe_check_import_utils
                if original_safe_check_modeling_utils is not None:
                    modeling_utils.check_torch_load_is_safe = original_safe_check_modeling_utils
            except Exception:
                pass

        self.device = torch.device(device)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, image_batch):
        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)

        image_batch = image_batch.clamp(0, 1)
        image_batch = F.interpolate(image_batch, size=(256, 256), mode="bilinear", align_corners=False)
        image_batch = center_crop(image_batch, [224, 224])

        model_dtype = next(self.engine.model.parameters()).dtype
        image_batch = image_batch.to(device=self.device, dtype=model_dtype)
        image_batch = (image_batch - self.mean.to(dtype=model_dtype)) / self.std.to(dtype=model_dtype)

        quality, _, _ = self.engine.model(image_batch)
        return quality.view(-1)


def create_fgresq_metric(device):
    """Create (or reuse cached) FGResQ batch metric on the target device."""
    cache_key = str(device)
    if cache_key not in _FGRESQ_METRIC_CACHE:
        _FGRESQ_METRIC_CACHE[cache_key] = FGResQBatchMetric(device=device)
    return _FGRESQ_METRIC_CACHE[cache_key]


def save_image_tensor(tensor, path):
    """Save a tensor image to file using torchvision."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor.clamp(0, 1), path)


def setup_ddp():
    """Setup Distributed Data Parallel training."""
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


@torch.no_grad()
def update_teacher_ema(teacher_model, student_model, alpha=0.996, global_step=0):
    """
    Update teacher model using Exponential Moving Average of student weights.
    Only updates LoRA parameters (cached in model.lora_params).
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    student = student_model.module if hasattr(student_model, "module") else student_model

    for t_param, s_param in zip(teacher_model.lora_params, student.lora_params):
        t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)


@torch.no_grad()
def get_musiq_score(iqa_metric, image_list):
    """Compute IQA scores for a batch of images."""
    score = torch.zeros(image_list.shape[0], device=image_list.device)
    for idx in range(image_list.shape[0]):
        score[idx] = iqa_metric(image_list[idx : idx + 1])
    return score


@torch.no_grad()
def get_fgresq_score(iqa_metric, image_list):
    """Compute FGResQ scores for a batch of images."""
    return iqa_metric(image_list).view(-1).float()


def normalize_mode_name(mode):
    token = str(mode).strip().lower().replace("-", "").replace("_", "")
    if token in ("musiq",):
        return "musiq"
    if token in ("fgresq",):
        return "fgresq"
    if token in ("clipiqa", "clipiqa6", "clip"):
        return "clipiqa"
    if token in ("cgcd",):
        return "cgcd"
    if token in ("alternate",):
        return "alternate"
    raise ValueError(f"Unsupported round metric mode: {mode}")


def parse_round_metric_schedule(schedule_str, fallback_mode):
    if schedule_str is None:
        return [normalize_mode_name(fallback_mode)]
    if isinstance(schedule_str, (list, tuple)):
        tokens = [str(x).strip() for x in schedule_str if str(x).strip()]
    else:
        tokens = [x.strip() for x in str(schedule_str).split(",") if x.strip()]
    if not tokens:
        return [normalize_mode_name(fallback_mode)]
    return [normalize_mode_name(x) for x in tokens]


def select_round_mode(global_step, round_steps, schedule, policy):
    if round_steps <= 0:
        return 0, schedule[0]
    round_idx = int(global_step // round_steps)
    if policy == "cycle":
        idx = round_idx % len(schedule)
    else:
        idx = min(round_idx, len(schedule) - 1)
    return round_idx, schedule[idx]


def sigmoid_rampup(current, rampup_length):
    if rampup_length <= 0:
        return 1.0
    current = min(max(float(current), 0.0), float(rampup_length))
    phase = 1.0 - current / float(rampup_length)
    return float(np.exp(-5.0 * phase * phase))


def consistency_weight_by_step(global_step, consistency_weight, rampup_steps):
    return float(consistency_weight) * sigmoid_rampup(global_step, rampup_steps)


def _resolve_reference_exists_mask(pseudo_names, device):
    if pseudo_names is None:
        return None
    return torch.tensor([os.path.exists(path) for path in pseudo_names], device=device, dtype=torch.bool)


def _summarize_pseudo_update_delta(
    score_teacher,
    score_student,
    score_reference,
    margin,
    reference_exists_mask=None,
):
    score_teacher = score_teacher.detach().float().view(-1)
    score_student = score_student.detach().float().view(-1)
    score_reference = score_reference.detach().float().view(-1)
    if reference_exists_mask is not None:
        reference_exists_mask = reference_exists_mask.detach().bool().view(-1)
        if reference_exists_mask.numel() != score_reference.numel():
            raise ValueError(
                f"reference_exists_mask size mismatch: {reference_exists_mask.numel()} vs {score_reference.numel()}"
            )
        effective_reference = torch.where(reference_exists_mask, score_reference, score_student)
    else:
        effective_reference = score_reference
    baseline = torch.maximum(score_student, effective_reference)
    delta_raw = score_teacher - baseline
    update_mask = delta_raw > float(margin)
    updated_delta = delta_raw[update_mask]

    def _scalar_stats(values):
        if values.numel() == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "sum": 0.0,
                "sq_sum": 0.0,
            }
        vals = values.detach().float().cpu()
        quantiles = torch.quantile(vals, torch.tensor([0.5, 0.75, 0.9], dtype=vals.dtype))
        return {
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
            "min": float(vals.min().item()),
            "max": float(vals.max().item()),
            "p50": float(quantiles[0].item()),
            "p75": float(quantiles[1].item()),
            "p90": float(quantiles[2].item()),
            "sum": float(vals.sum().item()),
            "sq_sum": float((vals * vals).sum().item()),
        }

    raw_stats = _scalar_stats(delta_raw)
    updated_stats = _scalar_stats(updated_delta)
    count = int(delta_raw.numel())
    update_count = int(updated_delta.numel())
    positive_count = int((delta_raw > 0).sum().item())
    return {
        "margin": float(margin),
        "count": count,
        "update_count": update_count,
        "positive_count": positive_count,
        "update_ratio": float(update_count / count) if count > 0 else 0.0,
        "positive_ratio": float(positive_count / count) if count > 0 else 0.0,
        "delta_raw_mean": raw_stats["mean"],
        "delta_raw_std": raw_stats["std"],
        "delta_raw_min": raw_stats["min"],
        "delta_raw_max": raw_stats["max"],
        "delta_raw_p50": raw_stats["p50"],
        "delta_raw_p75": raw_stats["p75"],
        "delta_raw_p90": raw_stats["p90"],
        "delta_raw_sum": raw_stats["sum"],
        "delta_raw_sq_sum": raw_stats["sq_sum"],
        "delta_update_mean": updated_stats["mean"],
        "delta_update_std": updated_stats["std"],
        "delta_update_min": updated_stats["min"],
        "delta_update_max": updated_stats["max"],
        "delta_update_p50": updated_stats["p50"],
        "delta_update_p75": updated_stats["p75"],
        "delta_update_p90": updated_stats["p90"],
        "delta_update_sum": updated_stats["sum"],
        "delta_update_sq_sum": updated_stats["sq_sum"],
        "score_teacher_mean": float(score_teacher.mean().item()) if count > 0 else 0.0,
        "score_student_mean": float(score_student.mean().item()) if count > 0 else 0.0,
        "score_reference_mean": float(score_reference.mean().item()) if count > 0 else 0.0,
    }


@torch.no_grad()
def get_reliable_clip_iqa(
    iqa_metric, teacher_predict, student_predict, pseudo_list, pseudo_names, rank, update_margin=0.0, return_stats=False
):
    # CLIP-IQA expects inputs strictly within [0, 1].
    teacher_predict = torch.clamp(teacher_predict, 0.0, 1.0)
    student_predict = torch.clamp(student_predict, 0.0, 1.0)
    pseudo_list = torch.clamp(pseudo_list, 0.0, 1.0)

    score_teacher = iqa_metric(teacher_predict).view(-1).float()
    score_student = iqa_metric(student_predict).view(-1).float()
    score_reference = iqa_metric(pseudo_list).view(-1).float()
    reference_exists_mask = _resolve_reference_exists_mask(pseudo_names, teacher_predict.device)
    effective_reference_score = torch.where(reference_exists_mask, score_reference, score_student)

    final_pseudo_labels = pseudo_list.clone()
    if reference_exists_mask.numel() > 0 and torch.any(~reference_exists_mask):
        final_pseudo_labels[~reference_exists_mask] = student_predict[~reference_exists_mask]
    updated_cnt = 0
    margin = float(update_margin)
    stats = _summarize_pseudo_update_delta(
        score_teacher,
        score_student,
        score_reference,
        margin,
        reference_exists_mask=reference_exists_mask,
    )
    for idx in range(teacher_predict.shape[0]):
        threshold = torch.maximum(score_student[idx], effective_reference_score[idx]) + margin
        if score_teacher[idx] > threshold:
            final_pseudo_labels[idx] = teacher_predict[idx]
            updated_cnt += 1
            save_image_tensor(teacher_predict[idx], pseudo_names[idx])
    if return_stats:
        return final_pseudo_labels, updated_cnt, "CLIP-IQA", stats
    return final_pseudo_labels, updated_cnt, "CLIP-IQA"


def _preprocess_feature_for_mahalanobis(feature, cgcd_model):
    feat_out = feature
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


def _build_inv_sqrt_covs_full(covs):
    covs = covs.float()
    evals, evecs = torch.linalg.eigh(covs)
    evals = evals.clamp(min=1e-8)
    inv_sqrt_evals = 1.0 / torch.sqrt(evals)
    return torch.bmm(evecs, torch.bmm(torch.diag_embed(inv_sqrt_evals), evecs.mT))


def compute_cgcd_mahalanobis_margin_score(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    amp_dtype=None,
    contrastive_tau=1.0,
    mahalanobis_temp=20.0,
):
    """Differentiable mahalanobis_margin score in [0, 1]."""
    use_amp = amp_dtype is not None and image_list.is_cuda
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        dino_feat = dino_model(dino_transform(image_list)).pooler_output

    processed_feat = _preprocess_feature_for_mahalanobis(dino_feat, cgcd_model)
    diffs = processed_feat.float().unsqueeze(1) - cgcd_model.means.float()

    if hasattr(cgcd_model, "inv_covs"):
        dist_sq = torch.einsum("bci,cij,bcj->bc", diffs, cgcd_model.inv_covs.float(), diffs)
    elif hasattr(cgcd_model, "inv_sqrt_covs"):
        whitened_diff = torch.einsum("cij,bcj->bci", cgcd_model.inv_sqrt_covs.float(), diffs)
        dist_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
    elif hasattr(cgcd_model, "covs"):
        inv_sqrt_covs = _build_inv_sqrt_covs_full(cgcd_model.covs)
        whitened_diff = torch.einsum("cij,bcj->bci", inv_sqrt_covs, diffs)
        dist_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
    else:
        raise AttributeError(
            "mahalanobis_margin requires one of: cgcd_model.inv_covs, " "cgcd_model.inv_sqrt_covs, or cgcd_model.covs."
        )

    if clear_idx < 0 or clear_idx >= dist_sq.shape[1]:
        raise ValueError(f"clear_idx out of range: clear_idx={clear_idx}, num_classes={dist_sq.shape[1]}")

    dist_clear = dist_sq[:, clear_idx]
    if dist_sq.shape[1] <= 1:
        return torch.sigmoid(torch.zeros_like(dist_clear))

    mask = torch.ones(dist_sq.shape[1], device=dist_sq.device, dtype=torch.bool)
    mask[clear_idx] = False
    dist_neg = dist_sq[:, mask]

    tau = max(float(contrastive_tau), 1e-6)
    agg_dist_neg = -tau * torch.logsumexp(-dist_neg / tau, dim=-1)
    margin = agg_dist_neg - dist_clear
    return torch.sigmoid(margin / max(float(mahalanobis_temp), 1e-6))


@torch.no_grad()
def get_cgcd_score(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    mode="clear",
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
):
    """
    CGCD 기반 pseudo label 신뢰도 점수 계산.

    Args:
        image_list: 이미지 배치 [B, C, H, W]
        dino_model: DINO 특징 추출기
        cgcd_model: CGCD 분류 모델
        dino_transform: DINO 입력용 변환
        clear_idx: 분류기에서 'clear' 클래스 인덱스
        mode: 점수 계산 방식
            - "clear": P(clear) - clear 클래스 확률 (높을수록 깨끗한 이미지)
            - "neg_distance": 1 - max(P(non-clear))
            - "contrastive": sigmoid((w_pos*logit(clear) - w_neg*agg_neg) / score_temp)
              where agg_neg = tau * logsumexp(logit(neg)/tau)
            - "anchor": (neg_weight * neg_dist) / (neg_weight * neg_dist + clear_weight * clear_dist + eps)
              where neg_dist is softmin(non-clear mahal dist, tau) if tau>0 else min(non-clear mahal dist)
              and the existing contrastive_pos_weight/contrastive_neg_weight/contrastive_tau are reused as
              clear_weight/neg_weight/softmin_tau for this mode.

    Returns:
        score: [B] 크기의 텐서
    """
    use_amp = amp_dtype is not None and image_list.is_cuda
    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        # DINO 특징 추출
        dino_feat = dino_model(dino_transform(image_list)).pooler_output

        # CGCD 로짓 계산
        _, logits = cgcd_model(dino_feat)

    logits = logits.float()
    probs = F.softmax(logits, dim=-1)

    if is_cgcd_anchor_mode(mode):
        return get_cgcd_anchor_score(
            image_list=image_list,
            dino_model=dino_model,
            cgcd_model=cgcd_model,
            dino_transform=dino_transform,
            clear_idx=clear_idx,
            amp_dtype=amp_dtype,
            clear_weight=contrastive_pos_weight,
            neg_weight=contrastive_neg_weight,
            softmin_tau=contrastive_tau,
        )

    if mode == "clear":
        return probs[:, clear_idx]

    elif mode == "neg_distance":
        mask = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
        mask[clear_idx] = False
        neg_probs = probs[:, mask]
        if neg_probs.shape[1] == 0:
            return torch.ones_like(probs[:, clear_idx])
        return 1.0 - neg_probs.max(dim=-1).values

    elif mode == "contrastive":
        mask = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
        mask[clear_idx] = False
        pos_logit = logits[:, clear_idx]
        neg_logits = logits[:, mask]

        tau = max(float(contrastive_tau), 1e-6)
        score_temp = max(float(contrastive_score_temp), 1e-6)
        w_pos = float(contrastive_pos_weight)
        w_neg = float(contrastive_neg_weight)

        # Smooth hard-negative aggregation.
        if neg_logits.shape[1] == 0:
            neg_agg = torch.zeros_like(pos_logit)
        else:
            neg_agg = tau * torch.logsumexp(neg_logits / tau, dim=-1)

        margin_like = (w_pos * pos_logit) - (w_neg * neg_agg)
        return torch.sigmoid(margin_like / score_temp)
    else:
        raise ValueError(
            f"알 수 없는 cgcd_score_mode: {mode}. 'clear', 'neg_distance', 'contrastive', 'anchor' 중 선택하세요."
        )


@torch.no_grad()
def get_cgcd_score_ver2(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    mode="clear",
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
    mahalanobis_temp=20.0,  # 절대 거리 스케일링을 위한 온도 하이퍼파라미터
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
            mask = torch.zeros_like(inv_sqrt_evals)
            # eigh는 오름차순 정렬이므로 큰 고유값(top-k)은 뒤쪽.
            mask[:, -k:] = 1.0
            inv_sqrt_evals = inv_sqrt_evals * mask

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
    n_classes = logits.shape[1]
    if clear_idx < 0 or clear_idx >= n_classes:
        raise ValueError(f"clear_idx out of range: {clear_idx}, num_classes={n_classes}")

    means_dim = int(cgcd_model.means.shape[-1]) if hasattr(cgcd_model, "means") else int(processed_feat.shape[-1])
    if int(processed_feat.shape[-1]) != means_dim:
        processed_feat = _manual_preprocess_for_mahalanobis(dino_feat)
    if int(processed_feat.shape[-1]) != means_dim:
        raise RuntimeError(
            "CGCD preprocessing dimension mismatch: "
            f"processed_feat_dim={int(processed_feat.shape[-1])}, means_dim={means_dim}. "
            "Check cgcd pca/scaler config."
        )

    if mode == "mahalanobis_margin":
        diffs = processed_feat.float().unsqueeze(1) - cgcd_model.means.float()
        if hasattr(cgcd_model, "inv_covs"):
            dist_sq = torch.einsum("bci,cij,bcj->bc", diffs, cgcd_model.inv_covs.float(), diffs)
        elif hasattr(cgcd_model, "inv_sqrt_covs"):
            whitened_diff = torch.einsum("cij,bcj->bci", cgcd_model.inv_sqrt_covs.float(), diffs)
            dist_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
        elif hasattr(cgcd_model, "covs"):
            inv_sqrt_covs = _build_inv_sqrt_covs(cgcd_model.covs, top_k=None)
            whitened_diff = torch.einsum("cij,bcj->bci", inv_sqrt_covs, diffs)
            dist_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
        else:
            raise AttributeError(
                "mahalanobis_margin requires one of: cgcd_model.inv_covs, "
                "cgcd_model.inv_sqrt_covs, or cgcd_model.covs."
            )

        dist_clear = dist_sq[:, clear_idx]
        mask = torch.ones(dist_sq.shape[1], device=dist_sq.device, dtype=torch.bool)
        mask[clear_idx] = False
        dist_neg = dist_sq[:, mask]

        tau = max(float(contrastive_tau), 1e-6)
        if dist_neg.shape[1] == 0:
            agg_dist_neg = torch.zeros_like(dist_clear)
        else:
            agg_dist_neg = -tau * torch.logsumexp(-dist_neg / tau, dim=-1)

        margin = agg_dist_neg - dist_clear
        return torch.sigmoid(margin / max(float(mahalanobis_temp), 1e-6))

    elif mode == "mahalanobis_pca":
        diffs = processed_feat.float().unsqueeze(1) - cgcd_model.means.float()

        if hasattr(cgcd_model, "inv_sqrt_covs") and mahalanobis_pca_top_k is None:
            inv_sqrt_covs = cgcd_model.inv_sqrt_covs.float()
        elif hasattr(cgcd_model, "covs"):
            inv_sqrt_covs = _build_inv_sqrt_covs(cgcd_model.covs, top_k=mahalanobis_pca_top_k)
        elif hasattr(cgcd_model, "inv_covs") and mahalanobis_pca_top_k is None:
            # inv_sqrt_covs가 없는 경우 full Mahalanobis로 fallback.
            dist_sq = torch.einsum("bci,cij,bcj->bc", diffs, cgcd_model.inv_covs.float(), diffs)
            dist_clear = dist_sq[:, clear_idx]
            mask = torch.ones(dist_sq.shape[1], device=dist_sq.device, dtype=torch.bool)
            mask[clear_idx] = False
            dist_neg = dist_sq[:, mask]
            tau = max(float(contrastive_tau), 1e-6)
            if dist_neg.shape[1] == 0:
                agg_dist_neg = torch.zeros_like(dist_clear)
            else:
                agg_dist_neg = -tau * torch.logsumexp(-dist_neg / tau, dim=-1)
            margin = agg_dist_neg - dist_clear
            return torch.sigmoid(margin / max(float(mahalanobis_temp), 1e-6))
        else:
            raise AttributeError(
                "mahalanobis_pca requires one of: cgcd_model.inv_sqrt_covs, cgcd_model.covs, "
                "or cgcd_model.inv_covs (fallback only when top-k is None)."
            )

        whitened_diff = torch.einsum("cij,bcj->bci", inv_sqrt_covs, diffs)
        dist_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
        dist_clear = dist_sq[:, clear_idx]
        mask = torch.ones(dist_sq.shape[1], device=dist_sq.device, dtype=torch.bool)
        mask[clear_idx] = False
        dist_neg = dist_sq[:, mask]

        tau = max(float(contrastive_tau), 1e-6)
        if dist_neg.shape[1] == 0:
            agg_dist_neg = torch.zeros_like(dist_clear)
        else:
            agg_dist_neg = -tau * torch.logsumexp(-dist_neg / tau, dim=-1)

        margin = agg_dist_neg - dist_clear
        return torch.sigmoid(margin / max(float(mahalanobis_temp), 1e-6))

    else:
        raise ValueError("알 수 없는 cgcd_score_mode: " f"{mode}. 지원 모드: mahalanobis_margin, mahalanobis_pca")


@torch.no_grad()
def get_reliable(
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
    cgcd_score_mode="clear",
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
    """
    신뢰할 수 있는 pseudo label 선택.
    Teacher 출력이 student와 reference보다 좋을 때만 pseudo label 업데이트.

    Args:
        pseudo_update_mode: pseudo label 업데이트 전략
            - "musiq": MUSIQ (시각적 품질) 점수만 사용
            - "fgresq": FGResQ (세밀 품질) 점수만 사용
            - "cgcd": CGCD (의미적 명확도) 점수만 사용
            - "alternate": 짝수 epoch은 MUSIQ, 홀수 epoch은 CGCD 번갈아 사용
        cgcd_score_mode: CGCD 점수 계산 방식
            - "clear", "neg_distance", "contrastive"
            - "anchor": (neg_weight*neg_dist) / (neg_weight*neg_dist + clear_weight*clear_dist + eps)
              (tau<=0이면 hardest negative, tau>0이면 softmin negative)
            - "mahalanobis_margin", "mahalanobis_pca"
    """
    total_candidates = teacher_predict.shape[0]
    updated_indices = []
    final_pseudo_labels = pseudo_list.clone()

    # 현재 epoch에서 사용할 모드 결정
    if pseudo_update_mode == "musiq":
        use_musiq = True
        use_fgresq = False
    elif pseudo_update_mode == "fgresq":
        use_musiq = True
        use_fgresq = True
    elif pseudo_update_mode == "cgcd":
        use_musiq = False
        use_fgresq = False
    elif pseudo_update_mode == "alternate":
        # 짝수 epoch → MUSIQ, 홀수 epoch → CGCD (CGCD 모델 없으면 MUSIQ fallback)
        use_musiq = (epoch % 2 == 0) or (dino_model is None) or (cgcd_model is None)
        use_fgresq = False
    else:
        raise ValueError(f"알 수 없는 pseudo_update_mode: {pseudo_update_mode}")

    if use_musiq:
        if use_fgresq:
            # FGResQ 기반 업데이트 (세밀 품질)
            score_teacher = get_fgresq_score(iqa_metric, teacher_predict)
            score_student = get_fgresq_score(iqa_metric, student_predict)
            if score_reference is None:
                score_reference = get_fgresq_score(iqa_metric, pseudo_list)
            update_mode = "FGResQ"
        else:
            # MUSIQ 기반 업데이트 (시각적 품질)
            score_teacher = get_musiq_score(iqa_metric, teacher_predict)
            score_student = get_musiq_score(iqa_metric, student_predict)
            if score_reference is None:
                score_reference = get_musiq_score(iqa_metric, pseudo_list)
            update_mode = "MUSIQ"
    else:
        # CGCD 기반 업데이트 (의미적 명확도)
        clear_idx = orig2classifier[0] if orig2classifier is not None else 0
        ver2_modes = {"mahalanobis_margin", "mahalanobis_pca"}
        score_fn = get_cgcd_score_ver2 if cgcd_score_mode in ver2_modes else get_cgcd_score
        score_kwargs = dict(
            mode=cgcd_score_mode,
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
        )
        if score_fn is get_cgcd_score_ver2:
            score_kwargs["mahalanobis_temp"] = mahalanobis_temp
            score_kwargs["mahalanobis_pca_top_k"] = mahalanobis_pca_top_k

        score_teacher = score_fn(teacher_predict, dino_model, cgcd_model, dino_transform, clear_idx, **score_kwargs)
        score_student = score_fn(student_predict, dino_model, cgcd_model, dino_transform, clear_idx, **score_kwargs)
        score_reference = score_fn(pseudo_list, dino_model, cgcd_model, dino_transform, clear_idx, **score_kwargs)
        update_mode = f"CGCD_{cgcd_score_mode}"

    reference_exists_mask = _resolve_reference_exists_mask(pseudo_names, teacher_predict.device)
    effective_reference_score = torch.where(reference_exists_mask, score_reference, score_student)
    if reference_exists_mask.numel() > 0 and torch.any(~reference_exists_mask):
        final_pseudo_labels[~reference_exists_mask] = student_predict[~reference_exists_mask]

    margin = float(update_margin)
    stats = _summarize_pseudo_update_delta(
        score_teacher,
        score_student,
        score_reference,
        margin,
        reference_exists_mask=reference_exists_mask,
    )
    for idx in range(total_candidates):
        threshold = torch.maximum(score_student[idx], effective_reference_score[idx]) + margin
        if score_teacher[idx] > threshold:
            final_pseudo_labels[idx] = teacher_predict[idx]
            updated_indices.append(idx)
            save_image_tensor(teacher_predict[idx], pseudo_names[idx])

    if return_stats:
        return final_pseudo_labels, len(updated_indices), update_mode, stats
    return final_pseudo_labels, len(updated_indices), update_mode


def freeze_teachers_parameters(teacher):
    """Freeze all parameters in teacher model."""
    for p in teacher.parameters():
        p.requires_grad = False


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from 0 to 1."""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, consistency=0.2, rampup_length=100.0):
    """Calculate current consistency weight with rampup."""
    return consistency * sigmoid_rampup(epoch, rampup_length)


# @torch.no_grad()
# def log_pseudo_labels(pseudo_patches_dir, writer, epoch, rank):
#     """Log pseudo label samples to TensorBoard."""
#     if rank != 0 or writer is None:
#         return

#     from datasets_my import NEW_CLASSES

#     if NEW_CLASSES is None:
#         return

#     for deg_name in NEW_CLASSES:
#         pseudo_dir = os.path.join(pseudo_patches_dir, deg_name)

#         if os.path.isdir(pseudo_dir):
#             images = sorted([f for f in os.listdir(pseudo_dir) if f.endswith(".png")])
#             if len(images) > 0:
#                 num_samples = min(5, len(images))
#                 indices = np.linspace(0, len(images) - 1, num_samples, dtype=int)

#                 for i, idx in enumerate(indices):
#                     img_path = os.path.join(pseudo_dir, images[idx])
#                     # Use torchvision to read image
#                     img_tensor = read_image(img_path).float() / 255.0
#                     writer.add_image(f"PseudoLabel/{deg_name}/sample_{i+1}", img_tensor, global_step=epoch)


@torch.no_grad()
def log_pseudo_labels(pseudo_patches_dir, writer, epoch, rank):
    """Log pseudo label samples to TensorBoard."""
    if rank != 0 or writer is None:
        return

    from datasets_my import NEW_CLASSES

    if NEW_CLASSES is None:
        return

    for deg_name in NEW_CLASSES:
        pseudo_dir = os.path.join(pseudo_patches_dir, deg_name)

        if os.path.isdir(pseudo_dir):
            images = sorted([f for f in os.listdir(pseudo_dir) if f.endswith(".png")])
            if len(images) > 0:
                # 1. 샘플 개수를 10개로 늘립니다.
                num_samples = min(10, len(images))
                indices = np.linspace(0, len(images) - 1, num_samples, dtype=int)

                img_list = []
                for idx in indices:
                    img_path = os.path.join(pseudo_dir, images[idx])
                    img_tensor = read_image(img_path).float() / 255.0
                    img_list.append(img_tensor)

                # 2. 이미지 텐서 리스트를 하나의 배치 텐서로 합칩니다. 모양: [N, C, H, W]
                batch_tensor = torch.stack(img_list)

                # 3. make_grid로 10장의 이미지를 1열에 5개씩(nrow=5) 배치합니다.
                # (10장이므로 자연스럽게 2줄(row)이 됩니다.)
                # padding=2는 이미지 사이의 간격(픽셀)입니다. 취향껏 조절하세요.
                grid_img = make_grid(batch_tensor, nrow=5, padding=2)

                # 4. 텐서보드에 묶여진 그리드 이미지 하나만 올립니다.
                writer.add_image(f"PseudoLabel/{deg_name}", grid_img, global_step=epoch)

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
    """
    Initialize pseudo labels using teacher model predictions.

    Args:
        teacher_model: Teacher model for generating pseudo labels
        dino_model: DINO feature extractor
        cgcd_model: CGCD signal module
        loader: DataLoader for unlabeled data
        device: CUDA device
        dino_transform: Transform to apply before DINO
        amp_dtype: AMP dtype (torch.float16 or torch.bfloat16)
        use_class_routing: If True, route to LoRA expert based on predicted class (Multi-LoRA)
    """
    teacher_model.eval()
    dino_model.eval()
    cgcd_model.eval()

    if show_progress:
        desc = "Initializing Pseudo Labels with Teacher"
        if rank != 0:
            desc = f"{desc} (rank{rank})"
        iterator = tqdm(loader, desc=desc)
    else:
        iterator = loader

    for _, batch in enumerate(iterator):
        lq_unlabeled = batch[0].to(device, non_blocking=True)
        pseudo_paths = batch[2]

        with autocast("cuda", dtype=amp_dtype):
            dino_feat = dino_model(dino_transform(lq_unlabeled)).pooler_output
            embedding, logits = cgcd_model(dino_feat)

            if use_class_routing:
                # Multi-LoRA: Route to correct expert based on predicted class
                class_ids = torch.argmax(logits, dim=1)
                restored_teacher = teacher_model(lq_unlabeled, embedding, class_ids=class_ids)
            else:
                # Single LoRA: No class routing needed
                restored_teacher = teacher_model(lq_unlabeled, embedding)

        for idx, path in enumerate(pseudo_paths):
            save_image_tensor(restored_teacher[idx], path)

    if show_progress:
        print("✓ Initial Pseudo Labeling Completed!")


@torch.no_grad()
def copy_pseudo_labels(loader, rank=0, show_progress=True):
    """Materialize a full pseudo bank by copying unlabeled inputs into pseudo slots."""
    if show_progress:
        desc = "Copying Inputs to Pseudo Bank"
        if rank != 0:
            desc = f"{desc} (rank{rank})"
        iterator = tqdm(loader, desc=desc)
    else:
        iterator = loader

    copied = 0
    for _, batch in enumerate(iterator):
        lq_unlabeled = batch[0]
        pseudo_paths = batch[2]
        for idx, path in enumerate(pseudo_paths):
            save_image_tensor(lq_unlabeled[idx], path)
            copied += 1

    if show_progress:
        print(f"✓ Pseudo Label Copy Completed! ({copied} files)")


def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def setup_runtime(cfg, rank):
    # CUDA optimization settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Seed setting
    seed_everything(cfg.train.seed + rank)

    # Create directories (Rank 0 only)
    if rank == 0:
        os.makedirs(os.path.join(cfg.train.log_dir, cfg.exp_name), exist_ok=True)
        os.makedirs(cfg.train.pseudo_patches_dir, exist_ok=True)


def get_iter_steps_per_epoch(step_anchor, labeled_steps, unlabeled_steps):
    step_anchor = str(step_anchor).lower()
    if step_anchor == "unlabeled":
        steps_per_epoch = unlabeled_steps
    elif step_anchor == "labeled":
        steps_per_epoch = labeled_steps
    elif step_anchor == "max":
        steps_per_epoch = max(labeled_steps, unlabeled_steps)
    elif step_anchor == "min":
        steps_per_epoch = min(labeled_steps, unlabeled_steps)
    else:
        raise ValueError(f"Invalid train.step_anchor={step_anchor}. Choose one of: unlabeled, labeled, max, min.")
    return max(1, int(steps_per_epoch))


def init_iter_stage_runtime(cfg):
    from torch.utils.tensorboard import SummaryWriter

    current_stage = cfg.incremental.stage
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    setup_runtime(cfg, rank)

    checkpoint_dir = os.path.join(cfg.train.save_dir, cfg.train.config_parent_dir, cfg.exp_name)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    use_bf16 = bool(getattr(cfg.train, "use_bf16", False))
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.train.log_dir, cfg.exp_name))
        print(f"\n[Stage {current_stage}] Start - Device: {device}, World Size: {world_size}")
        print(f"AMP dtype: {'BF16' if use_bf16 else 'FP16'}")
    else:
        writer = None

    return {
        "current_stage": current_stage,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "checkpoint_dir": checkpoint_dir,
        "use_bf16": use_bf16,
        "amp_dtype": amp_dtype,
        "writer": writer,
    }


def resolve_iter_train_options(cfg):
    val_freq = int(getattr(cfg.train, "val_freq", 1) or 1)
    raw_val_step_freq = getattr(cfg.train, "val_step_freq", None)
    if raw_val_step_freq is None:
        raw_val_step_freq = getattr(cfg.train, "val_freq_steps", 0)
    val_step_freq = int(raw_val_step_freq or 0)
    if val_step_freq < 0:
        raise ValueError(f"val_step_freq must be >= 0, got {val_step_freq}")

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

    return {
        "use_validation": bool(getattr(cfg.train, "use_validation", False)),
        "use_real_eval": bool(getattr(cfg.train, "use_real_eval", False)),
        "val_freq": val_freq,
        "val_step_freq": val_step_freq,
        "val_freq_steps": val_step_freq,
        "save_freq": save_freq,
        "save_step_freq": save_step_freq,
        "save_freq_steps": save_step_freq,
        "pseudo_log_freq_steps": pseudo_log_freq_steps,
        "real_eval_freq": int(getattr(cfg.train, "real_eval_freq", val_freq)),
        "real_eval_freq_steps": int(getattr(cfg.train, "real_eval_freq_steps", 0) or 0),
        "real_eval_max_side": int(getattr(cfg.train, "real_eval_max_side", 0) or 0),
        "real_eval_allow_runtime_resize_retry": bool(getattr(cfg.train, "real_eval_allow_runtime_resize_retry", True)),
    }


def build_iter_models_optimizer_scaler(cfg, device, rank, use_bf16, model_builder):
    from torch.amp import GradScaler

    if rank == 0:
        student, teacher, dino, cgcd = model_builder(cfg, device, rank)
    dist.barrier()
    if rank != 0:
        student, teacher, dino, cgcd = model_builder(cfg, device, rank)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(cgcd.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = GradScaler("cuda") if not use_bf16 else None
    return student, teacher, dino, cgcd, optimizer, scaler


def resolve_iter_schedule_and_scheduler(
    cfg,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    cfg_getter,
    extra_schedule_parser=None,
):
    import math

    base_mode = normalize_mode_name(getattr(cfg.train, "pseudo_update_mode", "musiq"))
    cgcd_score_mode = str(cfg_getter(cfg, "cgcd_score_mode", "mahalanobis_margin"))

    round_metric_schedule = parse_round_metric_schedule(
        getattr(cfg.train, "round_metric_schedule", None)
        or getattr(cfg.train, "vlm_metric_schedule", "musiq,clipiqa"),
        fallback_mode=base_mode,
    )
    legacy_round_steps = int(getattr(cfg.train, "pseudo_round_steps", 0) or 0)
    vlm_round_steps = int(getattr(cfg.train, "vlm_switch_every", 0) or 0)
    round_steps = legacy_round_steps if legacy_round_steps > 0 else vlm_round_steps
    round_mode_policy = str(getattr(cfg.train, "round_mode_policy", "cycle")).lower()
    if round_mode_policy not in ("hold", "cycle"):
        raise ValueError(f"Unsupported round_mode_policy: {round_mode_policy}")

    max_train_steps = int(getattr(cfg.train, "max_train_steps", 0) or 0)
    if max_train_steps < 0:
        raise ValueError(f"max_train_steps must be >= 0, got {max_train_steps}")
    auto_round_steps = False
    if round_steps <= 0 and max_train_steps > 0:
        round_count = max(1, len(round_metric_schedule))
        round_steps = max(1, math.ceil(max_train_steps / round_count))
        auto_round_steps = True

    labeled_steps = len(labeled_loader)
    unlabeled_steps = len(unlabeled_loader)
    step_anchor = str(getattr(cfg.train, "step_anchor", "unlabeled")).lower()
    steps_per_epoch = get_iter_steps_per_epoch(step_anchor, labeled_steps, unlabeled_steps)

    configured_epochs_hint = int(getattr(cfg.train, "epochs", 0) or 0)
    if configured_epochs_hint <= 0:
        configured_epochs_hint = max(1, math.ceil(max_train_steps / steps_per_epoch) if max_train_steps > 0 else 1)

    legacy_musiq_warmup_epochs = int(getattr(cfg.train, "musiq_warmup_epochs", 0))
    musiq_warmup_steps = int(getattr(cfg.train, "musiq_warmup_steps", 0) or 0)
    if musiq_warmup_steps <= 0:
        if legacy_musiq_warmup_epochs < 0:
            musiq_warmup_steps = (
                (max_train_steps // 2) if max_train_steps > 0 else ((configured_epochs_hint // 2) * steps_per_epoch)
            )
        else:
            musiq_warmup_steps = max(0, legacy_musiq_warmup_epochs * steps_per_epoch)
    if max_train_steps > 0:
        musiq_warmup_steps = min(musiq_warmup_steps, max_train_steps)

    rampup_steps = int(getattr(cfg.train, "rampup_steps", 0) or 0)
    if rampup_steps <= 0:
        if max_train_steps > 0:
            rampup_steps = max(1, max_train_steps // 10)
        else:
            rampup_steps = max(0, int(getattr(cfg.train, "rampup_epoch", 0)) * steps_per_epoch)
    if max_train_steps > 0 and rampup_steps > max_train_steps:
        rampup_steps = max_train_steps

    extra_schedule = {}
    if callable(extra_schedule_parser):
        extra_schedule = extra_schedule_parser(cfg) or {}
        if not isinstance(extra_schedule, dict):
            raise TypeError("extra_schedule_parser must return a dict")

    scheduler_step_per_iter = bool(getattr(cfg.train, "iter_scheduler", True))
    if scheduler_step_per_iter:
        scheduler_t_max = int(max_train_steps if max_train_steps > 0 else (configured_epochs_hint * steps_per_epoch))
    else:
        scheduler_t_max = int(configured_epochs_hint)
    scheduler_t_max = max(1, scheduler_t_max)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=1e-6)

    schedule_cfg = {
        "base_mode": base_mode,
        "cgcd_score_mode": cgcd_score_mode,
        "round_metric_schedule": round_metric_schedule,
        "round_steps": round_steps,
        "round_mode_policy": round_mode_policy,
        "max_train_steps": max_train_steps,
        "auto_round_steps": auto_round_steps,
        "steps_per_epoch": steps_per_epoch,
        "configured_epochs_hint": configured_epochs_hint,
        "musiq_warmup_steps": musiq_warmup_steps,
        "rampup_steps": rampup_steps,
        "scheduler_step_per_iter": scheduler_step_per_iter,
        "scheduler_t_max": scheduler_t_max,
        "scheduler": scheduler,
    }
    schedule_cfg.update(extra_schedule)
    return schedule_cfg


def init_iter_iqa_metrics(round_metric_schedule, musiq_warmup_steps, device, rank):
    import pyiqa

    need_musiq = ("musiq" in round_metric_schedule) or (musiq_warmup_steps > 0)
    need_fgresq = "fgresq" in round_metric_schedule
    need_clipiqa = "clipiqa" in round_metric_schedule

    iqa_metrics = {"musiq": None, "fgresq": None, "clipiqa": None}
    if need_musiq:
        iqa_metrics["musiq"] = pyiqa.create_metric("musiq", as_loss=True).to(device)
        if rank == 0:
            print("✓ MUSIQ metric initialized")
    if need_fgresq:
        iqa_metrics["fgresq"] = create_fgresq_metric(device=device)
        if rank == 0:
            print("✓ FGResQ metric initialized")
    if need_clipiqa:
        iqa_metrics["clipiqa"] = pyiqa.create_metric("clipiqa", as_loss=False, device=device)
        if rank == 0:
            print("✓ CLIP-IQA metric initialized")
    return iqa_metrics


def print_iter_train_plan(rank, options, schedule_cfg, extra_plan_lines=None):
    if rank != 0:
        return

    print(
        f"✓ Iter-round schedule: steps={schedule_cfg['round_steps']}, policy={schedule_cfg['round_mode_policy']}, "
        f"schedule={schedule_cfg['round_metric_schedule']}"
    )
    print(f"✓ CGCD score mode (used only in 'cgcd' rounds): {schedule_cfg['cgcd_score_mode']}")
    if schedule_cfg["auto_round_steps"]:
        print(
            f"✓ Round step auto-set: ceil(max_train_steps/num_rounds) = "
            f"ceil({schedule_cfg['max_train_steps']}/{len(schedule_cfg['round_metric_schedule'])}) = "
            f"{schedule_cfg['round_steps']}"
        )
    print(
        f"✓ Step controls: steps_per_epoch={schedule_cfg['steps_per_epoch']}, "
        f"musiq_warmup_steps={schedule_cfg['musiq_warmup_steps']}, rampup_steps={schedule_cfg['rampup_steps']}"
    )
    if extra_plan_lines is not None:
        lines = extra_plan_lines(schedule_cfg) if callable(extra_plan_lines) else extra_plan_lines
        for line in lines or []:
            print(str(line))
    if options["use_validation"]:
        print(f"✓ Validation trigger: epoch-based (every {options['val_freq']} epochs)")
        if options["val_step_freq"] > 0:
            print(f"✓ Validation trigger: step-based (every {options['val_step_freq']} steps)")
        else:
            print("✓ Validation trigger: step-based disabled")
    if options["use_real_eval"]:
        if options["real_eval_freq_steps"] > 0:
            print(f"✓ Real eval trigger: step-based (every {options['real_eval_freq_steps']} steps)")
        else:
            print(f"✓ Real eval trigger: epoch-based (every {options['real_eval_freq']} epochs)")
        print(
            "✓ Real eval resize retry: "
            f"{'enabled' if options['real_eval_allow_runtime_resize_retry'] else 'disabled (strict benchmark mode)'}"
        )
    print(
        f"✓ LR scheduler: {'iter' if schedule_cfg['scheduler_step_per_iter'] else 'epoch'}-based, "
        f"T_max={schedule_cfg['scheduler_t_max']}"
    )
    if schedule_cfg["max_train_steps"] > 0:
        print(f"✓ Max train steps: {schedule_cfg['max_train_steps']}")
    print(f"✓ Checkpoint save: epoch-based (every {options['save_freq']} epochs)")
    if options["save_step_freq"] > 0:
        print(f"✓ Checkpoint save: step-based (every {options['save_step_freq']} steps)")
    else:
        print("✓ Checkpoint save: step-based disabled")
    if options["pseudo_log_freq_steps"] > 0:
        print(f"✓ Pseudo image log: step-based (every {options['pseudo_log_freq_steps']} steps)")
    else:
        print("✓ Pseudo image log: epoch-end")
    if schedule_cfg["musiq_warmup_steps"] > 0:
        print(
            f"✓ MUSIQ warmup enabled: first {schedule_cfg['musiq_warmup_steps']} steps use MUSIQ, "
            "then switch to iteration-based round schedule"
        )


def load_iter_training_state(
    cfg,
    student,
    teacher,
    cgcd,
    optimizer,
    scheduler,
    device,
    rank,
    max_train_steps,
    steps_per_epoch,
    configured_epochs_hint,
    scheduler_step_per_iter,
    load_checkpoints_fn,
):
    import math

    start_epoch, global_step, best_psnr, _ = load_checkpoints_fn(
        student, teacher, cgcd, optimizer, scheduler, cfg, device, rank
    )

    configured_epochs = int(configured_epochs_hint)
    if max_train_steps > 0:
        remaining_steps = max(0, max_train_steps - int(global_step))
        additional_epochs = math.ceil(remaining_steps / steps_per_epoch) if remaining_steps > 0 else 0
        total_epochs = max(configured_epochs, int(start_epoch) + max(0, additional_epochs) - 1)
    else:
        total_epochs = configured_epochs
    total_epochs = max(1, int(total_epochs))

    if not scheduler_step_per_iter:
        scheduler_t_max = max(1, total_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=1e-6)

    is_resume_run = start_epoch > 1
    init_validate_on_resume = bool(getattr(cfg.train, "init_validate_on_resume", False))
    cfg_init_pseudo = bool(getattr(cfg.train, "init_pseudo_label", False))
    cfg_copy_pseudo = bool(getattr(cfg.train, "copy_pseudo_label", False))
    run_init_validate = bool(getattr(cfg.train, "init_validate", False)) and (
        init_validate_on_resume or not is_resume_run
    )
    run_init_pseudo = cfg_init_pseudo and not is_resume_run
    run_init_copy = cfg_copy_pseudo and (not cfg_init_pseudo) and (not is_resume_run)

    if rank == 0 and cfg_init_pseudo and cfg_copy_pseudo and not is_resume_run:
        print("[InitPseudo] copy_pseudo_label is ignored because init_pseudo_label is enabled.")

    if rank == 0 and is_resume_run:
        if getattr(cfg.train, "init_validate", False) and not init_validate_on_resume:
            print("[Resume] Skip initial validation at epoch 0 to keep TensorBoard scalar steps monotonic.")
        if cfg_init_pseudo:
            print("[Resume] Skip pseudo-label initialization on resume.")
        if cfg_copy_pseudo:
            print("[Resume] Skip pseudo-label copy initialization on resume.")

    return {
        "start_epoch": int(start_epoch),
        "global_step": int(global_step),
        "best_psnr": float(best_psnr),
        "total_epochs": int(total_epochs),
        "scheduler": scheduler,
        "run_init_validate": run_init_validate,
        "run_init_pseudo": run_init_pseudo,
        "run_init_copy": run_init_copy,
    }


def prepare_iter_ddp_models(student, cgcd, device, local_rank, start_epoch):
    from torch.nn.parallel import DistributedDataParallel as DDP

    for param in student.parameters():
        param.requires_grad = True

    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())

    student = DDP(student.to(device), device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    cgcd = DDP(cgcd.to(device), device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    return student, cgcd, start_epoch


def run_iter_init_validation(
    run_init_validate,
    options,
    val_datasets,
    real_eval_datasets,
    teacher,
    dino,
    cgcd,
    device,
    cfg,
    writer,
    rank,
    world_size,
    amp_dtype,
    validate_stage_fn,
    evaluate_real_unpaired_fn,
):
    if options["use_validation"] and len(val_datasets) > 0 and run_init_validate:
        psnr_old, psnr_new, psnr_all = validate_stage_fn(
            teacher, dino, cgcd, val_datasets, device, 0, cfg, writer, rank, world_size, amp_dtype
        )
        if rank == 0:
            print(f"  Teacher validation: OLD={psnr_old:.2f}, NEW={psnr_new:.2f}, ALL={psnr_all:.2f}")
        dist.barrier()

    if options["use_real_eval"] and len(real_eval_datasets) > 0 and run_init_validate:
        evaluate_real_unpaired_fn(
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
            max_eval_side=options["real_eval_max_side"],
            allow_runtime_resize_retry=options["real_eval_allow_runtime_resize_retry"],
        )
        dist.barrier()


def run_iter_copy_pseudo_labeling(
    run_init_copy,
    cfg,
    unlabeled_loader,
    rank,
    world_size,
    copy_pseudo_labels_fn,
):
    from torch.utils.data import DataLoader, Subset

    if not run_init_copy:
        return

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
    if rank == 0:
        print(
            f"[CopyPseudo] Distributed copy enabled: world_size={world_size}, "
            f"per-rank samples≈{len(init_subset)}, batch_size={max(1, init_batch_size)}"
        )

    copy_pseudo_labels_fn(
        init_loader,
        rank=rank,
        show_progress=(rank == 0),
    )
    dist.barrier()


def run_iter_init_pseudo_labeling(
    run_init_pseudo,
    cfg,
    teacher,
    dino,
    cgcd,
    unlabeled_loader,
    device,
    amp_dtype,
    rank,
    world_size,
    initialize_pseudo_labels_fn,
    dino_transform,
):
    from torch.utils.data import DataLoader, Subset

    if not run_init_pseudo:
        return

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
    if rank == 0:
        print(
            f"[InitPseudo] Distributed init enabled: world_size={world_size}, "
            f"per-rank samples≈{len(init_subset)}, batch_size={max(1, init_batch_size)}"
        )

    initialize_pseudo_labels_fn(
        teacher,
        dino,
        cgcd.module,
        init_loader,
        device,
        dino_transform=dino_transform,
        amp_dtype=amp_dtype,
        use_class_routing=False,
        rank=rank,
        show_progress=(rank == 0),
    )
    dist.barrier()
