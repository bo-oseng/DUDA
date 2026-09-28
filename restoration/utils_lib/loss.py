import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torchvision.models import vgg16, vgg19


# ============================================
# Loss Functions
# ============================================


class StructureLoss(nn.Module):
    """Structure loss: 0.8 * L1 + 0.2 * L2."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, prediction, target):
        return 0.8 * self.l1(prediction, target) + 0.2 * self.l2(prediction, target)


class VGGPerceptualLoss(nn.Module):
    """VGG16 feature-space perceptual loss."""

    def __init__(self, vgg_model):
        super().__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3"}

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, prediction, target):
        losses = []
        prediction_features = self.output_features(prediction)
        target_features = self.output_features(target)
        for prediction_feature, target_feature in zip(prediction_features, target_features):
            losses.append(F.mse_loss(prediction_feature, target_feature))
        return sum(losses) / len(losses)


class GradientMagnitudeNoPadding(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, inp_feat):
        x_list = []
        for i in range(inp_feat.shape[1]):
            x_i = inp_feat[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
        return torch.cat(x_list, dim=1)


# Backward-compatible aliases used by existing trainers.
MyLoss = StructureLoss
PerpetualLoss = VGGPerceptualLoss
GetGradientNopadding = GradientMagnitudeNoPadding


def _resolve_cgcd_inner(cgcd_model):
    return cgcd_model.module if hasattr(cgcd_model, "module") else cgcd_model


def _preprocess_feature_for_cgcd(feature, cgcd_model):
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


def _build_inv_sqrt_covs(covs, top_k=None):
    covs = covs.float()
    evals, evecs = torch.linalg.eigh(covs)
    evals = evals.clamp(min=1e-8)
    inv_sqrt_evals = 1.0 / torch.sqrt(evals)

    if top_k is not None:
        dim = inv_sqrt_evals.shape[-1]
        k = int(max(1, min(int(top_k), dim)))
        mask = torch.zeros_like(inv_sqrt_evals)
        mask[:, -k:] = 1.0
        inv_sqrt_evals = inv_sqrt_evals * mask

    return torch.bmm(evecs, torch.bmm(torch.diag_embed(inv_sqrt_evals), evecs.mT))


def _softmin_torch(values, tau, dim=-1):
    tau = max(float(tau), 1e-6)
    return -tau * torch.logsumexp(-values / tau, dim=dim)


def _safe_ratio_torch(neg_ref, clear_ref, neg_weight=1.0, clear_weight=1.0, eps=1e-8):
    neg_term = float(neg_weight) * neg_ref
    clear_term = float(clear_weight) * clear_ref
    denom = (neg_term + clear_term).clamp_min(float(eps))
    return (neg_term - clear_term) / denom


def _compute_mahal_dist_sq(processed_feat, cgcd_model):
    means = cgcd_model.means.float().to(processed_feat.device)
    diffs = processed_feat.unsqueeze(1) - means

    if hasattr(cgcd_model, "inv_covs"):
        inv_covs = cgcd_model.inv_covs.float().to(processed_feat.device)
        dist_sq = torch.einsum("bci,cij,bcj->bc", diffs, inv_covs, diffs)
    elif hasattr(cgcd_model, "inv_sqrt_covs"):
        inv_sqrt_covs = cgcd_model.inv_sqrt_covs.float().to(processed_feat.device)
        whitened_diff = torch.einsum("cij,bcj->bci", inv_sqrt_covs, diffs)
        dist_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
    elif hasattr(cgcd_model, "covs"):
        inv_sqrt_covs = _build_inv_sqrt_covs(cgcd_model.covs.to(processed_feat.device), top_k=None)
        whitened_diff = torch.einsum("cij,bcj->bci", inv_sqrt_covs, diffs)
        dist_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
    else:
        raise AttributeError(
            "CGCD clear mahal metrics require one of: cgcd_model.inv_covs, cgcd_model.inv_sqrt_covs, or cgcd_model.covs."
        )

    return dist_sq.clamp_min(0.0)


def compute_cgcd_clear_anchor_distances(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    amp_dtype=None,
):
    """Compute clear/non-clear Mahalanobis distances from a frozen CGCD snapshot."""
    cgcd_model = _resolve_cgcd_inner(cgcd_model)
    image_list = torch.clamp(image_list, 0.0, 1.0)
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
            processed_feat = _preprocess_feature_for_cgcd(dino_feat, cgcd_model)

    processed_feat = processed_feat.float()
    means_dim = int(cgcd_model.means.shape[-1]) if hasattr(cgcd_model, "means") else int(processed_feat.shape[-1])
    if int(processed_feat.shape[-1]) != means_dim:
        processed_feat = _preprocess_feature_for_cgcd(dino_feat, cgcd_model).float()
    if int(processed_feat.shape[-1]) != means_dim:
        raise RuntimeError(
            "CGCD preprocessing dimension mismatch: "
            f"processed_feat_dim={int(processed_feat.shape[-1])}, means_dim={means_dim}."
        )

    dist_sq = _compute_mahal_dist_sq(processed_feat, cgcd_model)
    if clear_idx < 0 or clear_idx >= dist_sq.shape[1]:
        raise ValueError(f"clear_idx out of range: clear_idx={clear_idx}, num_classes={dist_sq.shape[1]}")

    mahal_dists = torch.sqrt(dist_sq.clamp_min(0.0))
    clear_mahal_dist = mahal_dists[:, clear_idx]

    mask = torch.ones(mahal_dists.shape[1], device=mahal_dists.device, dtype=torch.bool)
    mask[clear_idx] = False
    neg_mahal_dists = mahal_dists[:, mask]
    min_neg_mahal_dist = neg_mahal_dists.min(dim=-1).values

    return processed_feat, {
        "clear_mahal_dist": clear_mahal_dist,
        "min_neg_mahal_dist": min_neg_mahal_dist,
    }


def compute_cgcd_clear_mahal_distances(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    amp_dtype=None,
    contrastive_tau=5.0,
):
    """Compute clear/non-clear Mahalanobis distances from a frozen CGCD snapshot."""
    cgcd_model = _resolve_cgcd_inner(cgcd_model)
    image_list = torch.clamp(image_list, 0.0, 1.0)
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
            processed_feat = _preprocess_feature_for_cgcd(dino_feat, cgcd_model)

    processed_feat = processed_feat.float()
    means_dim = int(cgcd_model.means.shape[-1]) if hasattr(cgcd_model, "means") else int(processed_feat.shape[-1])
    if int(processed_feat.shape[-1]) != means_dim:
        processed_feat = _preprocess_feature_for_cgcd(dino_feat, cgcd_model).float()
    if int(processed_feat.shape[-1]) != means_dim:
        raise RuntimeError(
            "CGCD preprocessing dimension mismatch: "
            f"processed_feat_dim={int(processed_feat.shape[-1])}, means_dim={means_dim}."
        )

    dist_sq = _compute_mahal_dist_sq(processed_feat, cgcd_model)
    if clear_idx < 0 or clear_idx >= dist_sq.shape[1]:
        raise ValueError(f"clear_idx out of range: clear_idx={clear_idx}, num_classes={dist_sq.shape[1]}")

    mahal_dists = torch.sqrt(dist_sq.clamp_min(0.0))
    clear_mahal_dist = mahal_dists[:, clear_idx]

    if mahal_dists.shape[1] <= 1:
        min_neg_mahal_dist = clear_mahal_dist
        mean_neg_mahal_dist = clear_mahal_dist
        contrastive_neg_mahal_dist = clear_mahal_dist
    else:
        mask = torch.ones(mahal_dists.shape[1], device=mahal_dists.device, dtype=torch.bool)
        mask[clear_idx] = False
        neg_mahal_dists = mahal_dists[:, mask]
        min_neg_mahal_dist = neg_mahal_dists.min(dim=-1).values
        mean_neg_mahal_dist = neg_mahal_dists.mean(dim=-1)
        contrastive_neg_mahal_dist = _softmin_torch(neg_mahal_dists, contrastive_tau, dim=-1)

    return processed_feat, {
        "clear_mahal_dist": clear_mahal_dist,
        "min_neg_mahal_dist": min_neg_mahal_dist,
        "mean_neg_mahal_dist": mean_neg_mahal_dist,
        "contrastive_neg_mahal_dist": contrastive_neg_mahal_dist,
    }


def compute_cgcd_clear_mahal_metrics(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    amp_dtype=None,
    contrastive_tau=5.0,
    clear_weight=1.0,
    neg_weight=1.0,
    eps=1e-8,
):
    """Compute analysis-oriented clear-reference CGCD metrics on top of Mahalanobis distances."""
    processed_feat, dist_dict = compute_cgcd_clear_mahal_distances(
        image_list=image_list,
        dino_model=dino_model,
        cgcd_model=cgcd_model,
        dino_transform=dino_transform,
        clear_idx=clear_idx,
        amp_dtype=amp_dtype,
        contrastive_tau=contrastive_tau,
    )

    clear_mahal_dist = dist_dict["clear_mahal_dist"]
    min_neg_mahal_dist = dist_dict["min_neg_mahal_dist"]
    mean_neg_mahal_dist = dist_dict["mean_neg_mahal_dist"]
    contrastive_neg_mahal_dist = dist_dict["contrastive_neg_mahal_dist"]

    metric_dict = dict(dist_dict)
    metric_dict.update(
        {
            "clear_mahal_margin": _safe_ratio_torch(
                min_neg_mahal_dist, clear_mahal_dist, neg_weight=neg_weight, clear_weight=clear_weight, eps=eps
            ),
            "clear_mahal_mean_margin": _safe_ratio_torch(
                mean_neg_mahal_dist, clear_mahal_dist, neg_weight=neg_weight, clear_weight=clear_weight, eps=eps
            ),
            "clear_mahal_contrastive_margin": _safe_ratio_torch(
                contrastive_neg_mahal_dist, clear_mahal_dist, neg_weight=neg_weight, clear_weight=clear_weight, eps=eps
            ),
        }
    )
    return processed_feat, metric_dict


def compute_cgcd_repr_and_score(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    mode="mahalanobis_margin",
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
    mahalanobis_temp=20.0,
):
    """Differentiable CGCD representation + score for restored image losses."""
    cgcd_model = _resolve_cgcd_inner(cgcd_model)
    image_list = torch.clamp(image_list, 0.0, 1.0)
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
            processed_feat = _preprocess_feature_for_cgcd(dino_feat, cgcd_model)

        logits = None
        if mode in {"clear", "contrastive"}:
            _, logits = cgcd_model(dino_feat)

    processed_feat = processed_feat.float()

    if mode in {"clear", "contrastive"}:
        logits = logits.float()
        n_classes = logits.shape[1]
        if clear_idx < 0 or clear_idx >= n_classes:
            raise ValueError(f"clear_idx out of range: {clear_idx}, num_classes={n_classes}")

        if mode == "clear":
            score = F.softmax(logits, dim=-1)[:, clear_idx]
        else:
            mask = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
            mask[clear_idx] = False
            pos_logit = logits[:, clear_idx]
            neg_logits = logits[:, mask]
            tau = max(float(contrastive_tau), 1e-6)
            score_temp = max(float(contrastive_score_temp), 1e-6)
            w_pos = float(contrastive_pos_weight)
            w_neg = float(contrastive_neg_weight)
            if neg_logits.shape[1] == 0:
                neg_agg = torch.zeros_like(pos_logit)
            else:
                neg_agg = tau * torch.logsumexp(neg_logits / tau, dim=-1)
            margin_like = (w_pos * pos_logit) - (w_neg * neg_agg)
            score = torch.sigmoid(margin_like / score_temp)
        return processed_feat, score.float()

    means_dim = int(cgcd_model.means.shape[-1]) if hasattr(cgcd_model, "means") else int(processed_feat.shape[-1])
    if int(processed_feat.shape[-1]) != means_dim:
        processed_feat = _preprocess_feature_for_cgcd(dino_feat, cgcd_model).float()
    if int(processed_feat.shape[-1]) != means_dim:
        raise RuntimeError(
            "CGCD preprocessing dimension mismatch: "
            f"processed_feat_dim={int(processed_feat.shape[-1])}, means_dim={means_dim}."
        )

    diffs = processed_feat.unsqueeze(1) - cgcd_model.means.float()

    if mode == "mahalanobis_margin":
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
                "mahalanobis_margin requires one of: cgcd_model.inv_covs, cgcd_model.inv_sqrt_covs, or cgcd_model.covs."
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose from clear, contrastive, mahalanobis_margin.")

    if clear_idx < 0 or clear_idx >= dist_sq.shape[1]:
        raise ValueError(f"clear_idx out of range: clear_idx={clear_idx}, num_classes={dist_sq.shape[1]}")

    dist_clear = dist_sq[:, clear_idx]
    if dist_sq.shape[1] <= 1:
        return processed_feat, torch.sigmoid(torch.zeros_like(dist_clear))

    mask = torch.ones(dist_sq.shape[1], device=dist_sq.device, dtype=torch.bool)
    mask[clear_idx] = False
    dist_neg = dist_sq[:, mask]
    tau = max(float(contrastive_tau), 1e-6)
    if dist_neg.shape[1] == 0:
        agg_dist_neg = torch.zeros_like(dist_clear)
    else:
        agg_dist_neg = -tau * torch.logsumexp(-dist_neg / tau, dim=-1)
    margin = agg_dist_neg - dist_clear
    score = torch.sigmoid(margin / max(float(mahalanobis_temp), 1e-6))
    return processed_feat, score.float()


def compute_cgcd_unsupervised_loss(
    restored,
    pseudo,
    degraded,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    mode="mahalanobis_margin",
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
    mahalanobis_temp=20.0,
    feature_weight=1.0,
    score_weight=1.0,
    score_mode="pseudo",
    rank_weight=1.0,
    triplet_weight=1.0,
    input_margin=0.05,
    triplet_margin=0.2,
    clear_input_threshold=1.1,
    clear_gate_score_type="mahal",
    clear_secondary_threshold=1.1,
    pseudo_gain_margin=-1.0,
):
    """
    CGCD-aware unsupervised loss.

    - Pull restored toward pseudo in CGCD feature/score space.
    - Push restored away from degraded input via score ranking + triplet separation.
    """
    restored_repr, restored_score = compute_cgcd_repr_and_score(
        restored,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        mode=mode,
        amp_dtype=amp_dtype,
        contrastive_pos_weight=contrastive_pos_weight,
        contrastive_neg_weight=contrastive_neg_weight,
        contrastive_tau=contrastive_tau,
        contrastive_score_temp=contrastive_score_temp,
        mahalanobis_temp=mahalanobis_temp,
    )

    with torch.no_grad():
        pseudo_repr, pseudo_score = compute_cgcd_repr_and_score(
            pseudo.detach(),
            dino_model,
            cgcd_model,
            dino_transform,
            clear_idx,
            mode=mode,
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
            mahalanobis_temp=mahalanobis_temp,
        )
        degraded_repr, degraded_score = compute_cgcd_repr_and_score(
            degraded.detach(),
            dino_model,
            cgcd_model,
            dino_transform,
            clear_idx,
            mode=mode,
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
            mahalanobis_temp=mahalanobis_temp,
        )

    loss_feature = F.smooth_l1_loss(restored_repr, pseudo_repr, reduction="none").mean(dim=1)
    score_mode = str(score_mode).strip().lower()
    if score_mode == "pseudo":
        loss_score = F.smooth_l1_loss(restored_score, pseudo_score, reduction="none")
    elif score_mode == "maximize":
        loss_score = 1.0 - restored_score
    else:
        raise ValueError(f"Unsupported CGCD unsupervised score_mode: {score_mode}")

    dist_pos = torch.norm(restored_repr - pseudo_repr, p=2, dim=1)
    dist_neg = torch.norm(restored_repr - degraded_repr, p=2, dim=1)
    loss_triplet = F.relu(dist_pos - dist_neg + float(triplet_margin))
    loss_rank = F.relu((degraded_score + float(input_margin)) - restored_score)

    clear_threshold = float(clear_input_threshold)
    secondary_threshold = float(clear_secondary_threshold)
    gain_margin = float(pseudo_gain_margin)
    gate_mask = torch.ones_like(restored_score, dtype=restored_score.dtype)

    if clear_threshold <= 1.0 or secondary_threshold <= 1.0:
        _, degraded_clear_score = compute_cgcd_repr_and_score(
            degraded.detach(),
            dino_model,
            cgcd_model,
            dino_transform,
            clear_idx,
            mode="clear",
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
            mahalanobis_temp=mahalanobis_temp,
        )
    else:
        degraded_clear_score = torch.zeros_like(degraded_score)

    gate_type = str(clear_gate_score_type).strip().lower()
    if gate_type == "clear_then_mahal":
        primary_clear_like = (
            degraded_clear_score >= clear_threshold
            if clear_threshold <= 1.0
            else torch.zeros_like(degraded_score, dtype=torch.bool)
        )
        secondary_clear_like = (
            (degraded_clear_score < clear_threshold) & (degraded_score >= secondary_threshold)
            if secondary_threshold <= 1.0
            else torch.zeros_like(degraded_score, dtype=torch.bool)
        )
        clear_like_mask = primary_clear_like | secondary_clear_like
    elif gate_type == "clear":
        clear_like_mask = (
            degraded_clear_score >= clear_threshold
            if clear_threshold <= 1.0
            else torch.zeros_like(degraded_score, dtype=torch.bool)
        )
        secondary_clear_like = torch.zeros_like(clear_like_mask)
    elif gate_type == "mahal":
        clear_like_mask = (
            degraded_score >= clear_threshold
            if clear_threshold <= 1.0
            else torch.zeros_like(degraded_score, dtype=torch.bool)
        )
        secondary_clear_like = torch.zeros_like(clear_like_mask)
    else:
        raise ValueError(f"Unsupported CGCD clear gate score type: {clear_gate_score_type}")

    if clear_threshold <= 1.0 or secondary_threshold <= 1.0:
        gate_mask = gate_mask * (~clear_like_mask).to(restored_score.dtype)
    else:
        clear_like_mask = torch.zeros_like(degraded_score, dtype=torch.bool)
        secondary_clear_like = torch.zeros_like(degraded_score, dtype=torch.bool)

    if gain_margin >= 0.0:
        improve_mask = (pseudo_score - degraded_score) > gain_margin
        gate_mask = gate_mask * improve_mask.to(restored_score.dtype)
    else:
        improve_mask = torch.ones_like(restored_score, dtype=torch.bool)

    loss_feature_eff = loss_feature * gate_mask
    loss_score_eff = loss_score * gate_mask
    loss_rank_eff = loss_rank * gate_mask
    loss_triplet_eff = loss_triplet * gate_mask
    dist_pos_eff = dist_pos * gate_mask
    dist_neg_eff = dist_neg * gate_mask

    total = (
        float(feature_weight) * loss_feature_eff.mean()
        + float(score_weight) * loss_score_eff.mean()
        + float(rank_weight) * loss_rank_eff.mean()
        + float(triplet_weight) * loss_triplet_eff.mean()
    )

    stats = {
        "feature": float(loss_feature_eff.mean().detach().item()),
        "score": float(loss_score_eff.mean().detach().item()),
        "rank": float(loss_rank_eff.mean().detach().item()),
        "triplet": float(loss_triplet_eff.mean().detach().item()),
        "restored_score": float(restored_score.mean().detach().item()),
        "pseudo_score": float(pseudo_score.mean().detach().item()),
        "degraded_score": float(degraded_score.mean().detach().item()),
        "degraded_clear_score": float(degraded_clear_score.mean().detach().item()),
        "dist_pos": float(dist_pos_eff.mean().detach().item()),
        "dist_neg": float(dist_neg_eff.mean().detach().item()),
        "gate_active_ratio": float(gate_mask.mean().detach().item()),
        "gate_clear_like_ratio": float(clear_like_mask.to(restored_score.dtype).mean().detach().item()),
        "gate_primary_clear_ratio": (
            float((degraded_clear_score >= clear_threshold).to(restored_score.dtype).mean().detach().item())
            if clear_threshold <= 1.0
            else 0.0
        ),
        "gate_secondary_mahal_ratio": float(secondary_clear_like.to(restored_score.dtype).mean().detach().item()),
        "gate_improve_ratio": float(improve_mask.to(restored_score.dtype).mean().detach().item()),
    }
    return total, stats


class CGCDUnsupervisedLoss(nn.Module):
    def __init__(
        self,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        amp_dtype=None,
        mode="mahalanobis_margin",
        feature_weight=1.0,
        score_weight=1.0,
        score_mode="pseudo",
        rank_weight=1.0,
        triplet_weight=1.0,
        input_margin=0.05,
        triplet_margin=0.2,
        clear_input_threshold=1.1,
        clear_gate_score_type="mahal",
        clear_secondary_threshold=1.1,
        pseudo_gain_margin=-1.0,
        contrastive_pos_weight=1.0,
        contrastive_neg_weight=1.0,
        contrastive_tau=1.0,
        contrastive_score_temp=1.0,
        mahalanobis_temp=20.0,
    ):
        super().__init__()
        # Keep external models as plain refs so CGCDUnsupervisedLoss itself has no trainable parameters.
        object.__setattr__(self, "_dino_model_ref", dino_model)
        object.__setattr__(self, "_cgcd_model_ref", cgcd_model)
        object.__setattr__(self, "_dino_transform_ref", dino_transform)
        self.clear_idx = int(clear_idx)
        self.amp_dtype = amp_dtype
        self.mode = mode
        self.feature_weight = float(feature_weight)
        self.score_weight = float(score_weight)
        self.score_mode = str(score_mode).strip().lower()
        self.rank_weight = float(rank_weight)
        self.triplet_weight = float(triplet_weight)
        self.input_margin = float(input_margin)
        self.triplet_margin = float(triplet_margin)
        self.clear_input_threshold = float(clear_input_threshold)
        self.clear_gate_score_type = str(clear_gate_score_type).strip().lower()
        self.clear_secondary_threshold = float(clear_secondary_threshold)
        self.pseudo_gain_margin = float(pseudo_gain_margin)
        self.contrastive_pos_weight = float(contrastive_pos_weight)
        self.contrastive_neg_weight = float(contrastive_neg_weight)
        self.contrastive_tau = float(contrastive_tau)
        self.contrastive_score_temp = float(contrastive_score_temp)
        self.mahalanobis_temp = float(mahalanobis_temp)

    def forward(self, anchor, positive, negative):
        return compute_cgcd_unsupervised_loss(
            restored=anchor,
            pseudo=positive,
            degraded=negative,
            dino_model=self._dino_model_ref,
            cgcd_model=self._cgcd_model_ref,
            dino_transform=self._dino_transform_ref,
            clear_idx=self.clear_idx,
            mode=self.mode,
            amp_dtype=self.amp_dtype,
            contrastive_pos_weight=self.contrastive_pos_weight,
            contrastive_neg_weight=self.contrastive_neg_weight,
            contrastive_tau=self.contrastive_tau,
            contrastive_score_temp=self.contrastive_score_temp,
            mahalanobis_temp=self.mahalanobis_temp,
            feature_weight=self.feature_weight,
            score_weight=self.score_weight,
            score_mode=self.score_mode,
            rank_weight=self.rank_weight,
            triplet_weight=self.triplet_weight,
            input_margin=self.input_margin,
            triplet_margin=self.triplet_margin,
            clear_input_threshold=self.clear_input_threshold,
            clear_gate_score_type=self.clear_gate_score_type,
            clear_secondary_threshold=self.clear_secondary_threshold,
            pseudo_gain_margin=self.pseudo_gain_margin,
        )


def compute_cgcd_unsupervised_loss_ablation_a(
    restored,
    pseudo,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    mode="mahalanobis_margin",
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
    mahalanobis_temp=20.0,
    feature_weight=1.0,
    score_weight=1.0,
    score_mode="pseudo",
):
    """Ablation A: compute only L_pos = feature pull + score pull."""
    restored_repr, restored_score = compute_cgcd_repr_and_score(
        restored,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        mode=mode,
        amp_dtype=amp_dtype,
        contrastive_pos_weight=contrastive_pos_weight,
        contrastive_neg_weight=contrastive_neg_weight,
        contrastive_tau=contrastive_tau,
        contrastive_score_temp=contrastive_score_temp,
        mahalanobis_temp=mahalanobis_temp,
    )

    with torch.no_grad():
        pseudo_repr, pseudo_score = compute_cgcd_repr_and_score(
            pseudo.detach(),
            dino_model,
            cgcd_model,
            dino_transform,
            clear_idx,
            mode=mode,
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
            mahalanobis_temp=mahalanobis_temp,
        )

    loss_feature = F.smooth_l1_loss(restored_repr, pseudo_repr, reduction="none").mean(dim=1)
    score_mode = str(score_mode).strip().lower()
    if score_mode == "pseudo":
        loss_score = F.smooth_l1_loss(restored_score, pseudo_score, reduction="none")
    elif score_mode == "maximize":
        loss_score = 1.0 - restored_score
    else:
        raise ValueError(f"Unsupported CGCD unsupervised score_mode: {score_mode}")

    total = (float(feature_weight) * loss_feature.mean()) + (float(score_weight) * loss_score.mean())
    zero = 0.0
    stats = {
        "feature": float(loss_feature.mean().detach().item()),
        "score": float(loss_score.mean().detach().item()),
        "rank": zero,
        "triplet": zero,
        "restored_score": float(restored_score.mean().detach().item()),
        "pseudo_score": float(pseudo_score.mean().detach().item()),
        "degraded_score": zero,
        "degraded_clear_score": zero,
        "dist_pos": float(torch.norm(restored_repr - pseudo_repr, p=2, dim=1).mean().detach().item()),
        "dist_neg": zero,
        "gate_active_ratio": 1.0,
        "gate_clear_like_ratio": zero,
        "gate_primary_clear_ratio": zero,
        "gate_secondary_mahal_ratio": zero,
        "gate_improve_ratio": 1.0,
    }
    return total, stats


def compute_cgcd_unsupervised_loss_ablation_c(
    restored,
    pseudo,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    mode="mahalanobis_margin",
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
    mahalanobis_temp=20.0,
    feature_weight=1.0,
):
    """Ablation C: compute only feature pull without score/rank/triplet terms."""
    restored_repr, restored_score = compute_cgcd_repr_and_score(
        restored,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        mode=mode,
        amp_dtype=amp_dtype,
        contrastive_pos_weight=contrastive_pos_weight,
        contrastive_neg_weight=contrastive_neg_weight,
        contrastive_tau=contrastive_tau,
        contrastive_score_temp=contrastive_score_temp,
        mahalanobis_temp=mahalanobis_temp,
    )

    with torch.no_grad():
        pseudo_repr, pseudo_score = compute_cgcd_repr_and_score(
            pseudo.detach(),
            dino_model,
            cgcd_model,
            dino_transform,
            clear_idx,
            mode=mode,
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
            mahalanobis_temp=mahalanobis_temp,
        )

    loss_feature = F.smooth_l1_loss(restored_repr, pseudo_repr, reduction="none").mean(dim=1)
    total = float(feature_weight) * loss_feature.mean()
    zero = 0.0
    stats = {
        "feature": float(loss_feature.mean().detach().item()),
        "score": zero,
        "rank": zero,
        "triplet": zero,
        "restored_score": float(restored_score.mean().detach().item()),
        "pseudo_score": float(pseudo_score.mean().detach().item()),
        "degraded_score": zero,
        "degraded_clear_score": zero,
        "dist_pos": float(torch.norm(restored_repr - pseudo_repr, p=2, dim=1).mean().detach().item()),
        "dist_neg": zero,
        "gate_active_ratio": 1.0,
        "gate_clear_like_ratio": zero,
        "gate_primary_clear_ratio": zero,
        "gate_secondary_mahal_ratio": zero,
        "gate_improve_ratio": 1.0,
    }
    return total, stats


def compute_cgcd_unsupervised_loss_ablation_b(
    restored,
    pseudo,
    degraded,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    mode="mahalanobis_margin",
    amp_dtype=None,
    contrastive_pos_weight=1.0,
    contrastive_neg_weight=1.0,
    contrastive_tau=1.0,
    contrastive_score_temp=1.0,
    mahalanobis_temp=20.0,
    rank_weight=1.0,
    triplet_weight=1.0,
    input_margin=0.05,
    triplet_margin=0.2,
    clear_input_threshold=1.1,
    clear_gate_score_type="mahal",
    clear_secondary_threshold=1.1,
    pseudo_gain_margin=-1.0,
):
    """Ablation B: compute only R_neg = rank + triplet regularizer."""
    restored_repr, restored_score = compute_cgcd_repr_and_score(
        restored,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        mode=mode,
        amp_dtype=amp_dtype,
        contrastive_pos_weight=contrastive_pos_weight,
        contrastive_neg_weight=contrastive_neg_weight,
        contrastive_tau=contrastive_tau,
        contrastive_score_temp=contrastive_score_temp,
        mahalanobis_temp=mahalanobis_temp,
    )

    with torch.no_grad():
        pseudo_repr, pseudo_score = compute_cgcd_repr_and_score(
            pseudo.detach(),
            dino_model,
            cgcd_model,
            dino_transform,
            clear_idx,
            mode=mode,
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
            mahalanobis_temp=mahalanobis_temp,
        )
        degraded_repr, degraded_score = compute_cgcd_repr_and_score(
            degraded.detach(),
            dino_model,
            cgcd_model,
            dino_transform,
            clear_idx,
            mode=mode,
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
            mahalanobis_temp=mahalanobis_temp,
        )

    dist_pos = torch.norm(restored_repr - pseudo_repr, p=2, dim=1)
    dist_neg = torch.norm(restored_repr - degraded_repr, p=2, dim=1)
    loss_triplet = F.relu(dist_pos - dist_neg + float(triplet_margin))
    loss_rank = F.relu((degraded_score + float(input_margin)) - restored_score)

    clear_threshold = float(clear_input_threshold)
    secondary_threshold = float(clear_secondary_threshold)
    gain_margin = float(pseudo_gain_margin)
    gate_mask = torch.ones_like(restored_score, dtype=restored_score.dtype)

    if clear_threshold <= 1.0 or secondary_threshold <= 1.0:
        _, degraded_clear_score = compute_cgcd_repr_and_score(
            degraded.detach(),
            dino_model,
            cgcd_model,
            dino_transform,
            clear_idx,
            mode="clear",
            amp_dtype=amp_dtype,
            contrastive_pos_weight=contrastive_pos_weight,
            contrastive_neg_weight=contrastive_neg_weight,
            contrastive_tau=contrastive_tau,
            contrastive_score_temp=contrastive_score_temp,
            mahalanobis_temp=mahalanobis_temp,
        )
    else:
        degraded_clear_score = torch.zeros_like(degraded_score)

    gate_type = str(clear_gate_score_type).strip().lower()
    if gate_type == "clear_then_mahal":
        primary_clear_like = (
            degraded_clear_score >= clear_threshold
            if clear_threshold <= 1.0
            else torch.zeros_like(degraded_score, dtype=torch.bool)
        )
        secondary_clear_like = (
            (degraded_clear_score < clear_threshold) & (degraded_score >= secondary_threshold)
            if secondary_threshold <= 1.0
            else torch.zeros_like(degraded_score, dtype=torch.bool)
        )
        clear_like_mask = primary_clear_like | secondary_clear_like
    elif gate_type == "clear":
        clear_like_mask = (
            degraded_clear_score >= clear_threshold
            if clear_threshold <= 1.0
            else torch.zeros_like(degraded_score, dtype=torch.bool)
        )
        secondary_clear_like = torch.zeros_like(clear_like_mask)
    elif gate_type == "mahal":
        clear_like_mask = (
            degraded_score >= clear_threshold
            if clear_threshold <= 1.0
            else torch.zeros_like(degraded_score, dtype=torch.bool)
        )
        secondary_clear_like = torch.zeros_like(clear_like_mask)
    else:
        raise ValueError(f"Unsupported CGCD clear gate score type: {clear_gate_score_type}")

    if clear_threshold <= 1.0 or secondary_threshold <= 1.0:
        gate_mask = gate_mask * (~clear_like_mask).to(restored_score.dtype)
    else:
        clear_like_mask = torch.zeros_like(degraded_score, dtype=torch.bool)
        secondary_clear_like = torch.zeros_like(degraded_score, dtype=torch.bool)

    if gain_margin >= 0.0:
        improve_mask = (pseudo_score - degraded_score) > gain_margin
        gate_mask = gate_mask * improve_mask.to(restored_score.dtype)
    else:
        improve_mask = torch.ones_like(restored_score, dtype=torch.bool)

    loss_rank_eff = loss_rank * gate_mask
    loss_triplet_eff = loss_triplet * gate_mask
    dist_pos_eff = dist_pos * gate_mask
    dist_neg_eff = dist_neg * gate_mask

    total = (float(rank_weight) * loss_rank_eff.mean()) + (float(triplet_weight) * loss_triplet_eff.mean())
    zero = 0.0
    stats = {
        "feature": zero,
        "score": zero,
        "rank": float(loss_rank_eff.mean().detach().item()),
        "triplet": float(loss_triplet_eff.mean().detach().item()),
        "restored_score": float(restored_score.mean().detach().item()),
        "pseudo_score": float(pseudo_score.mean().detach().item()),
        "degraded_score": float(degraded_score.mean().detach().item()),
        "degraded_clear_score": float(degraded_clear_score.mean().detach().item()),
        "dist_pos": float(dist_pos_eff.mean().detach().item()),
        "dist_neg": float(dist_neg_eff.mean().detach().item()),
        "gate_active_ratio": float(gate_mask.mean().detach().item()),
        "gate_clear_like_ratio": float(clear_like_mask.to(restored_score.dtype).mean().detach().item()),
        "gate_primary_clear_ratio": (
            float((degraded_clear_score >= clear_threshold).to(restored_score.dtype).mean().detach().item())
            if clear_threshold <= 1.0
            else 0.0
        ),
        "gate_secondary_mahal_ratio": float(secondary_clear_like.to(restored_score.dtype).mean().detach().item()),
        "gate_improve_ratio": float(improve_mask.to(restored_score.dtype).mean().detach().item()),
    }
    return total, stats


class CGCDUnsupervisedLossAblationA(nn.Module):
    def __init__(
        self,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        amp_dtype=None,
        mode="mahalanobis_margin",
        feature_weight=1.0,
        score_weight=1.0,
        score_mode="pseudo",
        contrastive_pos_weight=1.0,
        contrastive_neg_weight=1.0,
        contrastive_tau=1.0,
        contrastive_score_temp=1.0,
        mahalanobis_temp=20.0,
    ):
        super().__init__()
        object.__setattr__(self, "_dino_model_ref", dino_model)
        object.__setattr__(self, "_cgcd_model_ref", cgcd_model)
        object.__setattr__(self, "_dino_transform_ref", dino_transform)
        self.clear_idx = int(clear_idx)
        self.amp_dtype = amp_dtype
        self.mode = mode
        self.feature_weight = float(feature_weight)
        self.score_weight = float(score_weight)
        self.score_mode = str(score_mode).strip().lower()
        self.contrastive_pos_weight = float(contrastive_pos_weight)
        self.contrastive_neg_weight = float(contrastive_neg_weight)
        self.contrastive_tau = float(contrastive_tau)
        self.contrastive_score_temp = float(contrastive_score_temp)
        self.mahalanobis_temp = float(mahalanobis_temp)

    def forward(self, anchor, positive, negative):
        return compute_cgcd_unsupervised_loss_ablation_a(
            restored=anchor,
            pseudo=positive,
            dino_model=self._dino_model_ref,
            cgcd_model=self._cgcd_model_ref,
            dino_transform=self._dino_transform_ref,
            clear_idx=self.clear_idx,
            mode=self.mode,
            amp_dtype=self.amp_dtype,
            contrastive_pos_weight=self.contrastive_pos_weight,
            contrastive_neg_weight=self.contrastive_neg_weight,
            contrastive_tau=self.contrastive_tau,
            contrastive_score_temp=self.contrastive_score_temp,
            mahalanobis_temp=self.mahalanobis_temp,
            feature_weight=self.feature_weight,
            score_weight=self.score_weight,
            score_mode=self.score_mode,
        )


class CGCDUnsupervisedLossAblationB(nn.Module):
    def __init__(
        self,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        amp_dtype=None,
        mode="mahalanobis_margin",
        rank_weight=1.0,
        triplet_weight=1.0,
        input_margin=0.05,
        triplet_margin=0.2,
        clear_input_threshold=1.1,
        clear_gate_score_type="mahal",
        clear_secondary_threshold=1.1,
        pseudo_gain_margin=-1.0,
        contrastive_pos_weight=1.0,
        contrastive_neg_weight=1.0,
        contrastive_tau=1.0,
        contrastive_score_temp=1.0,
        mahalanobis_temp=20.0,
    ):
        super().__init__()
        object.__setattr__(self, "_dino_model_ref", dino_model)
        object.__setattr__(self, "_cgcd_model_ref", cgcd_model)
        object.__setattr__(self, "_dino_transform_ref", dino_transform)
        self.clear_idx = int(clear_idx)
        self.amp_dtype = amp_dtype
        self.mode = mode
        self.rank_weight = float(rank_weight)
        self.triplet_weight = float(triplet_weight)
        self.input_margin = float(input_margin)
        self.triplet_margin = float(triplet_margin)
        self.clear_input_threshold = float(clear_input_threshold)
        self.clear_gate_score_type = str(clear_gate_score_type).strip().lower()
        self.clear_secondary_threshold = float(clear_secondary_threshold)
        self.pseudo_gain_margin = float(pseudo_gain_margin)
        self.contrastive_pos_weight = float(contrastive_pos_weight)
        self.contrastive_neg_weight = float(contrastive_neg_weight)
        self.contrastive_tau = float(contrastive_tau)
        self.contrastive_score_temp = float(contrastive_score_temp)
        self.mahalanobis_temp = float(mahalanobis_temp)

    def forward(self, anchor, positive, negative):
        return compute_cgcd_unsupervised_loss_ablation_b(
            restored=anchor,
            pseudo=positive,
            degraded=negative,
            dino_model=self._dino_model_ref,
            cgcd_model=self._cgcd_model_ref,
            dino_transform=self._dino_transform_ref,
            clear_idx=self.clear_idx,
            mode=self.mode,
            amp_dtype=self.amp_dtype,
            contrastive_pos_weight=self.contrastive_pos_weight,
            contrastive_neg_weight=self.contrastive_neg_weight,
            contrastive_tau=self.contrastive_tau,
            contrastive_score_temp=self.contrastive_score_temp,
            mahalanobis_temp=self.mahalanobis_temp,
            rank_weight=self.rank_weight,
            triplet_weight=self.triplet_weight,
            input_margin=self.input_margin,
            triplet_margin=self.triplet_margin,
            clear_input_threshold=self.clear_input_threshold,
            clear_gate_score_type=self.clear_gate_score_type,
            clear_secondary_threshold=self.clear_secondary_threshold,
            pseudo_gain_margin=self.pseudo_gain_margin,
        )


class CGCDUnsupervisedLossAblationC(nn.Module):
    def __init__(
        self,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        amp_dtype=None,
        mode="mahalanobis_margin",
        feature_weight=1.0,
        contrastive_pos_weight=1.0,
        contrastive_neg_weight=1.0,
        contrastive_tau=1.0,
        contrastive_score_temp=1.0,
        mahalanobis_temp=20.0,
    ):
        super().__init__()
        object.__setattr__(self, "_dino_model_ref", dino_model)
        object.__setattr__(self, "_cgcd_model_ref", cgcd_model)
        object.__setattr__(self, "_dino_transform_ref", dino_transform)
        self.clear_idx = int(clear_idx)
        self.amp_dtype = amp_dtype
        self.mode = mode
        self.feature_weight = float(feature_weight)
        self.contrastive_pos_weight = float(contrastive_pos_weight)
        self.contrastive_neg_weight = float(contrastive_neg_weight)
        self.contrastive_tau = float(contrastive_tau)
        self.contrastive_score_temp = float(contrastive_score_temp)
        self.mahalanobis_temp = float(mahalanobis_temp)

    def forward(self, anchor, positive, negative):
        return compute_cgcd_unsupervised_loss_ablation_c(
            restored=anchor,
            pseudo=positive,
            dino_model=self._dino_model_ref,
            cgcd_model=self._cgcd_model_ref,
            dino_transform=self._dino_transform_ref,
            clear_idx=self.clear_idx,
            mode=self.mode,
            amp_dtype=self.amp_dtype,
            contrastive_pos_weight=self.contrastive_pos_weight,
            contrastive_neg_weight=self.contrastive_neg_weight,
            contrastive_tau=self.contrastive_tau,
            contrastive_score_temp=self.contrastive_score_temp,
            mahalanobis_temp=self.mahalanobis_temp,
            feature_weight=self.feature_weight,
        )


class CGCDLoss(nn.Module):
    def __init__(
        self,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        amp_dtype=None,
        metric_name="clear_mahal_contrastive_margin",
        contrastive_tau=5.0,
        clear_weight=1.0,
        neg_weight=1.0,
        margin_scale=1.0,
        margin_bias=0.0,
        eps=1e-8,
        reduction="mean",
    ):
        super().__init__()
        # Keep external models as plain refs so CGCDLoss itself has no trainable parameters.
        object.__setattr__(self, "_dino_model_ref", dino_model)
        object.__setattr__(self, "_cgcd_model_ref", cgcd_model)
        object.__setattr__(self, "_dino_transform_ref", dino_transform)
        self.clear_idx = int(clear_idx)
        self.amp_dtype = amp_dtype
        self.metric_name = str(metric_name).strip()
        self.contrastive_tau = float(contrastive_tau)
        self.clear_weight = float(clear_weight)
        self.neg_weight = float(neg_weight)
        self.margin_scale = float(margin_scale)
        self.margin_bias = float(margin_bias)
        self.eps = float(eps)
        self.reduction = str(reduction).strip().lower()
        valid_metrics = {
            "clear_mahal_margin",
            "clear_mahal_mean_margin",
            "clear_mahal_contrastive_margin",
        }
        if self.metric_name not in valid_metrics:
            raise ValueError(f"Unsupported CGCDLoss metric_name: {metric_name}. Choose from {sorted(valid_metrics)}")
        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported CGCDLoss reduction: {reduction}")

    def forward(self, image):
        _, metric_dict = compute_cgcd_clear_mahal_metrics(
            image_list=image,
            dino_model=self._dino_model_ref,
            cgcd_model=self._cgcd_model_ref,
            dino_transform=self._dino_transform_ref,
            clear_idx=self.clear_idx,
            amp_dtype=self.amp_dtype,
            contrastive_tau=self.contrastive_tau,
            clear_weight=self.clear_weight,
            neg_weight=self.neg_weight,
            eps=self.eps,
        )

        margin = metric_dict[self.metric_name]
        #! [debug] Keep the pre-softplus argument for tuning margin_scale/margin_bias during CGCD loss calibration.
        softplus_input = self.margin_bias - (self.margin_scale * margin)
        loss_vec = F.softplus(softplus_input)

        if self.reduction == "mean":
            loss = loss_vec.mean()
        elif self.reduction == "sum":
            loss = loss_vec.sum()
        else:
            loss = loss_vec

        stats = {
            "loss": float(loss_vec.mean().detach().item()),
            "metric": float(margin.mean().detach().item()),
            #! [debug] Selected margin distribution helps tune tau, clear/neg weights, and margin scaling.
            "selected_margin_std": float(margin.std(unbiased=False).detach().item()),
            "selected_margin_min": float(margin.min().detach().item()),
            "selected_margin_max": float(margin.max().detach().item()),
            "softplus_input_mean": float(softplus_input.mean().detach().item()),
            "softplus_input_std": float(softplus_input.std(unbiased=False).detach().item()),
            "softplus_input_min": float(softplus_input.min().detach().item()),
            "softplus_input_max": float(softplus_input.max().detach().item()),
            "clear_mahal_dist": float(metric_dict["clear_mahal_dist"].mean().detach().item()),
            "min_neg_mahal_dist": float(metric_dict["min_neg_mahal_dist"].mean().detach().item()),
            "mean_neg_mahal_dist": float(metric_dict["mean_neg_mahal_dist"].mean().detach().item()),
            "contrastive_neg_mahal_dist": float(metric_dict["contrastive_neg_mahal_dist"].mean().detach().item()),
            "clear_mahal_margin": float(metric_dict["clear_mahal_margin"].mean().detach().item()),
            "clear_mahal_mean_margin": float(metric_dict["clear_mahal_mean_margin"].mean().detach().item()),
            "clear_mahal_contrastive_margin": float(
                metric_dict["clear_mahal_contrastive_margin"].mean().detach().item()
            ),
        }
        return loss, stats
