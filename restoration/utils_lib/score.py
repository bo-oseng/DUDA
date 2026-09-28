import torch

from utils_lib.loss import compute_cgcd_clear_mahal_distances


ANCHOR_SCORE_MODE_ALIASES = {
    "anchor",
    "anchor_margin",
    "soft_anchor",
    "soft_anchor_margin",
}


def is_cgcd_anchor_mode(mode):
    token = str(mode).strip().lower().replace("-", "_")
    return token in ANCHOR_SCORE_MODE_ALIASES


def _resolve_cgcd_anchor_terms(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    amp_dtype=None,
    clear_weight=1.0,
    neg_weight=1.0,
    softmin_tau=0.0,
):
    softmin_tau = float(softmin_tau if softmin_tau is not None else 0.0)
    _, metric_dict = compute_cgcd_clear_mahal_distances(
        image_list=image_list,
        dino_model=dino_model,
        cgcd_model=cgcd_model,
        dino_transform=dino_transform,
        clear_idx=clear_idx,
        amp_dtype=amp_dtype,
        contrastive_tau=max(softmin_tau, 1e-6),
    )

    clear_dist = metric_dict["clear_mahal_dist"]
    if softmin_tau > 0.0:
        neg_dist = metric_dict["contrastive_neg_mahal_dist"]
    else:
        neg_dist = metric_dict["min_neg_mahal_dist"]

    neg_term = float(neg_weight) * neg_dist
    clear_term = float(clear_weight) * clear_dist
    return neg_term.float(), clear_term.float()


@torch.no_grad()
def get_cgcd_anchor_score(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    amp_dtype=None,
    clear_weight=1.0,
    neg_weight=1.0,
    softmin_tau=0.0,
):
    neg_term, clear_term = _resolve_cgcd_anchor_terms(
        image_list=image_list,
        dino_model=dino_model,
        cgcd_model=cgcd_model,
        dino_transform=dino_transform,
        clear_idx=clear_idx,
        amp_dtype=amp_dtype,
        clear_weight=clear_weight,
        neg_weight=neg_weight,
        softmin_tau=softmin_tau,
    )
    denom = (neg_term + clear_term).clamp_min(1e-8)
    return (neg_term / denom).float()


@torch.no_grad()
def get_cgcd_anchor_score_sharpen(
    image_list,
    dino_model,
    cgcd_model,
    dino_transform,
    clear_idx,
    amp_dtype=None,
    clear_weight=1.0,
    neg_weight=1.0,
    softmin_tau=0.0,
    score_temp=0.25,
):
    neg_term, clear_term = _resolve_cgcd_anchor_terms(
        image_list=image_list,
        dino_model=dino_model,
        cgcd_model=cgcd_model,
        dino_transform=dino_transform,
        clear_idx=clear_idx,
        amp_dtype=amp_dtype,
        clear_weight=clear_weight,
        neg_weight=neg_weight,
        softmin_tau=softmin_tau,
    )
    eps = 1e-8
    score_temp = max(float(score_temp), 1e-6)
    log_ratio = torch.log(neg_term.clamp_min(eps)) - torch.log(clear_term.clamp_min(eps))
    return torch.sigmoid(log_ratio / score_temp).float()
