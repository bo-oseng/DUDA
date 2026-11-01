"""Discovery-adaptive score S_DA (Sec. 3.3).

For the DINO feature z of a candidate pseudo-target,

    d_neg(z) = min_{c in C_deg} d_c(z)
    S_DA(z)  = d_neg(z) / (d_neg(z) + d_clear(z))

where d_c is the Mahalanobis distance to cluster c, C_deg are the known and discovered
degradation clusters, and d_clear is the distance to the anchored clear cluster.
A larger S_DA means the candidate is closer to clear and farther from every degradation.
"""

import torch
from torch.amp import autocast


@torch.no_grad()
def mahalanobis_distances(images, dino_model, descriptor, dino_transform, amp_dtype=None):
    """Mahalanobis distances d_c(z) of each image to every VB-CGCD cluster, shape [B, C]."""
    images = torch.clamp(images, 0.0, 1.0)
    with autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and images.is_cuda):
        z = descriptor.preprocess(dino_model(dino_transform(images)).pooler_output)
    z = z.float()
    diffs = z.unsqueeze(1) - descriptor.means.float().to(z.device)
    whitened = torch.einsum("cij,bcj->bci", descriptor.inv_sqrt_covs.float().to(z.device), diffs)
    dist_sq = torch.sum(whitened * whitened, dim=-1).clamp_min(0.0)
    return torch.sqrt(dist_sq.clamp_min(0.0))


@torch.no_grad()
def discovery_adaptive_score(images, dino_model, descriptor, dino_transform, clear_idx, amp_dtype=None):
    """S_DA for a batch of images, shape [B]."""
    distances = mahalanobis_distances(images, dino_model, descriptor, dino_transform, amp_dtype)
    d_clear = distances[:, clear_idx]
    degradation = torch.ones(distances.shape[1], device=distances.device, dtype=torch.bool)
    degradation[clear_idx] = False
    d_neg = distances[:, degradation].min(dim=-1).values
    d_neg, d_clear = d_neg.float(), d_clear.float()
    return (d_neg / (d_neg + d_clear).clamp_min(1e-8)).float()
