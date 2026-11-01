"""Discovery- and instance-conditioned descriptor (Sec. 3.2).

Given a DINO feature z, the descriptor uses the VB-CGCD clusters (mu_c, Sigma_c) to compute

    s_c(z)    = softmax_c(l_c(z))                       soft assignment from the log-likelihoods
    f_base(z) = sum_c s_c(z) E_c                         E initialized from the cluster means
    r_c(z)    = Sigma_c^{-1/2} (z - mu_c)                covariance-normalized residual
    f_inst(z) = sum_c s_c(z) r_c(z)
    v(z)      = f_base(z) + W_res f_inst(z)               W_res is zero-initialized

and projects v to the OneRestore query dimension. Attribute names are kept from the released
checkpoints: ``class_embeddings`` is E and ``residual_proj`` is W_res.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_vbcgcd_clusters(cluster_dir, stage):
    """Cluster means and covariances saved by VB-CGCD after ``stage``."""
    means = np.load(os.path.join(cluster_dir, f"stage{stage}class_means.npy"))
    covs = np.load(os.path.join(cluster_dir, f"stage{stage}class_covariances.npy"))
    return means, covs


def load_pca_layer(pca_path):
    """Frozen linear layer equivalent to the VB-CGCD PCA projection."""
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    components = torch.from_numpy(pca.components_).float()
    mean = torch.from_numpy(pca.mean_).float()
    layer = nn.Linear(components.shape[1], components.shape[0])
    layer.weight.data = components
    layer.bias.data = -torch.matmul(components, mean)
    for param in layer.parameters():
        param.requires_grad = False
    return layer


def load_scaler(scaler_path, expected_dim):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    if mean.shape[0] != expected_dim:
        raise ValueError(f"Scaler dimension mismatch at {scaler_path}: expected {expected_dim}, got {mean.shape[0]}")
    return torch.from_numpy(mean), torch.from_numpy(scale)


class DiscoveryInstanceDescriptor(nn.Module):
    def __init__(self, cluster_dir, output_dim=324, stage=1, pca_path=None, scaler_path=None):
        super().__init__()
        means_np, covs_np = load_vbcgcd_clusters(cluster_dir, stage)
        self.pca_layer = load_pca_layer(pca_path)
        scaler_mean, scaler_scale = load_scaler(
            scaler_path or os.path.join(cluster_dir, "scaler_model.pkl"), self.pca_layer.out_features
        )
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

        self.num_classes = means_np.shape[0]
        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))  # mu_c
        self.register_buffer("covs", torch.from_numpy(covs_np).float())  # Sigma_c

        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())  # E, initialized from mu_c
        self.projection = nn.Linear(self.class_embeddings.shape[1], output_dim)

        feature_dim = self.covs.shape[-1]
        covs_stable = self.covs + torch.eye(feature_dim, device=self.covs.device) * 1e-6
        self.register_buffer("log_2pi_term", feature_dim * torch.log(torch.tensor(2 * torch.pi)))
        eigvals, eigvecs = torch.linalg.eigh(covs_stable)
        eigvals = eigvals.clamp(min=1e-8)
        self.register_buffer("log_det", torch.log(eigvals).sum(dim=-1))
        inv_sqrt_covs = torch.bmm(eigvecs, torch.bmm(torch.diag_embed(1.0 / torch.sqrt(eigvals)), eigvecs.mT))
        self.register_buffer("inv_sqrt_covs", inv_sqrt_covs)  # Sigma_c^{-1/2}

        self.residual_proj = nn.Linear(feature_dim, feature_dim)  # W_res
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def preprocess(self, feature):
        """PCA and standardization applied by VB-CGCD before clustering."""
        feature = self.pca_layer(feature)
        if self.scaler_mean.numel() > 0 and self.scaler_scale.numel() > 0:
            feature = (feature - self.scaler_mean) / self.scaler_scale
        return feature

    def forward(self, feature):
        """Returns the normalized descriptor and the per-cluster log-likelihoods l_c(z)."""
        z = self.preprocess(feature)
        residual = torch.einsum("cij,bcj->bci", self.inv_sqrt_covs, z.unsqueeze(1) - self.means)  # r_c(z)
        mahalanobis_sq = torch.sum(residual * residual, dim=-1)
        log_likelihood = -0.5 * (mahalanobis_sq + self.log_det + self.log_2pi_term)  # l_c(z)

        assignment = F.softmax(log_likelihood, dim=-1)  # s_c(z)
        f_base = torch.matmul(assignment, self.class_embeddings)
        f_inst = torch.einsum("bc,bci->bi", assignment, residual)
        v = f_base + self.residual_proj(f_inst)

        embedding = F.normalize(self.projection(v), p=2, dim=1)
        return embedding, log_likelihood
