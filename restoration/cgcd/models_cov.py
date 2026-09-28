import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_pca_layer(pca_path):
    if pca_path and os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca_obj = pickle.load(f)
        components = torch.from_numpy(pca_obj.components_).float()
        mean = torch.from_numpy(pca_obj.mean_).float()

        pca_layer = nn.Linear(components.shape[1], components.shape[0])
        pca_layer.weight.data = components
        pca_layer.bias.data = -torch.matmul(components, mean)
        for param in pca_layer.parameters():
            param.requires_grad = False
        return pca_layer
    raise FileNotFoundError(f"PCA model not found at {pca_path}")


def _load_scaler_tensors(saved_models_dir, scaler_path=None, expected_dim=None):
    explicit_scaler_path = scaler_path is not None
    scaler_path = scaler_path or os.path.join(saved_models_dir, "scaler_model.pkl")

    if not os.path.exists(scaler_path):
        if explicit_scaler_path:
            raise FileNotFoundError(f"Scaler model not found at {scaler_path}")
        return None, None

    with open(scaler_path, "rb") as f:
        scaler_obj = pickle.load(f)

    if not hasattr(scaler_obj, "mean_") or not hasattr(scaler_obj, "scale_"):
        raise ValueError(f"Invalid scaler object at {scaler_path}: missing mean_/scale_")

    scaler_mean = np.asarray(scaler_obj.mean_, dtype=np.float32)
    scaler_scale = np.asarray(scaler_obj.scale_, dtype=np.float32)
    scaler_scale = np.where(np.abs(scaler_scale) < 1e-12, 1.0, scaler_scale)

    if expected_dim is not None and scaler_mean.shape[0] != expected_dim:
        raise ValueError(
            f"Scaler dimension mismatch at {scaler_path}: expected {expected_dim}, got {scaler_mean.shape[0]}"
        )

    return torch.from_numpy(scaler_mean), torch.from_numpy(scaler_scale)


def _register_scaler_buffers(module, scaler_mean, scaler_scale):
    if scaler_mean is None or scaler_scale is None:
        scaler_mean = torch.empty(0, dtype=torch.float32)
        scaler_scale = torch.empty(0, dtype=torch.float32)
    module.register_buffer("scaler_mean", scaler_mean)
    module.register_buffer("scaler_scale", scaler_scale)


def _preprocess_feature(feature, pca_layer, scaler_mean, scaler_scale):
    feature = pca_layer(feature)
    if scaler_mean.numel() > 0 and scaler_scale.numel() > 0:
        feature = (feature - scaler_mean) / scaler_scale
    return feature


def _unpack_feature_inputs(feature):
    patch_summary = None
    if isinstance(feature, dict):
        global_feature = feature.get("global_feature")
        patch_summary = feature.get("patch_summary")
    elif isinstance(feature, (tuple, list)):
        if len(feature) == 0:
            raise ValueError("Feature input tuple/list must not be empty.")
        global_feature = feature[0]
        if len(feature) > 1:
            patch_summary = feature[1]
    else:
        global_feature = feature

    if global_feature is None:
        raise ValueError("Global feature is required for CGCD forward.")
    return global_feature, patch_summary


def load_cgcd_stageN_parameters(saved_models_dir, stage, pca_path, scaler_path=None):
    means_path = os.path.join(saved_models_dir, f"stage{stage}class_means.npy")
    covs_path = os.path.join(saved_models_dir, f"stage{stage}class_covariances.npy")

    if os.path.exists(means_path):
        means_np = np.load(means_path)
        covs_np = np.load(covs_path)
    else:
        raise FileNotFoundError(f"Gaussian parameters not found in {saved_models_dir}")

    pca_layer = _load_pca_layer(pca_path)
    scaler_mean, scaler_scale = _load_scaler_tensors(
        saved_models_dir=saved_models_dir,
        scaler_path=scaler_path,
        expected_dim=pca_layer.out_features,
    )
    return means_np, covs_np, pca_layer, scaler_mean, scaler_scale


class CGCDSignalModuleSoft(nn.Module):
    """
    Covariance-aware descriptor:
    desc = MLP([prototype_feature ; r_bar ; margin ; entropy])
    r_bar = sum_c p_c * |D_c^{-1/2}(x - mu_c)|, where D_c = diag(Sigma_c)
    """

    def __init__(self, saved_models_dir, output_dim=256, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        feature_dim = means_np.shape[1]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))  # [1, C, D]
        self.register_buffer("covs", torch.from_numpy(covs_np).float())  # [C, D, D]

        # Learnable class embeddings initialized from class means
        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())  # [C, D]

        # Mahalanobis terms (full covariance for logits)
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        stabilized_covs = self.covs + eye_matrix

        self.register_buffer("inv_covs", torch.linalg.inv(stabilized_covs))

        self.register_buffer("log_det", torch.logdet(stabilized_covs))

        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

        # Diagonal covariance whitening for residual descriptor
        diag_cov = torch.diagonal(stabilized_covs, dim1=-2, dim2=-1)  # [C, D]
        self.register_buffer("diag_inv_sqrt_cov", torch.rsqrt(torch.clamp(diag_cov, min=1e-6)))

        # Descriptor MLP for [prototype ; r_bar ; margin ; entropy]
        descriptor_in_dim = 2 * feature_dim + 2
        hidden_dim = feature_dim
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(descriptor_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)  # [B, D]

        # Full-cov Mahalanobis logits
        diff = feature.unsqueeze(1) - self.means  # [B, C, D]
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)  # [B, C]
        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
        similarities = F.softmax(logits, dim=-1)  # [B, C]

        # Mean/prototype feature
        prototype_feature = torch.matmul(similarities, self.class_embeddings)  # [B, D]

        # Covariance-aware residual feature using diag whitening
        whitened_residual = torch.abs(diff * self.diag_inv_sqrt_cov.unsqueeze(0))  # [B, C, D]
        r_bar = torch.sum(similarities.unsqueeze(-1) * whitened_residual, dim=1)  # [B, D]

        # Confidence cues from class posterior
        if self.num_classes > 1:
            top2_vals, _ = torch.topk(logits, k=2, dim=-1)
            margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(-1)  # [B, 1]
        else:
            margin = torch.zeros((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
        entropy = -(similarities * torch.log(similarities.clamp_min(1e-12))).sum(dim=-1, keepdim=True)  # [B, 1]

        descriptor = torch.cat([prototype_feature, r_bar, margin, entropy], dim=-1)  # [B, 2D+2]
        embedding = self.descriptor_mlp(descriptor)  # [B, output_dim]
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


# class _CGCDSignalModuleCovBase(nn.Module):
#     """Shared backbone for covariance-decomposed descriptor variants."""

#     def __init__(self, saved_models_dir, stage=0, pca_path=None, scaler_path=None):
#         super().__init__()

#         means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
#             saved_models_dir, stage, pca_path, scaler_path
#         )
#         _register_scaler_buffers(self, scaler_mean, scaler_scale)

#         feature_dim = means_np.shape[1]
#         self.num_classes = int(means_np.shape[0])
#         self.feature_dim = int(feature_dim)

#         self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))  # [1, C, D]
#         self.register_buffer("covs", torch.from_numpy(covs_np).float())  # [C, D, D]

#         # Keep same learnable prototype mechanism as original Soft module.
#         self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())  # [C, D]

#         eps = 1e-6
#         eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
#         stabilized_covs = self.covs + eye_matrix

#         self.register_buffer("inv_covs", torch.linalg.inv(stabilized_covs))
#         self.register_buffer("log_det", torch.logdet(stabilized_covs))
#         self.register_buffer(
#             "log_2pi_term",
#             torch.tensor(float(feature_dim) * np.log(2.0 * np.pi), dtype=torch.float32),
#         )

#         diag_cov = torch.diagonal(stabilized_covs, dim1=-2, dim2=-1)  # [C, D]
#         self.register_buffer("diag_inv_sqrt_cov", torch.rsqrt(torch.clamp(diag_cov, min=1e-6)))
#         self.register_buffer("diag_trace", diag_cov.sum(dim=-1, keepdim=True))  # [C, 1]

#         eigvals = torch.linalg.eigvalsh(stabilized_covs)  # [C, D] ascending
#         self.register_buffer("eigvals", torch.clamp(eigvals, min=1e-12))

#     @staticmethod
#     def _normalize_per_class_stat(x):
#         # x: [C, S]
#         mu = x.mean(dim=0, keepdim=True)
#         std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
#         return (x - mu) / std

#     def _compute_core_features(self, feature):
#         feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)  # [B, D]

#         diff = feature.unsqueeze(1) - self.means  # [B, C, D]
#         dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)  # [B, C]
#         logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
#         similarities = F.softmax(logits, dim=-1)  # [B, C]

#         prototype_feature = torch.matmul(similarities, self.class_embeddings)  # [B, D]
#         whitened_residual = torch.abs(diff * self.diag_inv_sqrt_cov.unsqueeze(0))  # [B, C, D]
#         r_bar = torch.sum(similarities.unsqueeze(-1) * whitened_residual, dim=1)  # [B, D]

#         if self.num_classes > 1:
#             top2_vals, _ = torch.topk(logits, k=2, dim=-1)
#             margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(-1)  # [B, 1]
#         else:
#             margin = torch.zeros((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
#         entropy = -(similarities * torch.log(similarities.clamp_min(1e-12))).sum(dim=-1, keepdim=True)  # [B, 1]

#         return logits, similarities, prototype_feature, r_bar, margin, entropy


# class CGCDSignalModuleSoftCovDetTrace(_CGCDSignalModuleCovBase):
#     """
#     Covariance-decomposed descriptor (det/trace/anisotropy branch).
#     desc = MLP([prototype ; r_bar ; margin ; entropy ; stat_bar])
#     stat_bar uses soft-weighted class-level:
#       logdet(Σ_c), trace(Σ_c), log(lambda_max/lambda_min).
#     """

#     def __init__(self, saved_models_dir, output_dim=256, stage=0, pca_path=None, scaler_path=None):
#         super().__init__(saved_models_dir, stage=stage, pca_path=pca_path, scaler_path=scaler_path)

#         eig_min = self.eigvals[:, :1]
#         eig_max = self.eigvals[:, -1:]
#         anisotropy = torch.log(eig_max) - torch.log(eig_min)  # [C, 1]

#         logdet = self.log_det.unsqueeze(-1)  # [C, 1]
#         logtrace = torch.log(self.diag_trace.clamp_min(1e-12))  # [C, 1]
#         class_stats = torch.cat([logdet, logtrace, anisotropy], dim=-1)  # [C, 3]
#         class_stats = self._normalize_per_class_stat(class_stats)
#         self.register_buffer("class_stats_dettrace", class_stats)

#         in_dim = 2 * self.feature_dim + 2 + class_stats.shape[-1]
#         hid_dim = self.feature_dim
#         self.descriptor_mlp = nn.Sequential(
#             nn.Linear(in_dim, hid_dim),
#             nn.GELU(),
#             nn.Linear(hid_dim, output_dim),
#         )

#     def forward(self, feature):
#         logits, similarities, prototype_feature, r_bar, margin, entropy = self._compute_core_features(feature)
#         stat_bar = torch.matmul(similarities, self.class_stats_dettrace)  # [B, 3]

#         descriptor = torch.cat([prototype_feature, r_bar, margin, entropy, stat_bar], dim=-1)
#         embedding = self.descriptor_mlp(descriptor)
#         embedding = F.normalize(embedding, p=2, dim=1)
#         return embedding, logits


# class CGCDSignalModuleSoftCovEigen(_CGCDSignalModuleCovBase):
#     """
#     Covariance-decomposed descriptor (eigen spectrum branch).
#     desc = MLP([prototype ; r_bar ; margin ; entropy ; eig_bar])
#     eig_bar uses soft-weighted top-k log-eigenvalues per class.
#     """

#     def __init__(self, saved_models_dir, output_dim=256, stage=0, pca_path=None, scaler_path=None, topk_eig=8):
#         super().__init__(saved_models_dir, stage=stage, pca_path=pca_path, scaler_path=scaler_path)

#         k = int(max(1, min(topk_eig, self.eigvals.shape[-1])))
#         topk = self.eigvals[:, -k:]  # [C, k]
#         log_topk = torch.log(topk.clamp_min(1e-12))
#         log_topk = self._normalize_per_class_stat(log_topk)
#         self.topk_eig = k
#         self.register_buffer("class_stats_eigen", log_topk)

#         in_dim = 2 * self.feature_dim + 2 + k
#         hid_dim = self.feature_dim
#         self.descriptor_mlp = nn.Sequential(
#             nn.Linear(in_dim, hid_dim),
#             nn.GELU(),
#             nn.Linear(hid_dim, output_dim),
#         )

#     def forward(self, feature):
#         logits, similarities, prototype_feature, r_bar, margin, entropy = self._compute_core_features(feature)
#         eig_bar = torch.matmul(similarities, self.class_stats_eigen)  # [B, k]

#         descriptor = torch.cat([prototype_feature, r_bar, margin, entropy, eig_bar], dim=-1)
#         embedding = self.descriptor_mlp(descriptor)
#         embedding = F.normalize(embedding, p=2, dim=1)
#         return embedding, logits


# class CGCDSignalModuleSoftCovDecomposed(_CGCDSignalModuleCovBase):
#     """
#     Full covariance-decomposed descriptor.
#     desc = MLP([prototype ; r_bar ; margin ; entropy ; det/trace/aniso ; eig_topk])
#     """

#     def __init__(self, saved_models_dir, output_dim=256, stage=0, pca_path=None, scaler_path=None, topk_eig=8):
#         super().__init__(saved_models_dir, stage=stage, pca_path=pca_path, scaler_path=scaler_path)

#         eig_min = self.eigvals[:, :1]
#         eig_max = self.eigvals[:, -1:]
#         anisotropy = torch.log(eig_max) - torch.log(eig_min)  # [C, 1]
#         logdet = self.log_det.unsqueeze(-1)  # [C, 1]
#         logtrace = torch.log(self.diag_trace.clamp_min(1e-12))  # [C, 1]
#         dettrace = torch.cat([logdet, logtrace, anisotropy], dim=-1)  # [C, 3]

#         k = int(max(1, min(topk_eig, self.eigvals.shape[-1])))
#         eig_topk = torch.log(self.eigvals[:, -k:].clamp_min(1e-12))  # [C, k]

#         class_stats = torch.cat([dettrace, eig_topk], dim=-1)  # [C, 3+k]
#         class_stats = self._normalize_per_class_stat(class_stats)

#         self.topk_eig = k
#         self.register_buffer("class_stats_all", class_stats)

#         in_dim = 2 * self.feature_dim + 2 + class_stats.shape[-1]
#         hid_dim = self.feature_dim
#         self.descriptor_mlp = nn.Sequential(
#             nn.Linear(in_dim, hid_dim),
#             nn.GELU(),
#             nn.Linear(hid_dim, output_dim),
#         )

#     def forward(self, feature):
#         logits, similarities, prototype_feature, r_bar, margin, entropy = self._compute_core_features(feature)
#         stat_bar = torch.matmul(similarities, self.class_stats_all)  # [B, 3+k]

#         descriptor = torch.cat([prototype_feature, r_bar, margin, entropy, stat_bar], dim=-1)
#         embedding = self.descriptor_mlp(descriptor)
#         embedding = F.normalize(embedding, p=2, dim=1)
#         return embedding, logits


class CGCDSignalModuleControlNet(nn.Module):
    zero_init_residual = True

    def __init__(self, saved_models_dir, output_dim=256, stage=1, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        self.num_classes = means_np.shape[0]

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        # Initialize learnable class embeddings from loaded means
        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())
        self.projection = nn.Linear(self.class_embeddings.shape[1], output_dim)

        # Mahalanobis terms via inverse sqrt covariance
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        covs_stable = self.covs + eye_matrix

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

        L, V = torch.linalg.eigh(covs_stable)
        L_clamped = L.clamp(min=1e-8)
        self.register_buffer("log_det", torch.log(L_clamped).sum(dim=-1))
        L_inv_sqrt = 1.0 / torch.sqrt(L_clamped)

        inv_sqrt_covs = torch.bmm(V, torch.bmm(torch.diag_embed(L_inv_sqrt), V.mT))
        self.register_buffer("inv_sqrt_covs", inv_sqrt_covs)

        self.residual_proj = nn.Linear(feature_dim, feature_dim)
        if self.zero_init_residual:
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

    def forward(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # z*sigma + u = x   =>  z*sigma = x - u   =>  z = (x - u) / sigma
        feature_diff = feature.unsqueeze(1) - self.means  # [B, C, D]
        whitened_feature_diff = torch.einsum("cij,bcj->bci", self.inv_sqrt_covs, feature_diff)
        mahalanobis_distance_squared = torch.sum(whitened_feature_diff * whitened_feature_diff, dim=-1)
        logits = -0.5 * (mahalanobis_distance_squared + self.log_det + self.log_2pi_term)
        
        similarities = F.softmax(logits, dim=-1)  # [B, C]
        prototype_feature = torch.matmul(similarities, self.class_embeddings)  # [B, D]

        instance_residual = torch.einsum("bc,bci->bi", similarities, whitened_feature_diff)
        gated_residual = self.residual_proj(instance_residual)

        instance_specific_feature = prototype_feature + gated_residual

        embedding = self.projection(instance_specific_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleControlNetNoZeroInit(CGCDSignalModuleControlNet):
    """ControlNet descriptor with the default Linear initialization for f_inst."""

    zero_init_residual = False


class CGCDSignalModuleControlNetPatchSummary(nn.Module):
    """
    Minimal patch-token enhancement for the ControlNet-style CGCD prompt.

    The Mahalanobis/prototype path still uses the original pooled DINO feature.
    A patch-token summary only adds an extra zero-init residual branch for
    instance-specific conditioning, so the baseline behavior is preserved at init.
    """

    def __init__(self, saved_models_dir, output_dim=256, stage=1, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        self.num_classes = means_np.shape[0]

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())
        self.projection = nn.Linear(self.class_embeddings.shape[1], output_dim)

        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        covs_stable = self.covs + eye_matrix

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

        L, V = torch.linalg.eigh(covs_stable)
        L_clamped = L.clamp(min=1e-8)
        self.register_buffer("log_det", torch.log(L_clamped).sum(dim=-1))
        L_inv_sqrt = 1.0 / torch.sqrt(L_clamped)

        inv_sqrt_covs = torch.bmm(V, torch.bmm(torch.diag_embed(L_inv_sqrt), V.mT))
        self.register_buffer("inv_sqrt_covs", inv_sqrt_covs)

        self.residual_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

        self.patch_detail_norm = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.patch_summary_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.patch_summary_proj.weight)
        nn.init.zeros_(self.patch_summary_proj.bias)

    def forward(self, feature):
        global_feature, patch_summary = _unpack_feature_inputs(feature)

        global_feature = _preprocess_feature(global_feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        feature_diff = global_feature.unsqueeze(1) - self.means  # [B, C, D]
        whitened_feature_diff = torch.einsum("cij,bcj->bci", self.inv_sqrt_covs, feature_diff)
        mahalanobis_distance_squared = torch.sum(whitened_feature_diff * whitened_feature_diff, dim=-1)
        logits = -0.5 * (mahalanobis_distance_squared + self.log_det + self.log_2pi_term)

        similarities = F.softmax(logits, dim=-1)  # [B, C]
        prototype_feature = torch.matmul(similarities, self.class_embeddings)  # [B, D]

        instance_residual = torch.einsum("bc,bci->bi", similarities, whitened_feature_diff)
        gated_residual = self.residual_proj(instance_residual)

        instance_specific_feature = prototype_feature + gated_residual

        if patch_summary is not None:
            patch_summary = _preprocess_feature(patch_summary, self.pca_layer, self.scaler_mean, self.scaler_scale)
            patch_detail = self.patch_detail_norm(patch_summary - global_feature)
            instance_specific_feature = instance_specific_feature + self.patch_summary_proj(patch_detail)

        embedding = self.projection(instance_specific_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleMMdit(nn.Module):
    """
    MMDiT-style zero-init shift/scale conditioning for CGCD prompt features.

    prototype_feature stays as the default task prompt, while instance_residual
    predicts a zero-init modulation delta on top of a normalized prototype:
        prototype + scale * LN(prototype) + shift
    This keeps the pure prototype path available when modulation weights stay at 0.
    """

    def __init__(self, saved_models_dir, output_dim=256, stage=1, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        self.num_classes = means_np.shape[0]

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())

        feature_dim = self.covs.shape[-1]
        self.prototype_norm = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(feature_dim, 2 * feature_dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.projection = nn.Linear(feature_dim, output_dim)

        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        covs_stable = self.covs + eye_matrix

        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

        L, V = torch.linalg.eigh(covs_stable)
        L_clamped = L.clamp(min=1e-8)
        self.register_buffer("log_det", torch.log(L_clamped).sum(dim=-1))
        L_inv_sqrt = 1.0 / torch.sqrt(L_clamped)

        inv_sqrt_covs = torch.bmm(V, torch.bmm(torch.diag_embed(L_inv_sqrt), V.mT))
        self.register_buffer("inv_sqrt_covs", inv_sqrt_covs)

    def forward(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        diff = feature.unsqueeze(1) - self.means  # [B, C, D]
        whitened_diff = torch.einsum("cij,bcj->bci", self.inv_sqrt_covs, diff)

        mahalanobis_distance_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
        logits = -0.5 * (mahalanobis_distance_sq + self.log_det + self.log_2pi_term)
        similarities = F.softmax(logits, dim=-1)  # [B, C]
        prototype_feature = torch.matmul(similarities, self.class_embeddings)  # [B, D]

        instance_residual = torch.einsum("bc,bci->bi", similarities, whitened_diff)
        shift, scale = self.adaLN_modulation(instance_residual).chunk(2, dim=1)

        normalized_prototype = self.prototype_norm(prototype_feature)
        instance_specific_feature = prototype_feature + normalized_prototype * scale + shift

        embedding = self.projection(instance_specific_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleControlNetEigen(nn.Module):
    def __init__(self, saved_models_dir, output_dim=256, stage=1, pca_path=None, scaler_path=None, top_k=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        self.num_classes = means_np.shape[0]

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())
        self.projection = nn.Linear(self.class_embeddings.shape[1], output_dim)

        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        covs_stable = self.covs + eye_matrix

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

        L, V = torch.linalg.eigh(covs_stable)
        L_clamped = L.clamp(min=1e-8)

        # log_det는 분포의 전체 형태(밀도)를 유지하기 위해 마스킹 전 원본 사용
        self.register_buffer("log_det", torch.log(L_clamped).sum(dim=-1))

        L_inv_sqrt = 1.0 / torch.sqrt(L_clamped)

        # [Eigen-Truncation 적용] eigh는 오름차순 정렬이므로 뒤에서부터 상위 top_k개
        if top_k is not None and top_k < feature_dim:
            mask = torch.zeros_like(L_inv_sqrt)
            mask[:, -top_k:] = 1.0  # 의미 있는 상위 주성분만 1
            L_inv_sqrt = L_inv_sqrt * mask  # 하위 노이즈 성분 차단 (0으로 만듦)

        inv_sqrt_covs = torch.bmm(V, torch.bmm(torch.diag_embed(L_inv_sqrt), V.mT))
        self.register_buffer("inv_sqrt_covs", inv_sqrt_covs)

        self.residual_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def forward(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        diff = feature.unsqueeze(1) - self.means  # [B, C, D]

        # 마스킹된 inv_sqrt_covs로 인해 노이즈가 제거된 핵심 패턴(Subspace)만 추출됨
        whitened_diff = torch.einsum("cij,bcj->bci", self.inv_sqrt_covs, diff)

        # Subspace Mahalanobis Distance (핵심 패턴 기반 거리 측정으로 더욱 Robust함)
        mahalanobis_distance_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
        logits = -0.5 * (mahalanobis_distance_sq + self.log_det + self.log_2pi_term)
        similarities = F.softmax(logits, dim=-1)  # [B, C]
        prototype_feature = torch.matmul(similarities, self.class_embeddings)  # [B, D]

        instance_residual = torch.einsum("bc,bci->bi", similarities, whitened_diff)
        gated_residual = self.residual_proj(instance_residual)

        instance_specific_feature = prototype_feature + gated_residual

        embedding = self.projection(instance_specific_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits
