import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_relative_ranking(mahalanobis_dist):
    """
    분석용: 거리에 따른 클래스 순위 반환 (가까운 순)
    """
    sorted_dist, indices = torch.sort(mahalanobis_dist, dim=-1)
    return indices, sorted_dist


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
    def __init__(self, saved_models_dir, output_dim=256, stage=0, pca_path=None, scaler_path=None):
        super().__init__()
        print(f"[CGCDSignalModuleSoft] Set stage={stage}")
        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        # Initialize learnable class embeddings from loaded means
        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())
        self.projection = nn.Linear(self.class_embeddings.shape[1], output_dim)

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

    def forward(self, feature):  # mode="weighted_sum"
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        mahalanobis_distance_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        # Compute similarity-weighted prototype feature
        logits = -0.5 * (mahalanobis_distance_sq + self.log_det + self.log_2pi_term)
        similarities = F.softmax(logits, dim=-1)

        prototype_feature = torch.matmul(similarities, self.class_embeddings)
        embedding = self.projection(prototype_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleSoftMargin(nn.Module):
    def __init__(self, saved_models_dir, output_dim=256, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        # Initialize learnable class embeddings from loaded means
        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())
        self.projection = nn.Linear(self.class_embeddings.shape[1], output_dim)

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

        self.conf_scale = nn.Parameter(torch.tensor(0.5))
        self.conf_bias = nn.Parameter(torch.tensor(-2.5))

    def forward(self, feature):  # mode="weighted_sum"
        with torch.no_grad():
            feature_pca = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)
            diff = feature_pca.unsqueeze(1) - self.means
            dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)
            logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
            top2_vals, top2_idxs = torch.topk(logits, 2, dim=-1)
            top1_idx = top2_idxs[:, 0]
            margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(1)

        confidence = torch.sigmoid(margin * self.conf_scale + self.conf_bias)

        # Sharp component: standard softmax (confident → 이쪽 비중 높음)
        similarities = F.softmax(logits, dim=-1)
        sharp_feature = torch.matmul(similarities, self.class_embeddings)

        # Soft component: high-temp softmax (uncertain → 이쪽 비중 높음)
        temp_scale = 1.0 + 4.0 * (1.0 - confidence)
        soft_probs = F.softmax(logits / temp_scale, dim=-1)
        soft_feature = torch.matmul(soft_probs, self.class_embeddings)

        # Confidence-based interpolation
        combined = confidence * sharp_feature + (1.0 - confidence) * soft_feature
        embedding = self.projection(combined)
        final_embedding = F.normalize(embedding, p=2, dim=1)

        return final_embedding, logits


class CGCDSignalModuleHard(nn.Module):
    def __init__(self, saved_models_dir, output_dim=256, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = self._load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        self.projection = nn.Linear(num_classes, output_dim)

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

    def _load_cgcd_stageN_parameters(self, saved_models_dir, stage, pca_path, scaler_path=None):
        return load_cgcd_stageN_parameters(saved_models_dir, stage, pca_path, scaler_path)

    def forward(self, feature):  # mode="weighted_sum"
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        # Compute similarity-weighted prototype feature
        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
        similarities = F.softmax(logits, dim=-1)

        embedding = self.projection(similarities)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleStaticSoft(nn.Module):
    def __init__(self, saved_models_dir, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

    def forward(self, feature):  # mode="weighted_sum"
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        # Compute similarity-weighted prototype feature
        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
        similarities = F.softmax(logits, dim=-1)
        embedding = torch.matmul(similarities, self.means.squeeze(0))

        return embedding, logits


class CGCDSignalModuleStaticHard(nn.Module):
    def __init__(self, saved_models_dir, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

    def forward(self, feature):  # mode="weighted_sum"
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        # Compute similarity-weighted prototype feature
        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
        top1_idx = logits.argmax(dim=-1)
        embedding = self.means.squeeze(0)[top1_idx]

        return embedding, logits


class CGCDSignalModuleStaticHardFixedEmbedding(nn.Module):
    def __init__(self, saved_models_dir, output_dim=None, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        feature_dim = means_np.shape[1]
        self.output_projection = None
        if output_dim is not None:
            output_dim = int(output_dim)
            if output_dim == feature_dim:
                self.output_projection = nn.Identity()
            else:
                self.output_projection = nn.Linear(feature_dim, output_dim)

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

    def forward(self, feature):  # mode="weighted_sum"
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
        top1_idx = logits.argmax(dim=-1)
        embedding = self.means.squeeze(0)[top1_idx]
        if self.output_projection is not None:
            embedding = self.output_projection(embedding)
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleStaticSoftFixedEmbedding(CGCDSignalModuleStaticHardFixedEmbedding):
    """Soft posterior mixture of fixed Gaussian means."""

    def forward(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)
        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)

        similarities = F.softmax(logits, dim=-1)
        embedding = torch.matmul(similarities, self.means.squeeze(0))
        if self.output_projection is not None:
            embedding = self.output_projection(embedding)
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleStaticHardLeanableEmbedding(nn.Module):
    def __init__(self, saved_models_dir, output_dim=None, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())
        feature_dim = means_np.shape[1]
        self.output_projection = None
        if output_dim is not None:
            output_dim = int(output_dim)
            if output_dim == feature_dim:
                self.output_projection = nn.Identity()
            else:
                self.output_projection = nn.Linear(feature_dim, output_dim)

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

    def forward(self, feature):  # mode="weighted_sum"
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
        top1_idx = logits.argmax(dim=-1)
        embedding = self.class_embeddings[top1_idx]
        if self.output_projection is not None:
            embedding = self.output_projection(embedding)
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleStaticHardLeanableEmbeddingInst(nn.Module):
    def __init__(self, saved_models_dir, output_dim=None, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())
        feature_dim = means_np.shape[1]
        self.output_projection = None
        if output_dim is not None:
            output_dim = int(output_dim)
            if output_dim == feature_dim:
                self.output_projection = nn.Identity()
            else:
                self.output_projection = nn.Linear(feature_dim, output_dim)

        # Mahalanobis terms
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        covs_stable = self.covs + eye_matrix
        inv_covs = torch.linalg.inv(covs_stable)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(covs_stable)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

        L, V = torch.linalg.eigh(covs_stable)
        L_inv_sqrt = 1.0 / torch.sqrt(L.clamp(min=1e-8))
        inv_sqrt_covs = torch.bmm(V, torch.bmm(torch.diag_embed(L_inv_sqrt), V.mT))
        self.register_buffer("inv_sqrt_covs", inv_sqrt_covs)

        self.residual_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def forward(self, feature):  # mode="weighted_sum"
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
        top1_idx = logits.argmax(dim=-1)
        prototype_feature = self.class_embeddings[top1_idx]

        whitened_diff = torch.einsum("cij,bcj->bci", self.inv_sqrt_covs, diff)
        batch_idx = torch.arange(feature.shape[0], device=feature.device)
        instance_residual = whitened_diff[batch_idx, top1_idx]
        instance_specific_feature = prototype_feature + self.residual_proj(instance_residual)

        if self.output_projection is not None:
            embedding = self.output_projection(instance_specific_feature)
            embedding = F.normalize(embedding, p=2, dim=1)
        else:
            embedding = instance_specific_feature

        return embedding, logits


class CGCDSignalModuleStaticHardLeanableEmbeddingSoftInst(CGCDSignalModuleStaticHardLeanableEmbeddingInst):
    """Hard top-1 prototype with a posterior-weighted instance residual."""

    def forward(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)
        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)

        top1_idx = logits.argmax(dim=-1)
        prototype_feature = self.class_embeddings[top1_idx]

        similarities = F.softmax(logits, dim=-1)
        whitened_diff = torch.einsum("cij,bcj->bci", self.inv_sqrt_covs, diff)
        instance_residual = torch.einsum("bc,bci->bi", similarities, whitened_diff)
        instance_specific_feature = prototype_feature + self.residual_proj(instance_residual)

        if self.output_projection is not None:
            embedding = self.output_projection(instance_specific_feature)
            embedding = F.normalize(embedding, p=2, dim=1)
        else:
            embedding = instance_specific_feature

        return embedding, logits


class CGCDSignalModuleLearnable(nn.Module):
    def __init__(self, saved_models_dir, output_dim=400, stage=0, pca_path=None, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        feature_dim = means_np.shape[1]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        #! Initialize learnable class embeddings (random initialization)
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.projection = nn.Linear(feature_dim, output_dim)

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

    def forward(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        # Compute similarity-weighted prototype feature
        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
        similarities = F.softmax(logits, dim=-1)

        prototype_feature = torch.matmul(similarities, self.class_embeddings)
        embedding = self.projection(prototype_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleMargin(nn.Module):
    """CGCDSignalModuleSoft with margin-adaptive temperature.

    When top1-top2 margin is large (confident) → temperature drops → sharper softmax → one prototype dominates.
    When margin is small (uncertain) → temperature stays high → softer softmax → prototypes blend conservatively.
    This changes the embedding *direction*, surviving downstream L2 normalization.
    """

    def __init__(
        self, saved_models_dir, output_dim=256, stage=0, pca_path=None, base_temp=3.0, scale=30.0, scaler_path=None
    ):
        super().__init__()

        self.base_temp = base_temp
        self.scale = scale

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        # ! learnable parameteres
        # Initialize learnable class embeddings from loaded means
        self.class_embeddings = nn.Parameter(self.means.squeeze(0).clone())
        self.projection = nn.Linear(self.class_embeddings.shape[1], output_dim)

        # Mahalanobis Inverse Cov
        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

    def forward(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        # Compute Mahalanobis distance to each prototype
        diff = feature.unsqueeze(1) - self.means
        dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

        # Gaussian log-probability as logits
        logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)

        # Margin-adaptive temperature
        top2_vals = torch.topk(logits, k=2, dim=1).values  # [B, 2]
        margin = top2_vals[:, 0] - top2_vals[:, 1]  # [B], >= 0

        # margin=0  → temp=base_temp (softest, most uncertain)
        # margin=30 → temp=base_temp/2
        # margin=60 → temp=base_temp/3 (sharpest, most confident)
        adaptive_temp = self.base_temp / (1.0 + margin / self.scale)  # [B]

        similarities = F.softmax(logits / adaptive_temp.unsqueeze(1), dim=-1)

        prototype_feature = torch.matmul(similarities, self.class_embeddings)
        embedding = self.projection(prototype_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


# class Head(nn.Module):
#     def __init__(self, embedding_dim=384, hidden_dim=256, num_classes=12):
#         super(Head, self).__init__()

#         self.fc1 = nn.Linear(embedding_dim, hidden_dim)
#         # self.gelu = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         embd = self.fc1(x)
#         embd = F.normalize(embd, p=2, dim=1)
#         deg_pred = self.fc2(embd)
#         return embd, deg_pred


# def generate_restoration_prompt(classifier, features):
#     # 1. PCA 변환 (기존 모델과 차원 맞추기)
#     features_pca, _ = classifier.pre_processing(features, None)

#     # 2. 각 클래스별 log_prob 계산
#     _, logits = classifier._predict(jnp.array(features_pca), classifier.params)

#     # 3. 클래스 이름 정의 (코드에 적어주신 리스트)
#     class_names_list = ["clear", "haze", "haze_rain", "haze_snow", "low",
#                         "low_haze", "low_haze_rain", "low_haze_snow",
#                         "low_rain", "low_snow", "rain", "snow"]

#     prompts = []
#     for i in range(len(features)):
#         scores = logits[i]
#         # 확률(거리) 순으로 정렬 (큰 값이 위로 오게)
#         sorted_indices = jnp.argsort(scores)[::-1]

#         # 실제 클래스 이름으로 매핑 (class_order 고려)
#         def get_name(idx):
#             orig_id = classifier.class_order[idx] if classifier.class_order is not None else idx
#             return class_names_list[orig_id]

#         target = get_name(sorted_indices[0])      # 1순위 (예: low_rain)
#         near = get_name(sorted_indices[1])        # 2순위 (예: rain)
#         far = get_name(sorted_indices[-1])        # 최하위 (예: snow)

#         # 4. 프롬프트 구성
#         prompt = (f"The image is primarily {target}. "
#                   f"It is visually similar to {near}, "
#                   f"but significantly different from {far}.")
#         prompts.append(prompt)

#     return prompts


class CGCDSignalModuleTrainable(nn.Module):
    def __init__(self, saved_models_dir, stage=0, pca_path=None, embedding_dim=324, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)
        for param in self.pca_layer.parameters():
            param.requires_grad = False

        num_classes = means_np.shape[0]
        self.num_classes = num_classes

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)

        log_det = torch.logdet(self.covs + eye_matrix)
        self.register_buffer("log_det", log_det)

        feature_dim = self.covs.shape[-1]
        log_2pi_term = feature_dim * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer("log_2pi_term", log_2pi_term)

        self.prompt_bank = nn.Embedding(num_classes, embedding_dim)
        nn.init.orthogonal_(self.prompt_bank.weight)

    def forward(self, feature):
        with torch.no_grad():
            feature_pca = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)
            diff = feature_pca.unsqueeze(1) - self.means
            dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

            logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
            top1_idx = logits.argmax(dim=-1)

        embedding = self.prompt_bank(top1_idx)

        return embedding, logits, top1_idx
