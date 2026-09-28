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


class CGCDSignalModulePrompt(nn.Module):
    def __init__(self, saved_models_dir, stage=0, pca_path=None, output_dim=324, scaler_path=None):
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

        self.prompt_bank = nn.Embedding(num_classes, output_dim)
        nn.init.orthogonal_(self.prompt_bank.weight)

    def forward(self, feature):
        with torch.no_grad():
            feature_pca = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)
            diff = feature_pca.unsqueeze(1) - self.means
            dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)

            logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
            top1_idx = logits.argmax(dim=-1)

        embedding = self.prompt_bank(top1_idx)

        return embedding, logits


class CGCDSignalModuleMargin(nn.Module):
    def __init__(self, saved_models_dir, stage=0, pca_path=None, output_dim=512, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        for param in self.pca_layer.parameters():
            param.requires_grad = False

        self.num_classes = means_np.shape[0]
        self.dino_dim = means_np.shape[-1]

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

        self.task_query = nn.Embedding(self.num_classes, output_dim)
        nn.init.orthogonal_(self.task_query.weight)

        self.context_query = nn.Embedding(self.num_classes, output_dim)
        nn.init.orthogonal_(self.context_query.weight)
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, feature):
        with torch.no_grad():
            feature_pca = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)
            diff = feature_pca.unsqueeze(1) - self.means
            dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)
            logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)
            top1_idx = logits.argmax(dim=-1)

        base_query = self.task_query(top1_idx)

        similarities = F.softmax(logits / self.temperature, dim=-1)
        context_info = similarities @ self.context_query.weight

        final_embedding = base_query + context_info

        return final_embedding, logits


class CGCDSignalModuleMarginAdvanced(nn.Module):
    def __init__(self, saved_models_dir, stage=0, pca_path=None, output_dim=512, scaler_path=None):
        super().__init__()

        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        for param in self.pca_layer.parameters():
            param.requires_grad = False

        self.num_classes = means_np.shape[0]
        self.dino_dim = means_np.shape[-1]

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        eps = 1e-6
        eye = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        self.register_buffer("inv_covs", torch.linalg.inv(self.covs + eye))
        self.register_buffer("log_det", torch.logdet(self.covs + eye))
        self.register_buffer("log_2pi_term", self.covs.shape[-1] * torch.log(torch.tensor(2 * torch.pi)))

        # 2. Learnable Embeddings (Orthogonal Init)
        self.task_query = nn.Embedding(self.num_classes, output_dim)
        nn.init.orthogonal_(self.task_query.weight)

        self.context_query = nn.Embedding(self.num_classes, output_dim)
        nn.init.orthogonal_(self.context_query.weight)

        # 4. Confidence Control Parameters (Coupled)
        self.conf_scale = nn.Parameter(torch.tensor(0.5))
        self.conf_bias = nn.Parameter(torch.tensor(-2.5))

    def forward(self, feature):
        # CGCD Logic (Frozen)
        with torch.no_grad():
            feature_pca = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)
            diff = feature_pca.unsqueeze(1) - self.means
            dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)
            logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)

            # Margin Calculation
            top2_vals, top2_idxs = torch.topk(logits, 2, dim=-1)
            top1_idx = top2_idxs[:, 0]
            margin = (top2_vals[:, 0] - top2_vals[:, 1]).unsqueeze(1)

        # Confidence & Control Logic
        confidence = torch.sigmoid(margin * self.conf_scale + self.conf_bias)

        # Coupled Control: High Conf -> Use Hard Gate & Low Temp
        base_gate = confidence
        temp_scale = 1.0 + 4.0 * (1.0 - confidence)

        # Embedding Generation
        base_query = self.task_query(top1_idx) * base_gate

        soft_probs = F.softmax(logits / temp_scale, dim=-1)
        context_info = soft_probs @ self.context_query.weight

        final_embedding = base_query + context_info
        # extra_info = {"logits": logits, "margin": margin, "confidence": confidence, "temp": temp_scale}
        return final_embedding, logits


class CGCDSignalModuleDinoConfFinal(nn.Module):
    def __init__(self, saved_models_dir, stage=0, pca_path=None, output_dim=512, scaler_path=None):
        super().__init__()
        means_np, covs_np, self.pca_layer, scaler_mean, scaler_scale = load_cgcd_stageN_parameters(
            saved_models_dir, stage, pca_path, scaler_path
        )
        _register_scaler_buffers(self, scaler_mean, scaler_scale)

        for param in self.pca_layer.parameters():
            param.requires_grad = False

        self.num_classes = means_np.shape[0]
        self.dino_dim = means_np.shape[-1]

        self.register_buffer("means", torch.from_numpy(means_np).float().unsqueeze(0))
        self.register_buffer("covs", torch.from_numpy(covs_np).float())

        eps = 1e-6
        eye_matrix = torch.eye(self.covs.shape[-1], device=self.covs.device) * eps
        inv_covs = torch.linalg.inv(self.covs + eye_matrix)
        self.register_buffer("inv_covs", inv_covs)
        self.register_buffer("log_det", torch.logdet(self.covs + eye_matrix))
        self.register_buffer("log_2pi_term", self.covs.shape[-1] * torch.log(torch.tensor(2 * torch.pi)))

        self.task_query = nn.Embedding(self.num_classes, output_dim)
        nn.init.orthogonal_(self.task_query.weight)

        self.context_query = nn.Embedding(self.num_classes, output_dim)
        nn.init.orthogonal_(self.context_query.weight)

        self.confidence_control = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 2)  # [Temperature_Scale, Base_Gate]
        )
        nn.init.constant_(self.confidence_control[-1].bias, 1.0)

    def forward(self, feature):
        with torch.no_grad():
            feature_pca = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)
            diff = feature_pca.unsqueeze(1) - self.means
            dist_sq = torch.einsum("bci,cij,bcj->bc", diff, self.inv_covs, diff)
            logits = -0.5 * (dist_sq + self.log_det + self.log_2pi_term)

            top2_vals, top2_idxs = torch.topk(logits, 2, dim=-1)
            top1_idx = top2_idxs[:, 0]

            margin = top2_vals[:, 0] - top2_vals[:, 1]
            margin = margin.unsqueeze(1)

        controls = self.confidence_control(margin)  # [Batch, 2]

        # (1) Adaptive Temperature
        temp_scale = F.softplus(controls[:, 0:1]) + 0.1
        base_gate = torch.sigmoid(controls[:, 1:2])

        hard_embedding = self.task_query(top1_idx) * base_gate

        soft_probs = F.softmax(logits / temp_scale, dim=-1)
        soft_embedding = soft_probs @ self.context_query.weight

        final_embedding = hard_embedding + soft_embedding

        # extra_info = {"logits": logits, "top1_idx": top1_idx, "margin": margin, "temp": temp_scale, "gate": base_gate}

        return final_embedding, logits
