import torch
import torch.nn as nn
import torch.nn.functional as F

from cgcd.models_cov import CGCDSignalModuleControlNet, _preprocess_feature


class _CGCDAdaInPerturbBase(CGCDSignalModuleControlNet):
    """Base helper to toggle perturbation on/off at runtime (e.g., student on / teacher off)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_enabled = True

    def set_perturb_enabled(self, enabled=True):
        self.perturb_enabled = bool(enabled)

    def _compute_core(self, feature):
        feature = _preprocess_feature(feature, self.pca_layer, self.scaler_mean, self.scaler_scale)

        diff = feature.unsqueeze(1) - self.means  # [B, C, D]
        whitened_diff = torch.einsum("cij,bcj->bci", self.inv_sqrt_covs, diff)

        mahalanobis_distance_sq = torch.sum(whitened_diff * whitened_diff, dim=-1)
        logits = -0.5 * (mahalanobis_distance_sq + self.log_det + self.log_2pi_term)
        similarities = F.softmax(logits, dim=-1)  # [B, C]

        prototype_feature = torch.matmul(similarities, self.class_embeddings)  # [B, D]
        instance_residual = torch.einsum("bc,bci->bi", similarities, whitened_diff)

        return prototype_feature, instance_residual, logits


class CGCDSignalModuleControlNetWeightPerturb(_CGCDAdaInPerturbBase):
    def __init__(
        self,
        saved_models_dir,
        output_dim=256,
        stage=1,
        pca_path=None,
        scaler_path=None,
        noise_std=0.01,
        perturb_on_eval=False,
    ):
        super().__init__(
            saved_models_dir=saved_models_dir,
            output_dim=output_dim,
            stage=stage,
            pca_path=pca_path,
            scaler_path=scaler_path,
        )
        self.noise_std = float(noise_std)
        self.perturb_on_eval = bool(perturb_on_eval)

    def _use_weight_noise(self):
        if not self.perturb_enabled:
            return False
        if self.noise_std <= 0.0:
            return False
        return self.training or self.perturb_on_eval

    def forward(self, feature):
        prototype_feature, instance_residual, logits = self._compute_core(feature)

        if self._use_weight_noise():
            noise = torch.randn_like(self.residual_proj.weight) * self.noise_std
            noisy_weight = self.residual_proj.weight + noise
            gated_residual = F.linear(instance_residual, noisy_weight, self.residual_proj.bias)
        else:
            gated_residual = self.residual_proj(instance_residual)

        instance_specific_feature = prototype_feature + gated_residual
        embedding = self.projection(instance_specific_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleControlNetResidualDropout(_CGCDAdaInPerturbBase):
    """
    Option_B (Priority 1B): apply dropout on gated residual features.

    Perturbation is active only when:
    - self.perturb_enabled is True
    - training mode
    """

    def __init__(
        self,
        saved_models_dir,
        output_dim=256,
        stage=1,
        pca_path=None,
        scaler_path=None,
        dropout_p=0.4,
    ):
        super().__init__(
            saved_models_dir=saved_models_dir,
            output_dim=output_dim,
            stage=stage,
            pca_path=pca_path,
            scaler_path=scaler_path,
        )
        dropout_p = float(dropout_p)
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
        self.dropout = nn.Dropout(p=dropout_p)

    def _use_dropout(self):
        return self.perturb_enabled and self.training and self.dropout.p > 0.0

    def forward(self, feature):
        prototype_feature, instance_residual, logits = self._compute_core(feature)

        gated_residual = self.residual_proj(instance_residual)
        if self._use_dropout():
            gated_residual = self.dropout(gated_residual)

        instance_specific_feature = prototype_feature + gated_residual
        embedding = self.projection(instance_specific_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits


class CGCDSignalModuleControlNetResidualUniformMul(_CGCDAdaInPerturbBase):
    """
    Option_C: apply CCT-style uniform multiplicative noise only on gated residual.

    gated_residual <- gated_residual * (1 + eps), eps ~ U(-r, r)

    Perturbation is active only when:
    - self.perturb_enabled is True
    - training mode (or perturb_on_eval=True)
    - effective range r > 0
    """

    def __init__(
        self,
        saved_models_dir,
        output_dim=256,
        stage=1,
        pca_path=None,
        scaler_path=None,
        noise_range=0.3,
        perturb_on_eval=False,
    ):
        super().__init__(
            saved_models_dir=saved_models_dir,
            output_dim=output_dim,
            stage=stage,
            pca_path=pca_path,
            scaler_path=scaler_path,
        )
        self.noise_range = float(noise_range)
        if self.noise_range < 0.0:
            raise ValueError(f"noise_range must be >= 0, got {self.noise_range}")
        self.perturb_on_eval = bool(perturb_on_eval)
        # negative means "use default self.noise_range"
        self.register_buffer("_runtime_noise_range", torch.tensor(-1.0))

    def set_runtime_noise_range(self, noise_range=None):
        if noise_range is None:
            self._runtime_noise_range.fill_(-1.0)
            return
        noise_range = float(noise_range)
        if noise_range < 0.0:
            raise ValueError(f"runtime noise_range must be >= 0, got {noise_range}")
        self._runtime_noise_range.fill_(noise_range)

    def _effective_noise_range(self):
        runtime = float(self._runtime_noise_range.item())
        return self.noise_range if runtime < 0.0 else runtime

    def _use_noise(self, noise_range):
        if not self.perturb_enabled:
            return False
        if noise_range <= 0.0:
            return False
        return self.training or self.perturb_on_eval

    def forward(self, feature):
        prototype_feature, instance_residual, logits = self._compute_core(feature)

        gated_residual = self.residual_proj(instance_residual)
        noise_range = self._effective_noise_range()
        if self._use_noise(noise_range):
            eps = torch.empty_like(gated_residual).uniform_(-noise_range, noise_range)
            gated_residual = gated_residual * (1.0 + eps)

        instance_specific_feature = prototype_feature + gated_residual
        embedding = self.projection(instance_specific_feature)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding, logits


class CGCDSignalModuleWeightLamda(_CGCDAdaInPerturbBase):
    def __init__(
        self,
        saved_models_dir,
        output_dim=256,
        stage=1,
        pca_path=None,
        scaler_path=None,
        perturb_on_eval=False,
    ):
        super().__init__(
            saved_models_dir=saved_models_dir,
            output_dim=output_dim,
            stage=stage,
            pca_path=pca_path,
            scaler_path=scaler_path,
        )
        self.perturb_on_eval = bool(perturb_on_eval)

    def _use_lamda_weight(self):
        if not self.perturb_enabled:
            return False
        return self.training or self.perturb_on_eval

    def _prepare_lamda_weight(self, lamda_weight, batch_size, device, dtype):
        if lamda_weight is None:
            lamda_tensor = torch.ones((batch_size, 1), device=device, dtype=dtype)
        elif torch.is_tensor(lamda_weight):
            lamda_tensor = lamda_weight.to(device=device, dtype=dtype)
            if lamda_tensor.dim() == 0:
                lamda_tensor = lamda_tensor.view(1, 1).expand(batch_size, 1)
            elif lamda_tensor.dim() == 1:
                if lamda_tensor.shape[0] == 1:
                    lamda_tensor = lamda_tensor.view(1, 1).expand(batch_size, 1)
                elif lamda_tensor.shape[0] == batch_size:
                    lamda_tensor = lamda_tensor.view(batch_size, 1)
                else:
                    raise ValueError(
                        f"lamda_weight 1D tensor must have len 1 or batch_size={batch_size}, got {lamda_tensor.shape[0]}"
                    )
            elif lamda_tensor.dim() == 2 and lamda_tensor.shape == (batch_size, 1):
                pass
            else:
                raise ValueError(
                    f"Unsupported lamda_weight tensor shape {tuple(lamda_tensor.shape)}; expected scalar, [B], or [B,1]"
                )
        else:
            lamda_value = float(lamda_weight)
            lamda_tensor = torch.full((batch_size, 1), lamda_value, device=device, dtype=dtype)

        if torch.any(lamda_tensor < 0.0):
            raise ValueError(f"lamda_weight must be >= 0, got min={float(lamda_tensor.min().item())}")
        return lamda_tensor

    def forward(self, feature, lamda_weight=None):
        prototype_feature, instance_residual, logits = self._compute_core(feature)

        gated_residual = self.residual_proj(instance_residual)
        if self._use_lamda_weight():
            lamda_tensor = self._prepare_lamda_weight(
                lamda_weight=lamda_weight,
                batch_size=gated_residual.shape[0],
                device=gated_residual.device,
                dtype=gated_residual.dtype,
            )
            gated_residual = gated_residual * lamda_tensor

        instance_specific_feature = prototype_feature + gated_residual
        embedding = self.projection(instance_specific_feature)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, logits