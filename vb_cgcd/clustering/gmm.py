#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from typing import override
from typing_extensions import override
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.special import gammaln, multigammaln

from .clustering_base import ClusteringBase

import numpy as np


class GMMCluster(ClusteringBase):

    def __init__(
        self,
        init_components=None,
        label_offset=0,
        random_state=None,
        auto_estimate=False,
        min_components=1,
        max_components=10,
        criterion="bic",
        dp_weight_concentration_prior=None,
        dp_min_count=1,
        dp_min_weight=0.0,
        split_merge_max_iter=5,
        split_merge_min_cluster_size=6,
        split_merge_alpha=1.0,
        num_classes=None,
    ):
        if init_components is None:
            if num_classes is None:
                raise ValueError("GMMCluster requires init_components.")
            init_components = num_classes

        super().__init__(init_components, label_offset)

        self.random_state = random_state
        self.auto_estimate = auto_estimate
        self.min_components = min_components
        self.max_components = max_components
        self.criterion = criterion.lower()  # 'bic', 'aic', or 'dp_gmm'
        self.dp_weight_concentration_prior = dp_weight_concentration_prior
        self.dp_min_count = max(int(dp_min_count), 1)
        self.dp_min_weight = max(float(dp_min_weight), 0.0)
        self.split_merge_max_iter = max(int(split_merge_max_iter), 1)
        self.split_merge_min_cluster_size = max(int(split_merge_min_cluster_size), 2)
        self.split_merge_alpha = max(float(split_merge_alpha), 1e-8)
        self.estimated_n_components = init_components
        self.active_component_map = None
        self.active_component_ids = None
        self.component_fallback_map = None

        if not auto_estimate:
            self.model = GaussianMixture(n_components=init_components, random_state=random_state)
        else:
            self.model = None

    def _find_optimal_components(self, features):
        """Find optimal number of components using BIC or AIC."""
        scores = []
        upper = min(self.max_components, len(features))
        if upper < self.min_components:
            raise ValueError(
                f"Invalid component search range: min_components={self.min_components}, "
                f"available_samples={len(features)}"
            )
        n_components_range = range(self.min_components, upper + 1)

        print(f"\nSearching for optimal number of components ({self.criterion.upper()})...")

        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=self.random_state)
            gmm.fit(features)

            if self.criterion == "bic":
                score = gmm.bic(features)
            else:  # aic
                score = gmm.aic(features)

            scores.append(score)
            print(f"  n_components={n}: {self.criterion.upper()}={score:.2f}")

        optimal_idx = int(np.argmin(scores))
        optimal_n = list(n_components_range)[optimal_idx]

        print(f"✓ Optimal number of components: {optimal_n} ({self.criterion.upper()}={scores[optimal_idx]:.2f})\n")

        return optimal_n

    def _component_search_bounds(self, features):
        upper = min(self.max_components, len(features))
        if upper < self.min_components:
            raise ValueError(
                f"Invalid component search range: min_components={self.min_components}, "
                f"available_samples={len(features)}"
            )
        return upper

    def _stable_logdet(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        eye = np.eye(matrix.shape[0], dtype=np.float64)
        for jitter in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
            sign, logdet = np.linalg.slogdet(matrix + jitter * eye)
            if sign > 0:
                return float(logdet)
        return float("-inf")

    def _build_niw_prior(self, features):
        features = np.asarray(features, dtype=np.float64)
        feat_dim = int(features.shape[1])
        avg_var = float(np.mean(np.var(features, axis=0))) if len(features) > 1 else 1.0
        sigma_scale = max(avg_var * 5e-3, 1e-6)
        return {
            "mu0": features.mean(axis=0),
            "psi": np.eye(feat_dim, dtype=np.float64) * sigma_scale,
            "kappa": 1e-4,
            "nu": float(feat_dim + 2),
        }

    def _log_marginal_likelihood(self, codes_k, mu_k, prior):
        codes_k = np.asarray(codes_k, dtype=np.float64)
        mu_k = np.asarray(mu_k, dtype=np.float64)
        n_k, feat_dim = codes_k.shape
        if n_k == 0:
            return float("-inf")

        kappa0 = float(prior["kappa"])
        nu0 = float(prior["nu"])
        mu0 = np.asarray(prior["mu0"], dtype=np.float64)
        psi0 = np.asarray(prior["psi"], dtype=np.float64)

        sum_k = codes_k.sum(axis=0)
        kappa_star = kappa0 + n_k
        nu_star = nu0 + n_k
        centered = codes_k - mu_k
        scatter = centered.T @ centered
        delta = (mu_k - mu0).reshape(-1, 1)
        psi_star = psi0 + scatter + ((kappa0 * n_k) / kappa_star) * (delta @ delta.T)

        logdet_psi0 = self._stable_logdet(psi0)
        logdet_psistar = self._stable_logdet(psi_star)
        if not np.isfinite(logdet_psi0) or not np.isfinite(logdet_psistar):
            return float("-inf")

        return float(
            -(n_k * feat_dim / 2.0) * np.log(np.pi)
            + multigammaln(nu_star / 2.0, feat_dim)
            - multigammaln(nu0 / 2.0, feat_dim)
            + (nu0 / 2.0) * logdet_psi0
            - (nu_star / 2.0) * logdet_psistar
            + (feat_dim / 2.0) * (np.log(kappa0) - np.log(kappa_star))
        )

    def _fit_kmeans_assignments(self, features, n_components):
        model = KMeans(n_clusters=n_components, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(features)
        return model.cluster_centers_, labels

    def _fit_gmm_assignments(self, features, n_components):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=self.random_state,
            reg_covar=1e-6,
            n_init=3,
            max_iter=500,
        )
        model.fit(features)
        labels = model.predict(features)
        return model.means_, labels

    def _split_candidates(self, features, centers, labels, prior):
        split_ids = []
        for cluster_id in range(len(centers)):
            cluster_mask = labels == cluster_id
            cluster_feats = features[cluster_mask]
            min_size = self.split_merge_min_cluster_size
            if len(cluster_feats) < max(2 * min_size, 6):
                continue

            try:
                sub_centers, sub_labels = self._fit_kmeans_assignments(cluster_feats, 2)
            except Exception:
                continue

            sub_counts = np.bincount(sub_labels, minlength=2)
            if np.any(sub_counts < min_size):
                continue

            log_ll_parent = self._log_marginal_likelihood(cluster_feats, centers[cluster_id], prior)
            log_ll_0 = self._log_marginal_likelihood(cluster_feats[sub_labels == 0], sub_centers[0], prior)
            log_ll_1 = self._log_marginal_likelihood(cluster_feats[sub_labels == 1], sub_centers[1], prior)
            split_score = (
                np.log(self.split_merge_alpha)
                + gammaln(int(sub_counts[0]))
                + log_ll_0
                + gammaln(int(sub_counts[1]))
                + log_ll_1
                - (gammaln(int(len(cluster_feats))) + log_ll_parent)
            )
            if split_score > 0:
                split_ids.append(cluster_id)
        return split_ids

    def _merge_candidates(self, features, centers, labels, prior, frozen_ids=None):
        frozen_ids = set() if frozen_ids is None else set(int(x) for x in frozen_ids)
        if len(centers) < 2:
            return []

        dist_pairs = []
        for left in range(len(centers)):
            if left in frozen_ids:
                continue
            for right in range(left + 1, len(centers)):
                if right in frozen_ids:
                    continue
                dist = float(np.sum((centers[left] - centers[right]) ** 2))
                dist_pairs.append((dist, left, right))
        dist_pairs.sort(key=lambda item: item[0])

        used = set()
        merge_pairs = []
        for _, left, right in dist_pairs:
            if left in used or right in used:
                continue

            left_feats = features[labels == left]
            right_feats = features[labels == right]
            if len(left_feats) == 0 or len(right_feats) == 0:
                continue

            merged_feats = np.concatenate([left_feats, right_feats], axis=0)
            merged_center = (len(left_feats) * centers[left] + len(right_feats) * centers[right]) / float(
                len(merged_feats)
            )
            log_ll_merge = self._log_marginal_likelihood(merged_feats, merged_center, prior)
            log_ll_left = self._log_marginal_likelihood(left_feats, centers[left], prior)
            log_ll_right = self._log_marginal_likelihood(right_feats, centers[right], prior)
            merge_score = (
                gammaln(int(len(merged_feats)))
                - (np.log(self.split_merge_alpha) + gammaln(int(len(left_feats))) + gammaln(int(len(right_feats))))
                + (log_ll_merge - (log_ll_left + log_ll_right))
            )
            if merge_score > 0:
                merge_pairs.append((left, right))
                used.add(left)
                used.add(right)
        return merge_pairs

    def _estimate_deepdpm_components(self, features):
        features = np.asarray(features, dtype=np.float64)
        upper = self._component_search_bounds(features)
        current_k = int(np.clip(self.init_components, self.min_components, upper))
        prior = self._build_niw_prior(features)

        print(
            "\nSearching for active components (DEEPDPM split-merge): "
            f"init={current_k}, min_components={self.min_components}, max_components={upper}"
        )

        for iteration in range(self.split_merge_max_iter):
            try:
                centers, labels = self._fit_kmeans_assignments(features, current_k)
            except Exception as exc:
                print(f"  split-merge fit failed at iter={iteration}: {exc}")
                break

            split_ids = self._split_candidates(features, centers, labels, prior)
            merge_pairs = self._merge_candidates(features, centers, labels, prior, frozen_ids=split_ids)
            proposed_k = current_k + len(split_ids) - len(merge_pairs)
            proposed_k = int(np.clip(proposed_k, self.min_components, upper))

            print(
                f"  iter={iteration + 1}: k={current_k}, "
                f"splits={len(split_ids)}, merges={len(merge_pairs)}, proposed_k={proposed_k}"
            )

            if proposed_k == current_k:
                break
            current_k = proposed_k

        print(f"✓ Estimated number of components: {current_k} (DEEPDPM split-merge)\n")
        return current_k

    def _estimate_promptccd_components(self, features):
        features = np.asarray(features, dtype=np.float64)
        upper = self._component_search_bounds(features)
        current_k = int(np.clip(self.init_components, self.min_components, upper))
        prior = self._build_niw_prior(features)

        print(
            "\nSearching for active components (PROMPTCCD split-merge): "
            f"init={current_k}, min_components={self.min_components}, max_components={upper}"
        )

        for iteration in range(self.split_merge_max_iter):
            try:
                centers, labels = self._fit_gmm_assignments(features, current_k)
            except Exception as exc:
                print(f"  split-merge fit failed at iter={iteration}: {exc}")
                break

            split_ids = self._split_candidates(features, centers, labels, prior)
            merge_pairs = self._merge_candidates(features, centers, labels, prior, frozen_ids=split_ids)
            proposed_k = current_k + len(split_ids) - len(merge_pairs)
            proposed_k = int(np.clip(proposed_k, self.min_components, upper))

            print(
                f"  iter={iteration + 1}: k={current_k}, "
                f"splits={len(split_ids)}, merges={len(merge_pairs)}, proposed_k={proposed_k}"
            )

            if proposed_k == current_k:
                break
            current_k = proposed_k

        print(f"✓ Estimated number of components: {current_k} (PROMPTCCD split-merge)\n")
        return current_k

    def _estimate_silhouette_components(self, features):
        features = np.asarray(features, dtype=np.float64)
        upper = self._component_search_bounds(features)
        lower = max(int(self.min_components), 2)
        if upper < lower:
            fallback = int(np.clip(self.init_components, self.min_components, upper))
            print(
                "\nSearching for optimal number of components (SILHOUETTE)... "
                f"only one valid candidate available, fallback={fallback}\n"
            )
            return fallback

        print(
            "\nSearching for optimal number of components (SILHOUETTE)... "
            f"min_components={lower}, max_components={upper}"
        )

        best_n = None
        best_score = float("-inf")
        for n in range(lower, upper + 1):
            try:
                _, labels = self._fit_gmm_assignments(features, n)
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    score = float("-inf")
                else:
                    score = float(silhouette_score(features, labels, metric="euclidean"))
            except Exception as exc:
                print(f"  n_components={n}: silhouette failed ({exc})")
                continue

            print(f"  n_components={n}: SILHOUETTE={score:.6f}")
            if score > best_score:
                best_score = score
                best_n = int(n)

        if best_n is None:
            fallback = int(np.clip(self.init_components, self.min_components, upper))
            print(f"✓ Silhouette search failed; fallback to init_components={fallback}\n")
            return fallback

        print(f"✓ Optimal number of components: {best_n} (SILHOUETTE={best_score:.6f})\n")
        return best_n

    def _fit_dp_gmm(self, features):
        upper = self._component_search_bounds(features)

        print(
            f"\nSearching for active components (DP-GMM): "
            f"min_components={self.min_components}, max_components={upper}"
        )
        if self.dp_weight_concentration_prior is not None:
            print(f"  weight_concentration_prior={self.dp_weight_concentration_prior}")
        print(f"  active pruning: min_count={self.dp_min_count}, min_weight={self.dp_min_weight:.4f}")
        self.model = BayesianGaussianMixture(
            n_components=upper,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=self.dp_weight_concentration_prior,
            init_params="kmeans",
            random_state=self.random_state,
            max_iter=1000,
            reg_covar=1e-6,
        )
        self.model.fit(features)

        raw_pred = self.model.predict(features)
        active_ids, counts = np.unique(raw_pred, return_counts=True)
        order = np.argsort(active_ids)
        active_ids = active_ids[order]
        counts = counts[order]

        weights = self.model.weights_
        active_components = []
        pruned_components = []
        for comp_id, count in zip(active_ids.tolist(), counts.tolist()):
            weight = float(weights[int(comp_id)])
            comp_info = {"id": int(comp_id), "count": int(count), "weight": weight}
            if count >= self.dp_min_count and weight >= self.dp_min_weight:
                active_components.append(comp_info)
            else:
                pruned_components.append(comp_info)

        if not active_components:
            best_idx = int(np.argmax(counts))
            fallback_comp = {
                "id": int(active_ids[best_idx]),
                "count": int(counts[best_idx]),
                "weight": float(weights[int(active_ids[best_idx])]),
            }
            active_components = [fallback_comp]
            pruned_components = [x for x in pruned_components if x["id"] != fallback_comp["id"]]
            print(
                "  No components survived pruning; keeping the largest assigned component "
                f"{fallback_comp['id']} as a fallback."
            )

        self.active_component_ids = [item["id"] for item in active_components]
        self.active_component_map = {old: new for new, old in enumerate(self.active_component_ids)}
        self.component_fallback_map = self._build_component_fallback_map(upper)
        self.estimated_n_components = len(self.active_component_ids)
        self.init_components = self.estimated_n_components
        self.num_classes = self.estimated_n_components

        print("  Active DP-GMM components:")
        for item in active_components:
            print(f"    component={item['id']}: weight={item['weight']:.4f}, " f"assigned={item['count']}")
        if pruned_components:
            print("  Pruned DP-GMM components:")
            for item in pruned_components:
                fallback = self.component_fallback_map.get(item["id"], self.active_component_ids[0])
                print(
                    f"    component={item['id']}: weight={item['weight']:.4f}, "
                    f"assigned={item['count']} -> merge_to={fallback}"
                )
        print(f"✓ DP-GMM active components: {self.estimated_n_components}\n")

    def _build_component_fallback_map(self, total_components):
        if not self.active_component_ids:
            return {}

        active_set = set(self.active_component_ids)
        means = np.asarray(self.model.means_)
        fallback_map = {}
        anchor = self.active_component_ids[0]
        for comp_id in range(int(total_components)):
            if comp_id in active_set:
                fallback_map[comp_id] = comp_id
                continue
            comp_mean = means[comp_id]
            nearest_active = min(
                self.active_component_ids,
                key=lambda active_id: float(np.sum((comp_mean - means[active_id]) ** 2)),
                default=anchor,
            )
            fallback_map[comp_id] = int(nearest_active)
        return fallback_map

    def _remap_active_components(self, pred):
        if not self.active_component_map:
            return pred
        remapped = np.empty_like(pred)
        for comp in np.unique(pred):
            comp_int = int(comp)
            merged_comp = (
                self.component_fallback_map.get(comp_int, comp_int) if self.component_fallback_map else comp_int
            )
            target_idx = self.active_component_map.get(merged_comp)
            if target_idx is None:
                target_idx = 0
            remapped[pred == comp] = target_idx
        return remapped

    def _reset_active_component_state(self):
        self.active_component_map = None
        self.active_component_ids = None
        self.component_fallback_map = None

    def _estimate_auto_components(self, features):
        if self.criterion == "dp_gmm":
            self._fit_dp_gmm(features)
            return None
        if self.criterion == "silhouette":
            return self._estimate_silhouette_components(features)
        if self.criterion == "deepdpm":
            return self._estimate_deepdpm_components(features)
        if self.criterion == "promptccd":
            return self._estimate_promptccd_components(features)
        return self._find_optimal_components(features)

    @override
    def fit(self, features):
        self._reset_active_component_state()

        if self.auto_estimate:
            estimated_components = self._estimate_auto_components(features)
            if self.criterion == "dp_gmm":
                return
            self.estimated_n_components = estimated_components
            self.model = GaussianMixture(n_components=self.estimated_n_components, random_state=self.random_state)
            self.init_components = self.estimated_n_components
            self.num_classes = self.estimated_n_components

        self.model.fit(features)

    @override
    def _pre_predict(self, features):
        pred = self.model.predict(features)
        if self.auto_estimate and self.criterion == "dp_gmm":
            return self._remap_active_components(pred)
        return pred
