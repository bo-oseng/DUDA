import json
import os


from math import log
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI
from numpyro.infer import Trace_ELBO
import optax
from jax import random
import jax

from numpyro.infer.autoguide import AutoMultivariateNormal

from torch.utils.tensorboard import SummaryWriter

from numpy import save

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from collections import defaultdict

import copy
import pickle

from prettytable import PrettyTable

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving figures


from tqdm import tqdm


class MNGMMClassifier:

    def __init__(self, num_dim, num_classes, with_early_stop, use_pca=True):
        self.num_dim = num_dim
        self.num_classes = num_classes
        self.use_pca = use_pca
        self.pca = None
        self.scaler = None
        self.global_params = None
        self.label_offset = 0
        self.with_early_stop = with_early_stop
        self.class_order = None  # training_label -> original_class_id mapping

    def update_dir_infos(self, log_dir="logs/", save_dir="saved_models/"):
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir

    def init_parameters(
        self,
        n_epochs,
        lr,
        log_dir,
        save_dir,
        batch_size,
        increment=10,
        base=50,
        scaling_factor=1.2,
        use_correct_scaling_factor=True,
        early_stop_ratio=0,
        class_order=None,
    ):

        self.num_steps = n_epochs
        self.init_lr = lr
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir)
        self.increment = increment
        self.num_base = base
        self.scaling_factor = scaling_factor
        self.use_correct_scaling_factor = use_correct_scaling_factor
        self.early_stop_ratio = early_stop_ratio
        self.class_order = class_order

    def model(self, X, y=None, num_classes=2, global_params=None, **kwargs):
        num_features = X.shape[1]
        if global_params is None:
            class_means = numpyro.param("class_means", jnp.zeros((num_classes, num_features)))
            class_covs = numpyro.param("class_covs", jnp.stack([jnp.eye(num_features)] * num_classes))
        else:
            class_means = numpyro.param("class_means", global_params["class_means"])
            class_covs = numpyro.param("class_covs", global_params["class_covs"])

        with numpyro.plate("batch", X.shape[0], subsample_size=self.batch_size) as ind:
            X_batch = X[ind]
            y_batch = y[ind] if y is not None else None

            if y_batch is not None:
                base_dist = dist.MultivariateNormal(class_means[y_batch], class_covs[y_batch])
                numpyro.sample("obs", base_dist, obs=X_batch)

    def run_inference(self, X, y, test_X, test_y, log_prefix="", use_correct_scaling_factor=False):
        init_lr = self.init_lr
        scheduler = optax.join_schedules(
            schedules=[
                optax.linear_schedule(init_value=init_lr, end_value=init_lr * 10, transition_steps=100),
                optax.exponential_decay(init_value=init_lr * 10, transition_steps=500, decay_rate=0.85),
            ],
            boundaries=[100],
        )

        self.guide = lambda *args, **kwargs: None

        print("Initializing model")

        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(scheduler))
        self.svi = SVI(self.model, guide=self.guide, optim=optimizer, loss=Trace_ELBO())
        self.svi_state = self.svi.init(
            random.PRNGKey(0), X=X, y=y, num_classes=self.num_classes, global_params=self.global_params
        )
        last_state = None

        for step in tqdm(range(self.num_steps)):
            early_stop_flag, dets = self.calculate_metrics_on_covariances(
                self.svi.get_params(self.svi_state),
                increment=self.increment,
                use_correct_scaling_factor=use_correct_scaling_factor,
            )

            if self.with_early_stop & early_stop_flag & (last_state is not None) & (step > 1 / 3 * self.num_steps):

                self.svi_state = last_state

                early_stop_flag, dets = self.calculate_metrics_on_covariances(
                    self.svi.get_params(self.svi_state),
                    increment=self.increment,
                    use_correct_scaling_factor=use_correct_scaling_factor,
                )
                correct, total, acc = self.calculate_acc(self.svi.get_params(self.svi_state), X, y)
                correct_test, total_test, acc_test = self.calculate_acc(
                    self.svi.get_params(self.svi_state), test_X, test_y
                )

                self.writer.add_scalar(f"{log_prefix}/Accuracy/train", acc, step)

                self.writer.add_scalar(f"{log_prefix}/Accuracy/test", acc_test, step)

                self.writer.add_scalar(f"{log_prefix}/LastCovariance/det_0", dets[0].item(), step)

                self.writer.add_scalar(f"{log_prefix}/Covariance/det_0", dets[1].item(), step)

                print(
                    f"Step {step}: loss = {loss:.4f}, train_acc = {correct}/{total}, {acc:.2f}%,",
                    f" test_acc = {correct_test}/{total_test}, {acc_test:.2f}%, last_cov = {dets[0].item()}, cov = {dets[1].item()}, early_stop_flag = {early_stop_flag}",
                )

                break

            last_state = self.svi_state

            self.svi_state, loss = self.svi.update(
                self.svi_state, X=X, y=y, num_classes=self.num_classes, covs_dets=dets
            )

            self.writer.add_scalar(f"{log_prefix}/Loss/train", loss.item(), step)

            if step % 100 == 0:
                correct, total, acc = self.calculate_acc(self.svi.get_params(self.svi_state), X, y)
                correct_test, total_test, acc_test = self.calculate_acc(
                    self.svi.get_params(self.svi_state), test_X, test_y
                )

                self.writer.add_scalar(f"{log_prefix}/Accuracy/train", acc, step)

                self.writer.add_scalar(f"{log_prefix}/Accuracy/test", acc_test, step)

                self.writer.add_scalar(f"{log_prefix}/LastCovariance/det_0", dets[0].item(), step)

                self.writer.add_scalar(f"{log_prefix}/Covariance/det_1", dets[1].item(), step)

                print(
                    f"Step {step}: loss = {loss:.4f}, train_acc = {correct}/{total}, {acc:.2f}%,",
                    f" test_acc = {correct_test}/{total_test}, {acc_test:.2f}%, last_cov = {dets[0].item()}, cov = {dets[1].item()}",
                )

            if jnp.isnan(loss):
                print("Early stopping du to loss is NaN")
                self.svi_state = last_state
                break

            prev_loss = loss

        return self.svi.get_params(self.svi_state)

    def pre_processing(self, features, labels):
        if self.scaler is None:
            if self.use_pca:
                self.pca = PCA(n_components=self.num_dim, random_state=42)
                features = self.pca.fit_transform(features)
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            if self.pca is not None:
                features = self.pca.transform(features)
            features = self.scaler.transform(features)
        return features, labels

    def train(self, features, labels, test_features, test_labels, current_stage):
        features, labels = self.pre_processing(features, labels)
        test_features, test_labels = self.pre_processing(test_features, test_labels)

        labels = labels.astype(int)

        self.params = self.run_inference(
            jnp.array(features),
            jnp.array(labels),
            jnp.array(test_features),
            jnp.array(test_labels),
            log_prefix=f"stage_{current_stage}_Flearning",
            use_correct_scaling_factor=False,
        )

        pred_labels, log_probs, _, _ = self._predict(jnp.array(features), self.params)

        if self.global_params is not None:

            novel_idx = pred_labels >= self.label_offset

            print(f"Number of Novel Samples: {novel_idx.sum()} / {len(features)}")

            self.writer.add_scalar(f"Number/NovelSamples", novel_idx.sum().item(), current_stage)
            self.writer.add_scalar(f"Number/TotalSamples", len(features), current_stage)

            features = features[novel_idx]
            labels = labels[novel_idx]

            self.params = self.run_inference(
                jnp.array(features),
                jnp.array(labels),
                jnp.array(test_features),
                jnp.array(test_labels),
                log_prefix=f"stage_{current_stage}_Slearning",
                use_correct_scaling_factor=self.use_correct_scaling_factor,
            )

        self.global_params = copy.deepcopy(self.params)

    def calculate_acc(self, params, test_features, test_labels):
        pred_test_labels, _, _, _ = self._predict(jnp.array(test_features), params)
        correct = jnp.sum(pred_test_labels == test_labels).tolist()

        return correct, len(test_features), 100.0 * (correct / float(len(test_features)))

    def test(self, test_features, test_labels):
        test_features, test_labels = self.pre_processing(test_features, test_labels)
        pred_test_labels, _, _, _ = self._predict(jnp.array(test_features), self.params)
        correct = jnp.sum(pred_test_labels == test_labels).tolist()

        return correct, len(test_features), 100.0 * correct / float(len(test_features))

    def test_with_alpha(self, test_features, test_labels, dataset_name=""):
        """Test and return accuracy with alpha and confidence statistics"""
        test_features, test_labels = self.pre_processing(test_features, test_labels)
        pred_test_labels, log_probs, alpha, confidence = self._predict(jnp.array(test_features), self.params)

        correct_mask = pred_test_labels == test_labels
        incorrect_mask = ~correct_mask
        correct = jnp.sum(correct_mask).tolist()
        acc = 100.0 * correct / float(len(test_features))

        # Alpha statistics
        alpha_mean = float(jnp.mean(alpha))
        alpha_std = float(jnp.std(alpha))

        # Alpha for correct vs incorrect predictions
        if jnp.sum(correct_mask) > 0:
            alpha_correct_mean = float(jnp.mean(alpha[correct_mask]))
        else:
            alpha_correct_mean = 0.0

        if jnp.sum(incorrect_mask) > 0:
            alpha_incorrect_mean = float(jnp.mean(alpha[incorrect_mask]))
        else:
            alpha_incorrect_mean = 0.0

        # Alpha for old vs new class samples (ground truth)
        old_mask = test_labels < self.label_offset
        new_mask = test_labels >= self.label_offset

        if jnp.sum(old_mask) > 0:
            alpha_old_mean = float(jnp.mean(alpha[old_mask]))
        else:
            alpha_old_mean = 0.0

        if jnp.sum(new_mask) > 0:
            alpha_new_mean = float(jnp.mean(alpha[new_mask]))
        else:
            alpha_new_mean = 0.0

        alpha_stats = {
            "mean": alpha_mean,
            "std": alpha_std,
            "correct": alpha_correct_mean,
            "incorrect": alpha_incorrect_mean,
            "old_class": alpha_old_mean,
            "new_class": alpha_new_mean,
        }

        # Helper function for correct/incorrect stats
        def get_correct_incorrect_stats(values):
            if jnp.sum(correct_mask) > 0:
                val_correct = float(jnp.mean(values[correct_mask]))
            else:
                val_correct = 0.0
            if jnp.sum(incorrect_mask) > 0:
                val_incorrect = float(jnp.mean(values[incorrect_mask]))
            else:
                val_incorrect = 0.0
            return val_correct, val_incorrect

        # Method 1: Temperature Scaling
        temp_max_prob = confidence["temp_max_prob"]
        temp_entropy = confidence["temp_entropy"]
        temp_margin = confidence["temp_margin"]
        temp_correct, temp_incorrect = get_correct_incorrect_stats(temp_max_prob)
        temp_entropy_correct, temp_entropy_incorrect = get_correct_incorrect_stats(temp_entropy)
        temp_margin_correct, temp_margin_incorrect = get_correct_incorrect_stats(temp_margin)

        # Method 2: Raw Log Margin
        raw_log_margin = confidence["raw_log_margin"]
        raw_log_margin_norm = confidence["raw_log_margin_norm"]
        raw_log_conf_norm = confidence["raw_log_conf_norm"]
        raw_margin_correct, raw_margin_incorrect = get_correct_incorrect_stats(raw_log_margin)
        raw_margin_norm_correct, raw_margin_norm_incorrect = get_correct_incorrect_stats(raw_log_margin_norm)
        raw_conf_correct, raw_conf_incorrect = get_correct_incorrect_stats(raw_log_conf_norm)

        # Method 3: Mahalanobis Distance
        mahal_dist = confidence["mahal_dist"]
        mahal_conf = confidence["mahal_conf"]
        mahal_margin_norm = confidence["mahal_margin_norm"]
        mahal_dist_correct, mahal_dist_incorrect = get_correct_incorrect_stats(mahal_dist)
        mahal_conf_correct, mahal_conf_incorrect = get_correct_incorrect_stats(mahal_conf)
        mahal_margin_correct, mahal_margin_incorrect = get_correct_incorrect_stats(mahal_margin_norm)

        conf_stats = {
            # Method 1: Temperature Scaling
            "temp_max_prob_mean": float(jnp.mean(temp_max_prob)),
            "temp_max_prob_correct": temp_correct,
            "temp_max_prob_incorrect": temp_incorrect,
            "temp_entropy_mean": float(jnp.mean(temp_entropy)),
            "temp_entropy_correct": temp_entropy_correct,
            "temp_entropy_incorrect": temp_entropy_incorrect,
            "temp_margin_mean": float(jnp.mean(temp_margin)),
            "temp_margin_correct": temp_margin_correct,
            "temp_margin_incorrect": temp_margin_incorrect,
            "auto_temperature": float(confidence["auto_temperature"]),
            # Method 2: Raw Log Margin
            "raw_log_margin_mean": float(jnp.mean(raw_log_margin)),
            "raw_log_margin_correct": raw_margin_correct,
            "raw_log_margin_incorrect": raw_margin_incorrect,
            "raw_margin_norm_mean": float(jnp.mean(raw_log_margin_norm)),
            "raw_margin_norm_correct": raw_margin_norm_correct,
            "raw_margin_norm_incorrect": raw_margin_norm_incorrect,
            "raw_conf_norm_mean": float(jnp.mean(raw_log_conf_norm)),
            "raw_conf_norm_correct": raw_conf_correct,
            "raw_conf_norm_incorrect": raw_conf_incorrect,
            # Method 3: Mahalanobis Distance
            "mahal_dist_mean": float(jnp.mean(mahal_dist)),
            "mahal_dist_correct": mahal_dist_correct,
            "mahal_dist_incorrect": mahal_dist_incorrect,
            "mahal_conf_mean": float(jnp.mean(mahal_conf)),
            "mahal_conf_correct": mahal_conf_correct,
            "mahal_conf_incorrect": mahal_conf_incorrect,
            "mahal_margin_mean": float(jnp.mean(mahal_margin_norm)),
            "mahal_margin_correct": mahal_margin_correct,
            "mahal_margin_incorrect": mahal_margin_incorrect,
        }

        return correct, len(test_features), acc, alpha_stats, conf_stats

    def test_with_alpha_and_raw_values(self, test_features, test_labels, dataset_name=""):
        """Test and return accuracy with alpha, confidence statistics, AND raw values for histograms"""
        test_features, test_labels = self.pre_processing(test_features, test_labels)
        pred_test_labels, log_probs, alpha, confidence = self._predict(jnp.array(test_features), self.params)

        correct_mask = pred_test_labels == test_labels
        incorrect_mask = ~correct_mask
        correct = jnp.sum(correct_mask).tolist()
        acc = 100.0 * correct / float(len(test_features))

        # Return raw values for histogram plotting
        raw_values = {
            "correct_mask": np.array(correct_mask),
            "incorrect_mask": np.array(incorrect_mask),
            "temp_max_prob": np.array(confidence["temp_max_prob"]),
            "temp_margin": np.array(confidence["temp_margin"]),
            "raw_log_margin": np.array(confidence["raw_log_margin"]),
            "auto_temperature": float(confidence["auto_temperature"]),
        }

        return correct, len(test_features), acc, raw_values

    def plot_confidence_histograms(self, test_features, test_labels, current_stage, save_dir=None):
        """
        Plot histograms comparing confidence distributions for correct vs incorrect predictions.

        Creates 4 subplots:
        1. Temperature Scaling - max_prob (correct vs wrong)
        2. Temperature Scaling - margin (correct vs wrong)
        3. Raw Log Margin (correct vs wrong)
        4. Raw Log Margin - zoomed for wrong predictions
        """
        test_features_proc, test_labels_proc = self.pre_processing(test_features, test_labels)
        pred_labels, _, alpha, confidence = self._predict(jnp.array(test_features_proc), self.params)

        correct_mask = np.array(pred_labels == test_labels_proc)
        incorrect_mask = ~correct_mask

        # Extract values
        temp_max_prob = np.array(confidence["temp_max_prob"])
        temp_margin = np.array(confidence["temp_margin"])
        raw_log_margin = np.array(confidence["raw_log_margin"])
        auto_temp = float(confidence["auto_temperature"])

        n_correct = np.sum(correct_mask)
        n_incorrect = np.sum(incorrect_mask)

        if n_incorrect == 0:
            print("[Histogram] No incorrect predictions - skipping histogram generation")
            return

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Stage {current_stage}: Confidence Distribution (Correct vs Wrong)\n"
            f"Correct: {n_correct}, Wrong: {n_incorrect}, Acc: {100*n_correct/(n_correct+n_incorrect):.1f}%",
            fontsize=14,
            fontweight="bold",
        )

        # Color scheme
        correct_color = "#2ecc71"  # Green
        wrong_color = "#e74c3c"  # Red

        # ============================================================
        # Plot 1: Temperature Scaling - max_prob
        # ============================================================
        ax1 = axes[0, 0]

        correct_vals = temp_max_prob[correct_mask]
        wrong_vals = temp_max_prob[incorrect_mask]

        bins = np.linspace(0, 1, 31)

        ax1.hist(
            correct_vals,
            bins=bins,
            alpha=0.7,
            label=f"Correct (n={n_correct})",
            color=correct_color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax1.hist(
            wrong_vals,
            bins=bins,
            alpha=0.7,
            label=f"Wrong (n={n_incorrect})",
            color=wrong_color,
            edgecolor="white",
            linewidth=0.5,
        )

        ax1.axvline(
            np.mean(correct_vals),
            color=correct_color,
            linestyle="--",
            linewidth=2,
            label=f"Correct mean: {np.mean(correct_vals):.3f}",
        )
        ax1.axvline(
            np.mean(wrong_vals),
            color=wrong_color,
            linestyle="--",
            linewidth=2,
            label=f"Wrong mean: {np.mean(wrong_vals):.3f}",
        )

        ax1.set_xlabel("Temperature Scaled Max Probability", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.set_title(f"Method 1: Temperature Scaling (temp={auto_temp:.1f})\nmax_prob distribution", fontsize=12)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ============================================================
        # Plot 2: Temperature Scaling - margin
        # ============================================================
        ax2 = axes[0, 1]

        correct_vals = temp_margin[correct_mask]
        wrong_vals = temp_margin[incorrect_mask]

        bins = np.linspace(0, 1, 31)

        ax2.hist(
            correct_vals,
            bins=bins,
            alpha=0.7,
            label=f"Correct (n={n_correct})",
            color=correct_color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax2.hist(
            wrong_vals,
            bins=bins,
            alpha=0.7,
            label=f"Wrong (n={n_incorrect})",
            color=wrong_color,
            edgecolor="white",
            linewidth=0.5,
        )

        ax2.axvline(
            np.mean(correct_vals),
            color=correct_color,
            linestyle="--",
            linewidth=2,
            label=f"Correct mean: {np.mean(correct_vals):.3f}",
        )
        ax2.axvline(
            np.mean(wrong_vals),
            color=wrong_color,
            linestyle="--",
            linewidth=2,
            label=f"Wrong mean: {np.mean(wrong_vals):.3f}",
        )

        ax2.set_xlabel("Temperature Scaled Margin (top1 - top2)", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Method 1: Temperature Scaling\nmargin distribution", fontsize=12)
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ============================================================
        # Plot 3: Raw Log Margin - full range
        # ============================================================
        ax3 = axes[1, 0]

        correct_vals = raw_log_margin[correct_mask]
        wrong_vals = raw_log_margin[incorrect_mask]

        # Dynamic binning based on data range
        all_vals = np.concatenate([correct_vals, wrong_vals])
        min_val, max_val = np.min(all_vals), np.max(all_vals)
        bins = np.linspace(min_val, max_val, 41)

        ax3.hist(
            correct_vals,
            bins=bins,
            alpha=0.7,
            label=f"Correct (n={n_correct})",
            color=correct_color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax3.hist(
            wrong_vals,
            bins=bins,
            alpha=0.7,
            label=f"Wrong (n={n_incorrect})",
            color=wrong_color,
            edgecolor="white",
            linewidth=0.5,
        )

        ax3.axvline(
            np.mean(correct_vals),
            color=correct_color,
            linestyle="--",
            linewidth=2,
            label=f"Correct mean: {np.mean(correct_vals):.1f}",
        )
        ax3.axvline(
            np.mean(wrong_vals),
            color=wrong_color,
            linestyle="--",
            linewidth=2,
            label=f"Wrong mean: {np.mean(wrong_vals):.1f}",
        )

        ax3.set_xlabel("Raw Log Margin (log_prob[top1] - log_prob[top2])", fontsize=11)
        ax3.set_ylabel("Count", fontsize=11)
        ax3.set_title("Method 2: Raw Log Margin\nFull distribution", fontsize=12)
        ax3.legend(loc="upper right", fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ============================================================
        # Plot 4: Raw Log Margin - zoomed to low values (where wrong predictions cluster)
        # ============================================================
        ax4 = axes[1, 1]

        # Focus on lower margin values where wrong predictions tend to be
        percentile_95 = np.percentile(wrong_vals, 95) if len(wrong_vals) > 0 else 100
        zoom_max = max(percentile_95 * 1.5, 100)  # At least show up to 100

        bins = np.linspace(0, zoom_max, 41)

        ax4.hist(
            correct_vals[correct_vals <= zoom_max],
            bins=bins,
            alpha=0.7,
            label=f"Correct",
            color=correct_color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax4.hist(
            wrong_vals[wrong_vals <= zoom_max],
            bins=bins,
            alpha=0.7,
            label=f"Wrong",
            color=wrong_color,
            edgecolor="white",
            linewidth=0.5,
        )

        ax4.axvline(
            np.mean(wrong_vals),
            color=wrong_color,
            linestyle="--",
            linewidth=2,
            label=f"Wrong mean: {np.mean(wrong_vals):.1f}",
        )

        ax4.set_xlabel("Raw Log Margin", fontsize=11)
        ax4.set_ylabel("Count", fontsize=11)
        ax4.set_title(f"Method 2: Raw Log Margin\nZoomed view (0-{zoom_max:.0f})", fontsize=12)
        ax4.legend(loc="upper right", fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_dir is None:
            save_dir = self.save_dir.rsplit("stage", 1)[0]

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"confidence_histogram_stage{current_stage}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved confidence histogram to {save_path}")

        # Also print summary statistics
        print(f"\n[Histogram Summary - Stage {current_stage}]")
        print(f"  Temperature Scaling max_prob:")
        print(
            f"    Correct: mean={np.mean(temp_max_prob[correct_mask]):.3f}, std={np.std(temp_max_prob[correct_mask]):.3f}"
        )
        print(
            f"    Wrong:   mean={np.mean(temp_max_prob[incorrect_mask]):.3f}, std={np.std(temp_max_prob[incorrect_mask]):.3f}"
        )
        print(f"  Raw Log Margin:")
        print(
            f"    Correct: mean={np.mean(raw_log_margin[correct_mask]):.1f}, std={np.std(raw_log_margin[correct_mask]):.1f}"
        )
        print(
            f"    Wrong:   mean={np.mean(raw_log_margin[incorrect_mask]):.1f}, std={np.std(raw_log_margin[incorrect_mask]):.1f}"
        )

        return save_path

    def test_per_class(self, test_features, test_labels):
        """Test and return per-class accuracy with alpha and confidence statistics"""
        test_features, test_labels = self.pre_processing(test_features, test_labels)
        pred_test_labels, _, alpha, confidence = self._predict(jnp.array(test_features), self.params)

        # Calculate per-class accuracy
        unique_labels = jnp.unique(test_labels)
        per_class_acc = {}

        for label in unique_labels:
            label_int = int(label)
            mask = test_labels == label
            correct_mask = (pred_test_labels == test_labels) & mask
            incorrect_mask = mask & (pred_test_labels != test_labels)
            correct = jnp.sum(correct_mask).tolist()
            total = jnp.sum(mask).tolist()
            acc = 100.0 * correct / total if total > 0 else 0.0

            # Alpha stats for this class
            if jnp.sum(mask) > 0:
                alpha_mean = float(jnp.mean(alpha[mask]))
                alpha_std = float(jnp.std(alpha[mask]))
            else:
                alpha_mean = 0.0
                alpha_std = 0.0

            # Helper to get mean for a subset
            def safe_mean(arr, m):
                if jnp.sum(m) > 0:
                    return float(jnp.mean(arr[m]))
                return 0.0

            # All three methods for this class
            conf_stats = {
                # Method 1: Temperature Scaling
                "temp_max_prob": safe_mean(confidence["temp_max_prob"], mask),
                "temp_max_prob_correct": safe_mean(confidence["temp_max_prob"], correct_mask),
                "temp_max_prob_incorrect": safe_mean(confidence["temp_max_prob"], incorrect_mask),
                "temp_entropy": safe_mean(confidence["temp_entropy"], mask),
                "temp_margin": safe_mean(confidence["temp_margin"], mask),
                # Method 2: Raw Log Margin
                "raw_log_margin": safe_mean(confidence["raw_log_margin"], mask),
                "raw_log_margin_correct": safe_mean(confidence["raw_log_margin"], correct_mask),
                "raw_log_margin_incorrect": safe_mean(confidence["raw_log_margin"], incorrect_mask),
                "raw_margin_norm": safe_mean(confidence["raw_log_margin_norm"], mask),
                "raw_conf_norm": safe_mean(confidence["raw_log_conf_norm"], mask),
                # Method 3: Mahalanobis Distance
                "mahal_dist": safe_mean(confidence["mahal_dist"], mask),
                "mahal_dist_correct": safe_mean(confidence["mahal_dist"], correct_mask),
                "mahal_dist_incorrect": safe_mean(confidence["mahal_dist"], incorrect_mask),
                "mahal_conf": safe_mean(confidence["mahal_conf"], mask),
                "mahal_margin": safe_mean(confidence["mahal_margin_norm"], mask),
            }

            per_class_acc[label_int] = {
                "correct": correct,
                "total": total,
                "acc": acc,
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
                "conf_stats": conf_stats,
            }

        return per_class_acc

    def analyze_misclassifications(self, test_features, test_labels, class_names=None, top_n=5):
        """Analyze misclassified samples: which classes they were predicted as and their alpha/confidence values"""
        test_features, test_labels = self.pre_processing(test_features, test_labels)
        pred_labels, _, alpha, confidence = self._predict(jnp.array(test_features), self.params)

        # Get all confidence metrics
        temp_max_prob = confidence["temp_max_prob"]
        raw_log_margin = confidence["raw_log_margin"]
        mahal_dist = confidence["mahal_dist"]

        # Find misclassified samples
        misclassified_mask = pred_labels != test_labels

        if jnp.sum(misclassified_mask) == 0:
            print("\n[Misclassification Analysis] No misclassifications found!")
            return {}

        print(f"\n[Misclassification Analysis] Total errors: {int(jnp.sum(misclassified_mask))}")

        # Analyze per ground-truth class
        unique_labels = jnp.unique(test_labels)
        misclass_analysis = {}

        for gt_label in unique_labels:
            gt_label_int = int(gt_label)
            gt_mask = test_labels == gt_label
            error_mask = gt_mask & misclassified_mask

            num_errors = int(jnp.sum(error_mask))
            if num_errors == 0:
                continue

            # Get predicted labels, alpha, and all confidences for errors
            error_preds = pred_labels[error_mask]
            error_alphas = alpha[error_mask]
            error_temp_confs = temp_max_prob[error_mask]
            error_raw_margins = raw_log_margin[error_mask]
            error_mahal_dists = mahal_dist[error_mask]

            # Count predictions per class
            pred_counts = {}
            for i, pred in enumerate(error_preds):
                pred_int = int(pred)
                if pred_int not in pred_counts:
                    pred_counts[pred_int] = {
                        "count": 0,
                        "alphas": [],
                        "temp_confs": [],
                        "raw_margins": [],
                        "mahal_dists": [],
                    }
                pred_counts[pred_int]["count"] += 1
                pred_counts[pred_int]["alphas"].append(float(error_alphas[i]))
                pred_counts[pred_int]["temp_confs"].append(float(error_temp_confs[i]))
                pred_counts[pred_int]["raw_margins"].append(float(error_raw_margins[i]))
                pred_counts[pred_int]["mahal_dists"].append(float(error_mahal_dists[i]))

            # Sort by count
            sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1]["count"], reverse=True)

            # Determine old/new tag for ground truth
            if gt_label_int < self.label_offset:
                gt_tag = "old"
            else:
                gt_tag = "new"

            # Get class name
            if class_names and gt_label_int < len(class_names):
                gt_name = class_names[gt_label_int]
            else:
                gt_name = f"class_{gt_label_int}"

            print(f"\n  {gt_name} ({gt_tag}, {num_errors} errors):")

            for pred_label, info in sorted_preds[:top_n]:
                count = info["count"]
                alpha_mean = np.mean(info["alphas"])
                alpha_std = np.std(info["alphas"]) if len(info["alphas"]) > 1 else 0.0
                temp_conf_mean = np.mean(info["temp_confs"])
                raw_margin_mean = np.mean(info["raw_margins"])
                mahal_dist_mean = np.mean(info["mahal_dists"])

                # Determine old/new for predicted class
                if pred_label < self.label_offset:
                    pred_tag = "old"
                else:
                    pred_tag = "new"

                if class_names and pred_label < len(class_names):
                    pred_name = class_names[pred_label]
                else:
                    pred_name = f"class_{pred_label}"

                # Confusion type
                if gt_tag == "old" and pred_tag == "old":
                    confusion_type = "old→old"
                elif gt_tag == "old" and pred_tag == "new":
                    confusion_type = "old→new"
                elif gt_tag == "new" and pred_tag == "old":
                    confusion_type = "new→old"
                else:
                    confusion_type = "new→new"

                print(
                    f"    → {pred_name:15s}: {count:3d} samples, α={alpha_mean:.3f}±{alpha_std:.3f} "
                    f"| T:{temp_conf_mean:.3f} R:{raw_margin_mean:.1f} M:{mahal_dist_mean:.1f} [{confusion_type}]"
                )

            misclass_analysis[gt_label_int] = {
                "total_errors": num_errors,
                "predictions": sorted_preds,
            }

        return misclass_analysis

    # output the acc of training data
    def _predict(self, X, params, temperature=1.0):
        class_means = params["class_means"]
        class_covs = params["class_covs"]
        num_classes = class_means.shape[0]
        log_probs = []
        mahal_dists = []

        for i in range(num_classes):
            mvn = dist.MultivariateNormal(class_means[i], class_covs[i])
            log_probs.append(mvn.log_prob(X))

            # Mahalanobis distance: (x - μ)^T Σ^(-1) (x - μ)
            diff = X - class_means[i]
            cov_inv = jnp.linalg.inv(class_covs[i])
            mahal = jnp.sqrt(jnp.sum(diff @ cov_inv * diff, axis=-1))
            mahal_dists.append(mahal)

        log_probs = jnp.stack(log_probs, axis=-1)
        mahal_dists = jnp.stack(mahal_dists, axis=-1)
        pred_labels = jnp.argmax(log_probs, axis=-1)

        # ============================================================
        # Method 1: Temperature Scaling
        # Auto temperature based on log_prob range
        # ============================================================
        log_max = jnp.max(log_probs, axis=-1, keepdims=True)
        log_min = jnp.min(log_probs, axis=-1, keepdims=True)
        log_range = jnp.mean(log_max - log_min)
        auto_temp = jnp.maximum(log_range / 10.0, 1.0)

        probs_temp = jax.nn.softmax(log_probs / auto_temp, axis=-1)
        max_prob_temp = jnp.max(probs_temp, axis=-1)
        entropy_temp = -jnp.sum(probs_temp * jnp.log(probs_temp + 1e-10), axis=-1)
        # Normalize entropy to [0, 1] (max entropy = log(num_classes))
        entropy_temp_norm = entropy_temp / jnp.log(num_classes)
        sorted_probs_temp = jnp.sort(probs_temp, axis=-1)
        margin_temp = sorted_probs_temp[:, -1] - sorted_probs_temp[:, -2]

        # ============================================================
        # Method 2: Raw Log Margin (no softmax)
        # ============================================================
        sorted_log = jnp.sort(log_probs, axis=-1)
        raw_log_margin = sorted_log[:, -1] - sorted_log[:, -2]  # always positive
        # Normalize: sigmoid to [0, 1], scale by typical range
        raw_log_margin_norm = jax.nn.sigmoid(raw_log_margin / 100.0)  # 100 is typical scale

        # Raw max log prob (relative to mean)
        raw_max_log = jnp.max(log_probs, axis=-1)
        raw_mean_log = jnp.mean(log_probs, axis=-1)
        raw_log_conf = raw_max_log - raw_mean_log  # how much better than average
        raw_log_conf_norm = jax.nn.sigmoid(raw_log_conf / 100.0)

        # ============================================================
        # Method 3: Mahalanobis Distance
        # Lower distance = more confident
        # ============================================================
        min_mahal = jnp.min(mahal_dists, axis=-1)  # distance to nearest class
        sorted_mahal = jnp.sort(mahal_dists, axis=-1)
        mahal_margin = sorted_mahal[:, 1] - sorted_mahal[:, 0]  # gap between 1st and 2nd nearest

        # Confidence from Mahalanobis: use negative distance (closer = higher confidence)
        # Normalize using sigmoid (typical mahal dist in high-dim is sqrt(dim) ~ 17 for 290d)
        mahal_conf = jax.nn.sigmoid(-min_mahal / 10.0 + 2.0)  # shift so typical values are around 0.5
        mahal_margin_norm = jax.nn.sigmoid(mahal_margin / 5.0)  # margin normalized

        confidence = {
            # Method 1: Temperature Scaling
            "temp_max_prob": max_prob_temp,
            "temp_entropy": entropy_temp_norm,
            "temp_margin": margin_temp,
            "auto_temperature": auto_temp,
            # Method 2: Raw Log Margin
            "raw_log_margin": raw_log_margin,
            "raw_log_margin_norm": raw_log_margin_norm,
            "raw_log_conf": raw_log_conf,
            "raw_log_conf_norm": raw_log_conf_norm,
            # Method 3: Mahalanobis Distance
            "mahal_dist": min_mahal,
            "mahal_margin": mahal_margin,
            "mahal_conf": mahal_conf,
            "mahal_margin_norm": mahal_margin_norm,
        }

        # Calculate alpha (confidence/uncertainty)
        # alpha: old vs new class confidence
        # high alpha = confident it's old class, low alpha = confident it's new class
        if self.label_offset > 0 and self.label_offset < log_probs.shape[-1]:
            old_logits = log_probs[:, : self.label_offset]
            new_logits = log_probs[:, self.label_offset :]

            old_max = jnp.max(old_logits, axis=-1)
            new_max = jnp.max(new_logits, axis=-1)

            old_new_margin = old_max - new_max
            alpha = jax.nn.sigmoid(old_new_margin / temperature)
        else:
            # No label_offset set or only one group exists
            alpha = jnp.ones(X.shape[0]) * 0.5  # neutral

        return pred_labels, log_probs, alpha, confidence

    def _set_label_offset(self, label_offset):
        self.label_offset = label_offset

    def _correct_scaling_factors(self, n, total):
        return jnp.sqrt((n) / (total + n))

    def calculate_metrics_on_covariances(self, params, increment, use_correct_scaling_factor):
        class_covs = params["class_covs"]

        early_stop_flag = False
        if self.global_params is None:
            return early_stop_flag, [jnp.ones(1), jnp.ones(1)]

        global_class_covs = self.global_params["class_covs"]
        if not use_correct_scaling_factor:
            dets = [
                (jax.vmap(jnp.linalg.det)(global_class_covs[: self.num_base])).mean(),
                (jax.vmap(jnp.linalg.det)(class_covs[self.label_offset : self.label_offset + increment])).mean(),
            ]
            scaling_factor = (self.label_offset + increment) / self.num_base
        else:
            dets = [
                (jax.vmap(jnp.linalg.det)(global_class_covs[: self.label_offset])).mean(),
                (jax.vmap(jnp.linalg.det)(class_covs[self.label_offset : self.label_offset + increment])).mean(),
            ]
            scaling_factor = self._correct_scaling_factors(increment, self.label_offset)

        if not jnp.isnan(dets[0]):
            if (dets[0] > 1) & (dets[1] > scaling_factor * dets[0]):
                early_stop_flag = True
            if (dets[0] < 1) & (dets[1] < dets[0] / scaling_factor):
                early_stop_flag = True

        return early_stop_flag, dets

    def run(self, features, labels, test_features, test_labels, current_stage, testing_set):

        self.train(features, labels, test_features, test_labels, current_stage)

        # Collect all test results first
        all_results = {}

        # Test All
        correct, total, acc, alpha_stats, conf_stats = self.test_with_alpha(
            testing_set["test_all"]._x, testing_set["test_all"]._y, "All"
        )
        all_results["All"] = {"correct": correct, "total": total, "acc": acc, "alpha": alpha_stats, "conf": conf_stats}
        self.writer.add_scalar(f"Test/Accuracy/All", acc, current_stage)

        # Test Old
        correct, total, acc, alpha_stats, conf_stats = self.test_with_alpha(
            testing_set["test_old"]._x, testing_set["test_old"]._y, "Old"
        )
        all_results["Old"] = {"correct": correct, "total": total, "acc": acc, "alpha": alpha_stats, "conf": conf_stats}
        self.writer.add_scalar(f"Test/Accuracy/Old", acc, current_stage)

        # Test Novel
        correct, total, acc, alpha_stats, conf_stats = self.test_with_alpha(test_features, test_labels, "Novel")
        all_results["Novel"] = {
            "correct": correct,
            "total": total,
            "acc": acc,
            "alpha": alpha_stats,
            "conf": conf_stats,
        }
        self.writer.add_scalar(f"Test/Accuracy/Novel", acc, current_stage)

        # Per-session
        if "session_tests" in testing_set:
            for session_idx, session_test in enumerate(testing_set["session_tests"]):
                correct, total, acc, alpha_stats, conf_stats = self.test_with_alpha(
                    session_test._x, session_test._y, f"S{session_idx}"
                )
                all_results[f"S{session_idx}"] = {
                    "correct": correct,
                    "total": total,
                    "acc": acc,
                    "alpha": alpha_stats,
                    "conf": conf_stats,
                }
                self.writer.add_scalar(f"Test/Accuracy/S{session_idx}", acc, current_stage)

        # ============================================================
        # Table 1: Basic Accuracy + Alpha
        # ============================================================
        table = PrettyTable(["TestSet", "Correct", "Samples", "Accuracy", "α_mean", "α_std", "α_correct", "α_wrong"])
        table.float_format = ".2f"

        for name, r in all_results.items():
            table.add_row(
                [
                    name,
                    r["correct"],
                    r["total"],
                    r["acc"],
                    f"{r['alpha']['mean']:.3f}",
                    f"{r['alpha']['std']:.3f}",
                    f"{r['alpha']['correct']:.3f}",
                    f"{r['alpha']['incorrect']:.3f}",
                ]
            )
        print(table)

        # ============================================================
        # Table 2: Method 1 - Temperature Scaling
        # ============================================================
        print(f"\n[Method 1: Temperature Scaling] (auto_temp={all_results['All']['conf']['auto_temperature']:.1f})")
        temp_table = PrettyTable(["TestSet", "max_prob", "correct", "wrong", "entropy", "margin", "Δ(c-w)"])
        temp_table.float_format = ".3f"

        for name, r in all_results.items():
            c = r["conf"]
            delta = c["temp_max_prob_correct"] - c["temp_max_prob_incorrect"]
            temp_table.add_row(
                [
                    name,
                    f"{c['temp_max_prob_mean']:.3f}",
                    f"{c['temp_max_prob_correct']:.3f}",
                    f"{c['temp_max_prob_incorrect']:.3f}",
                    f"{c['temp_entropy_mean']:.3f}",
                    f"{c['temp_margin_mean']:.3f}",
                    f"{delta:+.3f}" if c["temp_max_prob_incorrect"] > 0 else "N/A",
                ]
            )
        print(temp_table)

        # ============================================================
        # Table 3: Method 2 - Raw Log Margin
        # ============================================================
        print("\n[Method 2: Raw Log Margin] (no softmax, raw log-prob difference)")
        raw_table = PrettyTable(["TestSet", "margin", "correct", "wrong", "margin_norm", "conf_norm", "Δ(c-w)"])
        raw_table.float_format = ".1f"

        for name, r in all_results.items():
            c = r["conf"]
            delta = c["raw_log_margin_correct"] - c["raw_log_margin_incorrect"]
            raw_table.add_row(
                [
                    name,
                    f"{c['raw_log_margin_mean']:.1f}",
                    f"{c['raw_log_margin_correct']:.1f}",
                    f"{c['raw_log_margin_incorrect']:.1f}",
                    f"{c['raw_margin_norm_mean']:.3f}",
                    f"{c['raw_conf_norm_mean']:.3f}",
                    f"{delta:+.1f}" if c["raw_log_margin_incorrect"] > 0 else "N/A",
                ]
            )
        print(raw_table)

        # ============================================================
        # Table 4: Method 3 - Mahalanobis Distance
        # ============================================================
        print("\n[Method 3: Mahalanobis Distance] (lower dist = more confident)")
        mahal_table = PrettyTable(["TestSet", "dist", "correct", "wrong", "conf", "margin", "Δ(w-c)"])
        mahal_table.float_format = ".2f"

        for name, r in all_results.items():
            c = r["conf"]
            # For Mahalanobis, lower is better, so delta = wrong - correct (positive = good)
            delta = c["mahal_dist_incorrect"] - c["mahal_dist_correct"]
            mahal_table.add_row(
                [
                    name,
                    f"{c['mahal_dist_mean']:.2f}",
                    f"{c['mahal_dist_correct']:.2f}",
                    f"{c['mahal_dist_incorrect']:.2f}",
                    f"{c['mahal_conf_mean']:.3f}",
                    f"{c['mahal_margin_mean']:.3f}",
                    f"{delta:+.2f}" if c["mahal_dist_incorrect"] > 0 else "N/A",
                ]
            )
        print(mahal_table)

        # ============================================================
        # Summary: Which method separates correct/wrong best?
        # ============================================================
        print("\n[Summary: Correct vs Wrong Separation]")
        print("  Good separation = model knows when it's uncertain")
        print("  Method 1 (Temp): Δ(c-w) > 0 means correct predictions have higher confidence")
        print("  Method 2 (Raw):  Δ(c-w) > 0 means correct predictions have larger margin")
        print("  Method 3 (Mahal): Δ(w-c) > 0 means wrong predictions are farther from class center")

        # ============================================================
        # Generate Confidence Histograms
        # ============================================================
        self.plot_confidence_histograms(testing_set["test_all"]._x, testing_set["test_all"]._y, current_stage)

        # ============================================================
        # Per-class Accuracy
        # ============================================================
        print("\nPer-class Accuracy:")
        per_class_acc = self.test_per_class(testing_set["test_all"]._x, testing_set["test_all"]._y)

        # CDD11 class names
        classes = [
            "clear",
            "haze",
            "haze_rain",
            "haze_snow",
            "low",
            "low_haze",
            "low_haze_rain",
            "low_haze_snow",
            "low_rain",
            "low_snow",
            "rain",
            "snow",
        ]
        class_names = {idx: cls for idx, cls in enumerate(classes)}

        if self.class_order is not None:
            print(f"  Class order mapping (training → original): {self.class_order}")

        # Sort by training label
        for train_label in sorted(per_class_acc.keys()):
            r = per_class_acc[train_label]
            correct, total, acc = r["correct"], r["total"], r["acc"]
            alpha_mean, alpha_std = r["alpha_mean"], r["alpha_std"]
            cs = r["conf_stats"]

            # Stage info
            if train_label < self.num_base:
                stage_info = "stage0"
            else:
                stage_num = 1 + (train_label - self.num_base) // self.increment
                stage_info = f"stage{stage_num}"

            old_new_tag = "old" if train_label < self.label_offset else "new"

            # Get class name
            if self.class_order is not None and train_label < len(self.class_order):
                orig_class_id = self.class_order[train_label]
                class_name = class_names.get(orig_class_id, f"class_{orig_class_id}")
            else:
                class_name = class_names.get(train_label, f"class_{train_label}")

            print(
                f"  {train_label:2d} → {class_name:15s}: {acc:6.2f}% ({correct:4d}/{total:4d}) "
                f"[{stage_info}/{old_new_tag}] α={alpha_mean:.3f}±{alpha_std:.3f} "
                f"T:{cs['temp_max_prob']:.3f} R:{cs['raw_log_margin']:.0f} M:{cs['mahal_dist']:.1f}"
            )

            self.writer.add_scalar(f"Test/Alpha/Class_{train_label}_mean", alpha_mean, current_stage)

        print()

        # Misclassification analysis
        if self.class_order is not None:
            class_names_by_train_label = []
            for train_label in range(len(self.class_order)):
                orig_id = self.class_order[train_label]
                class_names_by_train_label.append(class_names.get(orig_id, f"class_{orig_id}"))
            self.analyze_misclassifications(
                testing_set["test_all"]._x, testing_set["test_all"]._y, class_names=class_names_by_train_label
            )

        # save the class means , covariances and supports to numpy files
        save(f"{self.save_dir}class_means.npy", np.array(self.params["class_means"]))
        save(f"{self.save_dir}class_covariances.npy", np.array(self.params["class_covs"]))

        # Save PCA model for reproducibility (only save once from the saved_models directory)
        if self.pca is not None:
            # Extract the base saved_models directory (remove stage prefix)
            saved_models_dir = self.save_dir.rsplit("stage", 1)[0]
            pca_path = f"{saved_models_dir}pca_model.pkl"
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)
            print(f"Saved PCA model to {pca_path}")

            scaler_path = f"{saved_models_dir}scaler_model.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            print(f"✓ Saved StandardScaler to {scaler_path}")

            c2o = {str(i): int(orig_id) for i, orig_id in enumerate(self.class_order)}
            o2c = {str(int(orig_id)): i for i, orig_id in enumerate(self.class_order)}

            mapping_data = {
                "classifier2orig": c2o,
                "orig2classifier": o2c,
                "class_order": [int(x) for x in self.class_order],
            }

            json_path = os.path.join(saved_models_dir, "class_mappings.json")

            # 3. JSON 파일 저장
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, indent=4, ensure_ascii=False)

            print(f"Saved class mappings to {json_path}")
