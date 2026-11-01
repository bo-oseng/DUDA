"""VB-CGCD classifier: one Gaussian per class fitted by stochastic variational inference.

Adapted from VB-CGCD. After each stage the class means mu_c and covariances Sigma_c are saved as
``<save_dir>class_means.npy`` / ``<save_dir>class_covariances.npy`` (e.g. ``saved_models/stage1class_means.npy``),
together with the PCA, the standardization, and the class mapping used by the restoration model.
"""

import copy
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class VBCGCDClassifier:
    def __init__(self, num_dim, num_classes, with_early_stop=True, use_pca=True):
        self.num_dim = num_dim
        self.num_classes = num_classes
        self.with_early_stop = with_early_stop
        self.use_pca = use_pca
        self.pca = None
        self.scaler = None
        self.global_params = None
        self.label_offset = 0
        self.stage_offsets = [0]
        self.stage_increments = [num_classes]

    def init_parameters(
        self,
        n_epochs,
        lr,
        log_dir,
        save_dir,
        batch_size,
        increment,
        base,
        use_correct_scaling_factor,
        class_order,
        class_names,
    ):
        self.num_steps = n_epochs
        self.init_lr = lr
        self.batch_size = batch_size
        self.increment = increment
        self.num_base = base
        self.use_correct_scaling_factor = use_correct_scaling_factor
        self.class_order = class_order
        self.class_names = {int(k): str(v) for k, v in class_names.items()}
        self.set_stage_structure([0], [base])
        self.update_dir_infos(log_dir, save_dir)

    def update_dir_infos(self, log_dir, save_dir):
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir

    def set_stage_structure(self, stage_offsets, stage_increments):
        self.stage_offsets = [int(x) for x in stage_offsets]
        self.stage_increments = [int(x) for x in stage_increments]

    def set_label_offset(self, label_offset):
        self.label_offset = label_offset

    # ------------------------------------------------------------------ inference
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
                numpyro.sample("obs", dist.MultivariateNormal(class_means[y_batch], class_covs[y_batch]), obs=X_batch)

    def covariance_growth(self, params, use_correct_scaling_factor):
        """Stop when the covariance scale of the new classes outgrows that of the known classes."""
        if self.global_params is None:
            return False, [jnp.ones(1), jnp.ones(1)]
        old_covs = self.global_params["class_covs"]
        new_covs = params["class_covs"][self.label_offset : self.label_offset + self.increment]
        if not use_correct_scaling_factor:
            dets = [
                jax.vmap(jnp.linalg.det)(old_covs[: self.num_base]).mean(),
                jax.vmap(jnp.linalg.det)(new_covs).mean(),
            ]
            scaling_factor = (self.label_offset + self.increment) / self.num_base
        else:
            dets = [
                jax.vmap(jnp.linalg.det)(old_covs[: self.label_offset]).mean(),
                jax.vmap(jnp.linalg.det)(new_covs).mean(),
            ]
            scaling_factor = jnp.sqrt(self.increment / (self.label_offset + self.increment))
        stop = False
        if not jnp.isnan(dets[0]):
            if (dets[0] > 1) & (dets[1] > scaling_factor * dets[0]):
                stop = True
            if (dets[0] < 1) & (dets[1] < dets[0] / scaling_factor):
                stop = True
        return stop, dets

    def run_inference(self, X, y, test_X, test_y, log_prefix, use_correct_scaling_factor):
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(init_value=self.init_lr, end_value=self.init_lr * 10, transition_steps=100),
                optax.exponential_decay(init_value=self.init_lr * 10, transition_steps=500, decay_rate=0.85),
            ],
            boundaries=[100],
        )
        svi = SVI(
            self.model,
            guide=lambda *a, **k: None,
            optim=numpyro.optim.optax_to_numpyro(optax.adam(schedule)),
            loss=Trace_ELBO(),
        )
        state = svi.init(random.PRNGKey(0), X=X, y=y, num_classes=self.num_classes, global_params=self.global_params)
        last_state = None
        for step in tqdm(range(self.num_steps)):
            stop, dets = self.covariance_growth(svi.get_params(state), use_correct_scaling_factor)
            if self.with_early_stop & stop & (last_state is not None) & (step > 1 / 3 * self.num_steps):
                state = last_state
                print(f"Step {step}: early stop (old cov det {dets[0].item():.4g}, new cov det {dets[1].item():.4g})")
                break
            last_state = state
            state, loss = svi.update(state, X=X, y=y, num_classes=self.num_classes, covs_dets=dets)
            self.writer.add_scalar(f"{log_prefix}/Loss/train", loss.item(), step)
            if step % 100 == 0:
                params = svi.get_params(state)
                print(
                    f"Step {step}: loss = {loss:.4f}, train acc = {self.accuracy(params, X, y):.2f}%, "
                    f"test acc = {self.accuracy(params, test_X, test_y):.2f}%"
                )
            if jnp.isnan(loss):
                print("Early stopping due to NaN loss")
                state = last_state
                break
        return svi.get_params(state)

    def preprocess(self, features):
        """PCA to ``num_dim`` and standardization, fitted on the base-stage data."""
        if self.scaler is None:
            if self.use_pca:
                self.pca = PCA(n_components=self.num_dim, random_state=42)
                features = self.pca.fit_transform(features)
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(features)
        if self.pca is not None:
            features = self.pca.transform(features)
        return self.scaler.transform(features)

    def log_likelihoods(self, X, params):
        return jnp.stack(
            [dist.MultivariateNormal(m, c).log_prob(X) for m, c in zip(params["class_means"], params["class_covs"])],
            axis=-1,
        )

    def accuracy(self, params, X, y):
        pred = jnp.argmax(self.log_likelihoods(jnp.array(X), params), axis=-1)
        return 100.0 * jnp.sum(pred == y).tolist() / float(len(X))

    def train(self, features, labels, test_features, test_labels, stage):
        features = self.preprocess(features)
        test_features = self.preprocess(test_features)
        labels = labels.astype(int)
        self.params = self.run_inference(
            jnp.array(features),
            jnp.array(labels),
            jnp.array(test_features),
            jnp.array(test_labels),
            log_prefix=f"stage_{stage}_Flearning",
            use_correct_scaling_factor=False,
        )
        if self.global_params is not None:
            # Second pass on the samples assigned to the new clusters.
            pred = jnp.argmax(self.log_likelihoods(jnp.array(features), self.params), axis=-1)
            novel = pred >= self.label_offset
            print(f"Number of novel samples: {novel.sum()} / {len(features)}")
            self.params = self.run_inference(
                jnp.array(features[novel]),
                jnp.array(labels[novel]),
                jnp.array(test_features),
                jnp.array(test_labels),
                log_prefix=f"stage_{stage}_Slearning",
                use_correct_scaling_factor=self.use_correct_scaling_factor,
            )
        self.global_params = copy.deepcopy(self.params)

    # ------------------------------------------------------------------ stage
    def run(self, features, labels, test_features, test_labels, stage, testing_set):
        self.train(features, labels, test_features, test_labels, stage)
        for name, data in (("All", testing_set["test_all"]), ("Old", testing_set["test_old"])):
            acc = self.accuracy(self.params, self.preprocess(data._x), data._y)
            print(f"[Stage {stage}] {name} accuracy: {acc:.2f}%")
            self.writer.add_scalar(f"Test/Accuracy/{name}", acc, stage)
        acc = self.accuracy(self.params, self.preprocess(test_features), test_labels)
        print(f"[Stage {stage}] Novel accuracy: {acc:.2f}%")
        self.writer.add_scalar("Test/Accuracy/Novel", acc, stage)
        self.save()

    def save(self):
        np.save(f"{self.save_dir}class_means.npy", np.array(self.params["class_means"]))
        np.save(f"{self.save_dir}class_covariances.npy", np.array(self.params["class_covs"]))
        if self.pca is None:
            return
        root = self.save_dir.rsplit("stage", 1)[0]
        with open(f"{root}pca_model.pkl", "wb") as f:
            pickle.dump(self.pca, f)
        with open(f"{root}scaler_model.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        order = self.class_order
        names = {str(i): self.class_names.get(int(o), f"class_{int(o)}") for i, o in enumerate(order)}
        orig_names = {str(int(o)): self.class_names.get(int(o), f"class_{int(o)}") for o in order}
        mapping = {
            "classifier2orig": {str(i): int(o) for i, o in enumerate(order)},
            "orig2classifier": {str(int(o)): i for i, o in enumerate(order)},
            "classifier2name": names,
            "name2classifier": {name: int(i) for i, name in names.items()},
            "orig2name": orig_names,
            "name2orig": {name: int(o) for o, name in orig_names.items()},
            "class_order": [int(x) for x in order],
            "stage_offsets": self.stage_offsets,
            "stage_increments": self.stage_increments,
        }
        with open(os.path.join(root, "class_mappings.json"), "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)
        print(f"Saved clusters, PCA, scaler, and class mapping to {root}")
