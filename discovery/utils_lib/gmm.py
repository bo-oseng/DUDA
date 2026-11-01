"""Candidate clusters for the novel degradations: a Gaussian mixture with a fixed number of components."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture


class GMMCluster:
    def __init__(self, n_components, label_offset=0, random_state=None):
        self.n_components = n_components
        self.label_offset = label_offset
        self.model = GaussianMixture(n_components=n_components, random_state=random_state)

    def fit(self, features):
        self.model.fit(features)

    def predict(self, features, labels):
        """Cluster ids matched to the benchmark labels of the novel samples (Hungarian matching),
        shifted by ``label_offset``. Only used to index the discovered clusters consistently."""
        pred = self.model.predict(features)
        novel = labels >= self.label_offset
        conf = confusion_matrix(y_true=(labels - self.label_offset)[novel], y_pred=pred[novel])
        rows, cols = linear_sum_assignment(-conf)
        matched = np.zeros_like(pred)
        for i, j in zip(rows, cols):
            matched[pred == j] = i
        return matched + self.label_offset
