"""Continual category discovery with VB-CGCD (6 known + 3 discovered degradation classes).

Stage 0 fits one Gaussian per known class on labeled features. Each later stage clusters the
unlabeled features with a GMM, fits the new classes by variational inference, and saves the
clusters used by the restoration model (descriptor and discovery-adaptive score).

Usage (from discovery/):
    python run_vbcgcd.py --feature_root <dir containing <exp_name>/<model>> --output_dir <dir>
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import jax
import numpy as np
import torch
from continuum import ClassIncremental
from continuum.datasets import InMemoryDataset

from utils_lib.gmm import GMMCluster
from utils_lib.vbcgcd import VBCGCDClassifier

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

SAMPLES_PER_BASE = 1000  # labeled samples per known class
SAMPLES_PER_NOVEL = 500  # unlabeled samples per new class in its stage
SAMPLES_PER_OLD = 20  # unlabeled samples per previously seen class in each later stage


@dataclass
class DataPoint:
    _x: np.ndarray
    _y: np.ndarray
    _idx: Optional[np.ndarray] = None


def _concat(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return np.concatenate([a, b], axis=0)


def _shuffle(x, y):
    if x is None or len(x) == 0:
        return x, y
    perm = np.random.permutation(x.shape[0])
    return x[perm], y[perm]


def _take_known(pool_x, pool_y, seen_classes, per_class):
    """Move ``per_class`` samples of each seen class from the unlabeled pool into the stage."""
    if pool_x is None or len(pool_x) == 0 or len(seen_classes) == 0:
        return pool_x, pool_y, None, None
    perm = np.random.permutation(pool_x.shape[0])
    x, y = pool_x[perm], pool_y[perm]
    rest_x, rest_y, known_x, known_y = [], [], [], []
    for c in seen_classes:
        cx, cy = x[y == c], y[y == c]
        n = min(per_class, len(cx))
        if n > 0:
            known_x.append(cx[:n])
            known_y.append(cy[:n])
        if len(cx) > n:
            rest_x.append(cx[n:])
            rest_y.append(cy[n:])
    rest = (np.concatenate(rest_x), np.concatenate(rest_y)) if rest_x else (x[:0], y[:0])
    known = (np.concatenate(known_x), np.concatenate(known_y)) if known_x else (x[:0], y[:0])
    return rest[0], rest[1], known[0], known[1]


def build_stage_train_data(scenario):
    """Labeled base data for stage 0; for each later stage, new-class samples plus a few seen-class samples."""
    pool_x = pool_y = seen = None
    stages = []
    for stage, data in enumerate(scenario):
        classes = np.unique(data._y)
        if stage == 0:
            seen = classes
            labeled_x = labeled_y = None
            for c in classes:
                cx = data._x[data._y == c]
                n = min(SAMPLES_PER_BASE, len(cx))
                labeled_x, labeled_y = _concat(labeled_x, cx[:n]), _concat(labeled_y, np.ones(n) * c)
                pool_x, pool_y = _concat(pool_x, cx[n:]), _concat(pool_y, np.ones(len(cx) - n) * c)
            labeled_x, labeled_y = _shuffle(labeled_x, labeled_y)
            pool_x, pool_y = _shuffle(pool_x, pool_y)
            stages.append(DataPoint(labeled_x, labeled_y))
            continue
        pool_x, pool_y, known_x, known_y = _take_known(pool_x, pool_y, seen, SAMPLES_PER_OLD)
        seen = _concat(seen, classes)
        novel_x = novel_y = None
        for c in classes:
            cx = data._x[data._y == c]
            n = min(SAMPLES_PER_NOVEL, len(cx))
            novel_x, novel_y = _concat(novel_x, cx[:n]), _concat(novel_y, np.ones(n) * c)
            pool_x, pool_y = _concat(pool_x, cx[n:]), _concat(pool_y, np.ones(len(cx) - n) * c)
        stage_x, stage_y = _shuffle(_concat(known_x, novel_x), _concat(known_y, novel_y))
        stages.append(DataPoint(stage_x, stage_y))
        pool_x, pool_y = _shuffle(pool_x, pool_y)
    return stages


def build_test_data(test_x, test_y, increments, class_order):
    """Per-stage test data of the new classes and of all classes seen so far."""
    labels = test_y.astype(np.int64)
    classifier_labels = np.argsort(np.asarray(class_order, dtype=np.int64))[labels]
    novel, cumulative = [], []
    offset = 0
    for increment in increments:
        idx = np.flatnonzero(np.isin(labels, np.array(class_order[offset : offset + increment], dtype=np.int64)))
        novel.append(DataPoint(test_x[idx], classifier_labels[idx], idx))
        offset += increment
    seen_x = seen_y = None
    for data in novel:
        seen_x, seen_y = _concat(seen_x, data._x), _concat(seen_y, data._y)
        cumulative.append(DataPoint(seen_x, seen_y))
    return novel, cumulative


def load_features(args):
    feature_dir = os.path.join(args.feature_root, args.exp_name, args.model)
    arrays = {
        k: np.load(os.path.join(feature_dir, f"{k}-{args.model}.npy"))
        for k in ("features", "labels", "test_features", "test_labels")
    }
    with open(os.path.join(feature_dir, "class_order.txt")) as f:
        class_order = [int(x) for x in f.read().strip().split(",")]
    with open(os.path.join(feature_dir, "class_names.txt")) as f:
        class_names = {int(i): name for i, name in (line.strip().split(",", 1) for line in f if "," in line)}
    if sorted(class_order) != list(range(args.num_classes)):
        raise ValueError(f"class_order {class_order} must be a permutation of 0..{args.num_classes - 1}")
    missing = set(range(args.num_classes)) - set(np.unique(arrays["labels"]).tolist())
    if missing:
        raise ValueError(f"No features for classes {sorted(missing)}; run extract_real_features.py first.")
    return arrays, class_order, class_names


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--feature_root", required=True, help="Directory containing <exp_name>/<model>/features-<model>.npy"
    )
    parser.add_argument("--exp_name", default="wres_with_clear")
    parser.add_argument("--model", default="dinov3-vitl16-sl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="cgcd_6p3")
    parser.add_argument(
        "--increment_schedule", default="6,3", help="Known classes followed by the classes of each later stage"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_dim", type=int, default=384, help="PCA dimension")
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    increments = [int(x) for x in args.increment_schedule.split(",")]
    args.num_classes = sum(increments)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _ = jax.random.PRNGKey(args.seed)

    arrays, class_order, class_names = load_features(args)
    print(f"Classes per stage: {increments}, class order: {class_order}")
    train_scenario = ClassIncremental(
        InMemoryDataset(arrays["features"], arrays["labels"]), increment=increments, class_order=class_order
    )
    stage_train = build_stage_train_data(train_scenario)
    stage_test, stage_seen_test = build_test_data(
        arrays["test_features"], arrays["test_labels"], increments, class_order
    )

    run_dir = os.path.join(args.output_dir, args.exp_name, args.run_name)
    saved_models = os.path.join(run_dir, "saved_models")
    os.makedirs(saved_models, exist_ok=True)
    classifier = VBCGCDClassifier(num_dim=args.num_dim, num_classes=args.num_classes)
    classifier.init_parameters(
        n_epochs=args.n_epochs,
        lr=args.lr,
        log_dir=os.path.join(run_dir, "stage0"),
        save_dir=os.path.join(saved_models, "stage0"),
        batch_size=args.batch_size,
        increment=increments[1],
        base=increments[0],
        use_correct_scaling_factor=True,
        class_order=class_order,
        class_names=class_names,
    )

    used = [increments[0]]
    for stage, (train, test) in enumerate(zip(stage_train, stage_test)):
        if stage == 0:
            classifier.run(train._x, train._y, test._x, test._y, stage, {"test_all": test, "test_old": test})
            continue
        offset = sum(used)
        clustering = GMMCluster(n_components=increments[stage], label_offset=offset, random_state=args.seed)
        clustering.fit(train._x)
        pseudo_labels = clustering.predict(train._x, train._y)
        print(f"[Stage {stage}] GMM cluster sizes: {np.unique(pseudo_labels, return_counts=True)}")
        used.append(increments[stage])
        classifier.set_stage_structure(np.cumsum([0] + used[:-1]).tolist(), used)
        classifier.increment = increments[stage]
        classifier.set_label_offset(offset)
        classifier.update_dir_infos(
            os.path.join(run_dir, "log", f"stage{stage}"), os.path.join(saved_models, f"stage{stage}")
        )
        testing_set = {"test_all": stage_seen_test[stage], "test_old": stage_seen_test[stage - 1]}
        classifier.run(train._x, pseudo_labels, test._x, test._y, stage, testing_set)

    with open(os.path.join(run_dir, "increment_summary.json"), "w") as f:
        json.dump({"class_order": class_order, "increment_schedule": increments}, f, indent=2)
    print(f"Saved VB-CGCD clusters to {saved_models}")


if __name__ == "__main__":
    main()
