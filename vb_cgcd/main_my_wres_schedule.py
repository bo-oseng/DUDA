import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import numpy as np
import torch
from continuum import ClassIncremental
from continuum.datasets import InMemoryDataset

from classifier.mngmm_my_hist import MNGMMClassifier
from classifier.mngmm_wresvlm import MNGMMClassifier as MNGMMWRESVLMClassifier
from clustering.gmm import GMMCluster
from dataloaders import wresvlm as wres_mod
from dataloaders import wresvlm_safe as wres_safe_mod

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_current_time():
    return time.strftime("%Y%m%d-%H%M", time.localtime())


def Clustering_alg(alg):
    if alg == "gmm":
        return GMMCluster
    raise ValueError("Clustering algorithm not supported")


def Classifier_alg(alg):
    if alg == "mngmm":
        return MNGMMClassifier
    if alg == "mngmm_wresvlm":
        return MNGMMWRESVLMClassifier
    raise ValueError("Classifier algorithm not supported")


@dataclass
class DataPoint:
    _x: np.ndarray
    _y: np.ndarray
    _idx: Optional[np.ndarray] = None
    _meta: Optional[List[Dict[str, Any]]] = None


def _concat_data(data1, data2):
    if data1 is None:
        return data2
    if data2 is None:
        return data1
    return np.concatenate([data1, data2], axis=0)


def _concat_meta(meta1, meta2):
    if meta1 is None:
        return meta2
    if meta2 is None:
        return meta1
    return list(meta1) + list(meta2)


def _load_manifest_jsonl(path: str, expected_len: Optional[int] = None):
    if not os.path.exists(path):
        print(f"[WRES-SCHEDULE] manifest file not found: {path}")
        return None

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if expected_len is not None and len(records) != expected_len:
        print(
            f"[WRES-SCHEDULE] manifest length mismatch for {path}: "
            f"expected {expected_len}, found {len(records)}. Ignore manifest."
        )
        return None
    return records


def _resort_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x is None or len(x) == 0:
        return x, y
    perm = np.random.permutation(x.shape[0])
    return x[perm], y[perm]


def _sample_known_instance(
    unlabeled_x_pool: np.ndarray,
    unlabeled_y_pool: np.ndarray,
    seen_classes: np.ndarray,
    num_known_inc: int,
):
    if unlabeled_x_pool is None or len(unlabeled_x_pool) == 0 or len(seen_classes) == 0:
        return unlabeled_x_pool, unlabeled_y_pool, None, None

    perm = np.random.permutation(unlabeled_x_pool.shape[0])
    x = unlabeled_x_pool[perm]
    y = unlabeled_y_pool[perm]

    remained_x = []
    remained_y = []
    seen_samples_x = []
    seen_samples_y = []

    for c in seen_classes:
        cc_x = x[y == c]
        cc_y = y[y == c]

        take_n = min(num_known_inc, len(cc_x))
        if take_n > 0:
            seen_samples_x.append(cc_x[:take_n])
            seen_samples_y.append(cc_y[:take_n])
        if len(cc_x) > take_n:
            remained_x.append(cc_x[take_n:])
            remained_y.append(cc_y[take_n:])

    if len(remained_x) > 0:
        new_pool_x = np.concatenate(remained_x, axis=0)
        new_pool_y = np.concatenate(remained_y, axis=0)
    else:
        new_pool_x = x[:0]
        new_pool_y = y[:0]

    if len(seen_samples_x) > 0:
        known_x = np.concatenate(seen_samples_x, axis=0)
        known_y = np.concatenate(seen_samples_y, axis=0)
    else:
        known_x = x[:0]
        known_y = y[:0]

    return new_pool_x, new_pool_y, known_x, known_y


def _load_class_order(class_order_file: str, num_classes: int, seed: int) -> List[int]:
    if os.path.exists(class_order_file):
        with open(class_order_file, "r") as f:
            raw = f.read().strip()
        class_order = [int(x) for x in raw.split(",")] if raw else []
        print(f"[WRES-SCHEDULE] Loaded class order from: {class_order_file}")
    else:
        print(f"[WRES-SCHEDULE] class_order file not found: {class_order_file}")
        class_order = []

    unique = []
    seen = set()
    for c in class_order:
        if 0 <= c < num_classes and c not in seen:
            seen.add(c)
            unique.append(c)
    class_order = unique

    if len(class_order) < num_classes:
        rest = [c for c in range(num_classes) if c not in seen]
        class_order.extend(rest)
        print(f"[WRES-SCHEDULE] class_order extended with remaining classes: {rest}")
    elif len(class_order) > num_classes:
        class_order = class_order[:num_classes]
        print(f"[WRES-SCHEDULE] class_order truncated to num_classes={num_classes}")

    if not class_order:
        np.random.seed(seed)
        class_order = list(range(num_classes))

    return class_order


def _parse_class_order_override(raw: str, num_classes: int) -> Optional[List[int]]:
    if not raw.strip():
        return None

    try:
        class_order = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(
            "--class_order_override must be a comma-separated list of integer class IDs."
        ) from exc

    expected = set(range(num_classes))
    actual = set(class_order)
    if len(class_order) != num_classes or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        duplicates = sorted({class_id for class_id in class_order if class_order.count(class_id) > 1})
        raise ValueError(
            f"--class_order_override must be an exact permutation of 0..{num_classes - 1}. "
            f"Got {class_order}; missing={missing}, extra={extra}, duplicates={duplicates}."
        )

    return class_order


def _load_class_names(class_names_file: str, num_classes: int) -> Dict[int, str]:
    class_names = {}
    if os.path.exists(class_names_file):
        with open(class_names_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "," not in line:
                    continue
                idx_str, name = line.split(",", 1)
                class_names[int(idx_str)] = name
        print(f"[WRES-SCHEDULE] Loaded class names from: {class_names_file}")
    else:
        print(f"[WRES-SCHEDULE] class_names file not found: {class_names_file}")

    for idx in range(num_classes):
        class_names.setdefault(idx, f"class_{idx}")
    return class_names


def _parse_increment_schedule(args) -> List[int]:
    if args.increment_schedule:
        schedule = [int(x.strip()) for x in args.increment_schedule.split(",") if x.strip()]
        if len(schedule) < 2:
            raise ValueError("--increment_schedule must include at least base and one novel stage, e.g. 5,2,1")
        if schedule[0] != args.base:
            raise ValueError(
                f"First value of --increment_schedule must equal --base ({args.base}). Got: {schedule[0]}"
            )
        if sum(schedule) != args.num_classes:
            raise ValueError(
                f"Sum of --increment_schedule must equal --num_classes ({args.num_classes}). Got: {sum(schedule)}"
            )
        return schedule

    if args.increment <= 0:
        raise ValueError("--increment must be > 0")
    rem = args.num_classes - args.base
    if rem < 0:
        raise ValueError("--num_classes must be >= --base")
    if rem == 0:
        return [args.base]
    if rem % args.increment != 0:
        raise ValueError(
            f"Uniform increment is not compatible: num_classes-base={rem} is not divisible by increment={args.increment}. "
            f"Use --increment_schedule, e.g. '5,2,1' or '5,1,2'."
        )
    return [args.base] + [args.increment] * (rem // args.increment)


def _stage_offsets_from_increments(stage_increments: List[int]) -> List[int]:
    offsets = []
    total = 0
    for increment in stage_increments:
        offsets.append(total)
        total += int(increment)
    return offsets


def _resolve_head_num_classes(args) -> int:
    head_num_classes = int(args.head_num_classes) if int(args.head_num_classes) > 0 else int(args.num_classes)
    if head_num_classes < args.num_classes:
        raise ValueError(
            f"--head_num_classes ({head_num_classes}) must be >= --num_classes ({args.num_classes}). "
            "Use a larger head only for classifier capacity; dataset/eval classes still come from --num_classes."
        )
    return head_num_classes


def _sampling_cfg_for_load_mode(load_mode: str) -> Tuple[int, int, int]:
    if load_mode in {"t1", "t2"}:
        return 1000, 500, 20
    raise ValueError(f"load_mode '{load_mode}' is not supported in schedule script (use t1/t2).")


def _build_stage_train_data(
    train_scenario,
    base: int,
    samples_per_base: int,
    samples_per_novel: int,
    samples_per_old_mem: int,
):
    labeled_per_class = samples_per_base
    num_novel_per_stage_per_class = samples_per_novel
    num_known_inc = samples_per_old_mem

    unlabeled_x_pool = None
    unlabeled_y_pool = None

    dataset_per_stage = []
    seen_classes = None

    for stage_i, train_data in enumerate(train_scenario):
        if stage_i == 0:
            labeled_x = None
            labeled_y = None

            idx_classes = np.unique(train_data._y)
            if len(idx_classes) != base:
                print(
                    f"[WRES-SCHEDULE] Warning: stage0 class count is {len(idx_classes)} (expected base={base})."
                )
            seen_classes = idx_classes

            for idx in idx_classes:
                cls_x = train_data._x[train_data._y == idx]
                take_n = min(labeled_per_class, len(cls_x))
                labeled_x = _concat_data(labeled_x, cls_x[:take_n])
                labeled_y = _concat_data(labeled_y, np.ones(take_n) * idx)

                unlabeled_x = cls_x[take_n:]
                unlabeled_y = np.ones(len(unlabeled_x)) * idx
                unlabeled_x_pool = _concat_data(unlabeled_x_pool, unlabeled_x)
                unlabeled_y_pool = _concat_data(unlabeled_y_pool, unlabeled_y)

            labeled_x, labeled_y = _resort_data(labeled_x, labeled_y)
            unlabeled_x_pool, unlabeled_y_pool = _resort_data(unlabeled_x_pool, unlabeled_y_pool)
            dataset_per_stage.append(DataPoint(labeled_x, labeled_y))

        else:
            unlabeled_x_pool, unlabeled_y_pool, known_x, known_y = _sample_known_instance(
                unlabeled_x_pool, unlabeled_y_pool, seen_classes, num_known_inc
            )

            idx_classes = np.unique(train_data._y)
            seen_classes = _concat_data(seen_classes, idx_classes)

            unlabeled_novels_x = None
            unlabeled_novels_y = None

            for idx in idx_classes:
                cls_x = train_data._x[train_data._y == idx]
                take_n = min(num_novel_per_stage_per_class, len(cls_x))

                unlabeled_novels_x = _concat_data(unlabeled_novels_x, cls_x[:take_n])
                unlabeled_novels_y = _concat_data(unlabeled_novels_y, np.ones(take_n) * idx)

                remain_x = cls_x[take_n:]
                remain_y = np.ones(len(remain_x)) * idx
                unlabeled_x_pool = _concat_data(unlabeled_x_pool, remain_x)
                unlabeled_y_pool = _concat_data(unlabeled_y_pool, remain_y)

            stage_x = _concat_data(known_x, unlabeled_novels_x)
            stage_y = _concat_data(known_y, unlabeled_novels_y)
            stage_x, stage_y = _resort_data(stage_x, stage_y)
            dataset_per_stage.append(DataPoint(stage_x, stage_y))

            unlabeled_x_pool, unlabeled_y_pool = _resort_data(unlabeled_x_pool, unlabeled_y_pool)

    return dataset_per_stage


def _build_test_data_points(
    test_features: np.ndarray,
    test_labels: np.ndarray,
    increments_full: List[int],
    class_order: List[int],
    feature_dim: int,
    test_metadata=None,
):
    label_array = test_labels.astype(np.int64)
    classifier_labels = np.argsort(np.asarray(class_order, dtype=np.int64))[label_array]
    novel_per_stage = []

    offset = 0
    for increment in increments_full:
        stage_classes = np.array(class_order[offset : offset + increment], dtype=np.int64)
        stage_mask = np.isin(label_array, stage_classes)
        stage_idx = np.flatnonzero(stage_mask)
        stage_meta = [test_metadata[i] for i in stage_idx] if test_metadata is not None else None
        novel_per_stage.append(
            DataPoint(test_features[stage_idx], classifier_labels[stage_idx], stage_idx, stage_meta)
        )
        offset += increment

    all_per_stage = []
    old_per_stage = [
        DataPoint(
            np.empty((0, feature_dim), dtype=test_features.dtype),
            np.empty((0,), dtype=test_labels.dtype),
            np.empty((0,), dtype=np.int64),
            [] if test_metadata is not None else None,
        )
    ]
    tmp_x = None
    tmp_y = None
    tmp_idx = None
    tmp_meta = None
    for test_data in novel_per_stage:
        tmp_x = _concat_data(tmp_x, test_data._x)
        tmp_y = _concat_data(tmp_y, test_data._y)
        tmp_idx = _concat_data(tmp_idx, test_data._idx)
        tmp_meta = _concat_meta(tmp_meta, test_data._meta)
        all_dp = DataPoint(tmp_x, tmp_y, tmp_idx, tmp_meta)
        all_per_stage.append(all_dp)
        old_per_stage.append(all_dp)
    return novel_per_stage, old_per_stage, all_per_stage


def _ensure_wres_real_split(args):
    if "wresvlm_safe" in args.dataset:
        split_mod = wres_safe_mod
        tag = "WRES-SAFE"
    else:
        split_mod = wres_mod
        tag = "WRES"

    print(
        f"[{tag}-SCHEDULE] Ensuring split manifests in {args.real_data_dir} "
        f"(test_split={args.real_test_split}, seed={args.seed})"
    )
    split_mod.ensure_real_train_split_manifests(
        real_data_dir=args.real_data_dir,
        class_names=split_mod.REAL_NEW_CLASSES,
        test_split=args.real_test_split,
        seed=args.seed,
        split_filename=args.real_split_file,
        force_resplit=args.real_force_resplit,
    )


def build_wres_schedule_dataloader(args, increments_full: List[int]):
    _ensure_wres_real_split(args)

    exp_root_dir = os.path.join(args.data_dir, args.exp_name)
    feature_dir = os.path.join(exp_root_dir, args.pretrained_model_name)

    features = np.load(os.path.join(feature_dir, f"features-{args.pretrained_model_name}.npy"))
    labels = np.load(os.path.join(feature_dir, f"labels-{args.pretrained_model_name}.npy"))
    test_features = np.load(os.path.join(feature_dir, f"test_features-{args.pretrained_model_name}.npy"))
    test_labels = np.load(os.path.join(feature_dir, f"test_labels-{args.pretrained_model_name}.npy"))
    test_manifest_path = os.path.join(feature_dir, f"test_features_manifest-{args.pretrained_model_name}.jsonl")
    test_metadata = _load_manifest_jsonl(test_manifest_path, expected_len=len(test_features))

    class_order_file = os.path.join(feature_dir, "class_order.txt")
    class_names_file = os.path.join(feature_dir, "class_names.txt")
    class_order = _parse_class_order_override(args.class_order_override, num_classes=args.num_classes)
    if class_order is None:
        class_order = _load_class_order(class_order_file, num_classes=args.num_classes, seed=args.seed)
    else:
        print(f"[WRES-SCHEDULE] Using class order override: {class_order}")
    class_names = _load_class_names(class_names_file, num_classes=args.num_classes)

    print(f"[WRES-SCHEDULE] Increment schedule: {increments_full} (base + stage increments)")
    offset = 0
    for stage, increment in enumerate(increments_full):
        stage_class_ids = class_order[offset : offset + increment]
        stage_class_names = [class_names[class_id] for class_id in stage_class_ids]
        print(
            f"[WRES-SCHEDULE] Stage {stage} classes: "
            f"ids={stage_class_ids}, names={stage_class_names}"
        )
        offset += increment
    print(f"[WRES-SCHEDULE] Feature shapes: train={features.shape}, test={test_features.shape}")

    expected_labels = set(range(args.num_classes))
    train_labels_present = set(np.unique(labels.astype(np.int64)).tolist())
    test_labels_present = set(np.unique(test_labels.astype(np.int64)).tolist())
    missing_train = sorted(expected_labels - train_labels_present)
    missing_test = sorted(expected_labels - test_labels_present)
    if missing_train or missing_test:
        raise ValueError(
            "Feature labels do not cover all classes required by --num_classes.\n"
            f"  expected labels: {sorted(expected_labels)}\n"
            f"  train present:   {sorted(train_labels_present)}\n"
            f"  test present:    {sorted(test_labels_present)}\n"
            f"  missing train:   {missing_train}\n"
            f"  missing test:    {missing_test}\n"
            "For WRES, append real-class features first, e.g.:\n"
            "  python feature_extractor/append_wres_real_features.py "
            f"--exp_name {args.exp_name} --output_dir {args.data_dir} "
            f"--pretrained_model_name {args.pretrained_model_name} "
            "--real_root wres_datasets/real_train_dataset --real_start_label <base>"
        )

    train_dataset = InMemoryDataset(features, labels)
    test_dataset = InMemoryDataset(test_features, test_labels)

    train_scenario = ClassIncremental(train_dataset, increment=increments_full, class_order=class_order)
    test_scenario = ClassIncremental(test_dataset, increment=increments_full, class_order=class_order)

    samples_per_base, samples_per_novel, samples_per_old_mem = _sampling_cfg_for_load_mode(args.load_mode)
    print(
        f"[WRES-SCHEDULE] Sampling config: "
        f"SAMPLES_PER_BASE={samples_per_base}, SAMPLES_PER_NOVEL={samples_per_novel}, SAMPLES_PER_OLD_MEM={samples_per_old_mem}"
    )

    train_loader = _build_stage_train_data(
        train_scenario,
        base=args.base,
        samples_per_base=samples_per_base,
        samples_per_novel=samples_per_novel,
        samples_per_old_mem=samples_per_old_mem,
    )
    test_loader, test_old_loader, test_all_loader = _build_test_data_points(
        test_features=test_features,
        test_labels=test_labels,
        increments_full=increments_full,
        class_order=class_order,
        feature_dim=test_features.shape[1],
        test_metadata=test_metadata,
    )

    return train_loader, test_loader, test_old_loader, test_all_loader, class_order, class_names


def parse_arguments():
    parser = argparse.ArgumentParser(description="WRES Flexible Increment Schedule Runner")
    parser.add_argument("--dataset", type=str, default="wresvlm")
    parser.add_argument("--data_dir", type=str, default="wres_datasets")
    parser.add_argument("--output_dir", type=str, default="../outputs/cgcd")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--load_mode", type=str, default="t1")
    parser.add_argument("--pretrained_model_name", type=str, default="dinov3-vitl16-sl")
    parser.add_argument("--base", type=int, default=5)
    parser.add_argument("--increment", type=int, default=1)
    parser.add_argument(
        "--increment_schedule",
        type=str,
        default="",
        help="Comma-separated schedule including base. e.g. '5,3' or '5,2,1' or '5,1,2'.",
    )
    parser.add_argument(
        "--class_order_override",
        type=str,
        default="",
        help="Optional exact permutation of dataset class IDs, e.g. '0,1,2,3,4,5,7,8,6'.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--trail_name", type=str, default="")
    parser.add_argument("--clustering_alg", type=str, default="gmm")
    parser.add_argument("--classifier_alg", type=str, default="mngmm_wresvlm")

    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="Actual dataset/evaluation class count available in the feature files.",
    )
    parser.add_argument(
        "--head_num_classes",
        type=int,
        default=0,
        help="Classifier head capacity. 0 reuses --num_classes. Set larger for over-provisioned blind runs.",
    )
    parser.add_argument("--num_dim", type=int, default=384)

    parser.add_argument("--with_early_stop", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_correct_scaling_factor", action="store_true")
    parser.add_argument("--use_skip_pca", action="store_false")

    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--scaling_factor", type=float, default=1.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stop_ratio", type=float, default=0)
    parser.add_argument(
        "--auto_estimate_clusters",
        action="store_true",
        help="Estimate the number of stage-specific GMM components instead of assuming the planned increment.",
    )
    parser.add_argument(
        "--blind_increment",
        action="store_true",
        help="Use the estimated K_hat as the classifier increment and next-stage label offset. "
        "The provided increment schedule is then used only to build the benchmark sessions.",
    )
    parser.add_argument("--min_components", type=int, default=1)
    parser.add_argument("--max_components", type=int, default=10)
    parser.add_argument(
        "--cluster_criterion",
        type=str,
        default="bic",
        choices=["bic", "aic", "silhouette", "dp_gmm", "deepdpm", "promptccd"],
        help="Unknown-K estimator to use for each stage: information criterion (bic/aic), silhouette score, DP-GMM, or split-merge estimators inspired by DeepDPM/PromptCCD.",
    )
    parser.add_argument(
        "--dp_weight_concentration_prior",
        type=float,
        default=None,
        help="Optional DP-GMM concentration prior. Smaller values favor fewer active components.",
    )
    parser.add_argument(
        "--dp_min_count",
        type=int,
        default=1,
        help="Minimum assigned sample count required to keep a DP-GMM component active.",
    )
    parser.add_argument(
        "--dp_min_weight",
        type=float,
        default=0.0,
        help="Minimum DP-GMM mixture weight required to keep a component active.",
    )
    parser.add_argument(
        "--split_merge_max_iter",
        type=int,
        default=5,
        help="Maximum number of split-merge refinement iterations for deepdpm/promptccd estimators.",
    )
    parser.add_argument(
        "--split_merge_min_cluster_size",
        type=int,
        default=6,
        help="Minimum cluster size required before considering split/merge updates for deepdpm/promptccd estimators.",
    )
    parser.add_argument(
        "--split_merge_alpha",
        type=float,
        default=1.0,
        help="Split-merge prior strength for deepdpm/promptccd estimators. Larger values make splits easier and merges harder.",
    )

    parser.add_argument("--real_data_dir", type=str, default="wres_datasets/real_train_dataset")
    parser.add_argument("--real_test_split", type=float, default=0.1)
    parser.add_argument("--real_split_file", type=str, default="test_data_info.txt")
    parser.add_argument("--real_force_resplit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if "wresvlm" not in args.dataset:
        raise ValueError("main_my_wres_schedule.py only supports --dataset wresvlm or wresvlm_safe.")
    if args.blind_increment and not args.auto_estimate_clusters:
        raise ValueError("--blind_increment requires --auto_estimate_clusters.")
    if args.auto_estimate_clusters and not args.blind_increment:
        raise ValueError(
            "--auto_estimate_clusters must be paired with --blind_increment in main_my_wres_schedule.py, "
            "otherwise the fixed schedule offsets become inconsistent with the estimated cluster count."
        )

    increments_full = _parse_increment_schedule(args)
    if len(increments_full) < 2:
        raise ValueError("At least one incremental stage is required. Use schedule like 5,3 or 5,2,1.")
    head_num_classes = _resolve_head_num_classes(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _ = jax.random.PRNGKey(args.seed)

    Clustering = Clustering_alg(args.clustering_alg)
    Classifier = Classifier_alg(args.classifier_alg)

    train_loader, test_loader, test_old_loader, test_all_loader, class_order, class_names = build_wres_schedule_dataloader(
        args, increments_full
    )
    print(
        f"[WRES-SCHEDULE] Dataset classes={args.num_classes}, "
        f"head capacity={head_num_classes}, blind_increment={args.blind_increment}"
    )

    base_log_dir = os.path.join(args.output_dir, args.exp_name, args.trail_name)
    if os.path.exists(base_log_dir):
        base_log_dir = os.path.join(args.output_dir, args.exp_name, f"{args.trail_name}")

    saved_models_root = os.path.join(base_log_dir, "saved_models")
    os.makedirs(saved_models_root, exist_ok=True)

    s_classifier = Classifier(
        num_classes=head_num_classes,
        num_dim=args.num_dim,
        with_early_stop=args.with_early_stop,
        use_pca=args.use_skip_pca,
    )

    first_inc = increments_full[1]
    print(f"Scaling factor: {args.scaling_factor}")
    s_classifier.init_parameters(
        n_epochs=args.n_epochs,
        lr=args.lr,
        log_dir=os.path.join(base_log_dir, "stage0"),
        save_dir=os.path.join(saved_models_root, "stage0"),
        batch_size=args.batch_size,
        increment=first_inc,
        base=args.base,
        scaling_factor=args.scaling_factor,
        use_correct_scaling_factor=args.use_correct_scaling_factor,
        early_stop_ratio=args.early_stop_ratio,
        class_order=class_order,
        class_names_by_orig=class_names,
    )

    session_test_data = []
    effective_stage_increments = [int(args.base)]
    stage_records = [
        {
            "stage": 0,
            "planned_increment": int(args.base),
            "estimated_increment": int(args.base),
            "used_increment": int(args.base),
            "planned_offset": 0,
            "used_offset": 0,
            "original_class_ids": [int(x) for x in class_order[: args.base]],
            "class_names": [class_names[int(x)] for x in class_order[: args.base]],
        }
    ]
    if hasattr(s_classifier, "set_stage_structure"):
        s_classifier.set_stage_structure(
            stage_offsets=_stage_offsets_from_increments(effective_stage_increments),
            stage_increments=effective_stage_increments,
        )

    for stage, (train_data, test_data, test_old_data, test_all_data) in enumerate(
        zip(train_loader, test_loader, test_old_loader, test_all_loader)
    ):
        if stage == 0:
            testing_set = {
                "test_old": test_data,
                "test_all": test_data,
                "known_test": test_data,
                "session_tests": [test_data],
            }
            session_test_data.append(test_data)
            s_classifier.run(
                train_data._x,
                train_data._y,
                test_data._x,
                test_data._y,
                current_stage=stage,
                testing_set=testing_set,
            )
            known_test_data = test_data
            continue

        planned_inc = int(increments_full[stage])
        planned_label_offset = int(sum(increments_full[:stage]))
        label_offset = int(sum(effective_stage_increments))

        clustering = Clustering(
            init_components=planned_inc,
            label_offset=label_offset,
            random_state=args.seed,
            auto_estimate=args.auto_estimate_clusters,
            min_components=args.min_components,
            max_components=args.max_components,
            criterion=args.cluster_criterion,
            dp_weight_concentration_prior=args.dp_weight_concentration_prior,
            dp_min_count=args.dp_min_count,
            dp_min_weight=args.dp_min_weight,
            split_merge_max_iter=args.split_merge_max_iter,
            split_merge_min_cluster_size=args.split_merge_min_cluster_size,
            split_merge_alpha=args.split_merge_alpha,
        )
        clustering.fit(train_data._x)
        estimated_inc = int(getattr(clustering, "estimated_n_components", planned_inc))
        current_inc = estimated_inc if args.blind_increment else planned_inc
        if label_offset + current_inc > head_num_classes:
            raise ValueError(
                f"Classifier head capacity exceeded at stage {stage}: "
                f"offset {label_offset} + increment {current_inc} > head_num_classes {head_num_classes}. "
                "Increase --head_num_classes."
            )

        print(
            f"[WRES-SCHEDULE] Stage {stage}: "
            f"planned_inc={planned_inc}, estimated_inc={estimated_inc}, used_inc={current_inc}, "
            f"planned_offset={planned_label_offset}, used_offset={label_offset}"
        )

        print("❗Pseudo label prediction")
        pred = clustering.predict(train_data._x, train_data._y, with_known=True)
        print(np.unique(pred, return_counts=True))

        effective_structure = effective_stage_increments + [current_inc]
        if hasattr(s_classifier, "set_stage_structure"):
            s_classifier.set_stage_structure(
                stage_offsets=_stage_offsets_from_increments(effective_structure),
                stage_increments=effective_structure,
            )
        s_classifier.increment = current_inc
        s_classifier._set_label_offset(label_offset)
        s_classifier.update_dir_infos(
            log_dir=os.path.join(base_log_dir, "log", f"stage{stage}"),
            save_dir=os.path.join(saved_models_root, f"stage{stage}"),
        )
        session_test_data.append(test_data)
        testing_set = {
            "test_old": test_old_data,
            "test_all": test_all_data,
            "known_test": known_test_data,
            "session_tests": session_test_data,
        }

        s_classifier.run(
            features=train_data._x,
            labels=pred,
            test_features=test_data._x,
            test_labels=test_data._y,
            current_stage=stage,
            testing_set=testing_set,
        )
        effective_stage_increments.append(current_inc)
        stage_records.append(
            {
                "stage": int(stage),
                "planned_increment": planned_inc,
                "estimated_increment": estimated_inc,
                "used_increment": current_inc,
                "planned_offset": planned_label_offset,
                "used_offset": label_offset,
                "original_class_ids": [
                    int(x) for x in class_order[planned_label_offset : planned_label_offset + planned_inc]
                ],
                "class_names": [
                    class_names[int(x)]
                    for x in class_order[planned_label_offset : planned_label_offset + planned_inc]
                ],
            }
        )

    summary = {
        "dataset_num_classes": int(args.num_classes),
        "head_num_classes": int(head_num_classes),
        "class_order": [int(x) for x in class_order],
        "planned_increment_schedule": [int(x) for x in increments_full],
        "used_increment_schedule": [int(x) for x in effective_stage_increments],
        "used_stage_offsets": [int(x) for x in _stage_offsets_from_increments(effective_stage_increments)],
        "blind_increment": bool(args.blind_increment),
        "auto_estimate_clusters": bool(args.auto_estimate_clusters),
        "cluster_criterion": args.cluster_criterion if args.auto_estimate_clusters else None,
        "min_components": int(args.min_components),
        "max_components": int(args.max_components),
        "dp_weight_concentration_prior": args.dp_weight_concentration_prior,
        "dp_min_count": int(args.dp_min_count),
        "dp_min_weight": float(args.dp_min_weight),
        "split_merge_max_iter": int(args.split_merge_max_iter),
        "split_merge_min_cluster_size": int(args.split_merge_min_cluster_size),
        "split_merge_alpha": float(args.split_merge_alpha),
        "stages": stage_records,
    }
    summary_path = os.path.join(base_log_dir, "increment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[WRES-SCHEDULE] Saved increment summary to {summary_path}")
    discovered_total = int(sum(effective_stage_increments))
    if args.blind_increment and discovered_total != args.num_classes:
        print(
            f"[WRES-SCHEDULE] Warning: discovered class count ({discovered_total}) "
            f"differs from dataset class count ({args.num_classes}). "
            "Evaluation still uses the benchmark labels, so accuracy can degrade sharply when K_hat drifts."
        )
