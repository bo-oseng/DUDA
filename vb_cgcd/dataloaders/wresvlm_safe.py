import os
from typing import Dict, List

import numpy as np

from utils import dataloader_wres_safe as dataloader


REAL_NEW_CLASSES = ["RainReal", "SnowReal", "UnannotatedHazyImages"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _normalize_relpath(path: str) -> str:
    return path.replace("\\", "/")


def _list_image_relpaths(class_dir: str) -> List[str]:
    rel_paths = []
    for root, _, files in os.walk(class_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            if not _is_image_file(full_path):
                continue
            rel = os.path.relpath(full_path, class_dir)
            rel_paths.append(_normalize_relpath(rel))
    return sorted(rel_paths)


def _read_split_file(path: str) -> List[str]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def _write_split_file(path: str, class_name: str, rel_paths: List[str], split_ratio: float, seed: int):
    with open(path, "w") as f:
        f.write("# Auto-generated split manifest\n")
        f.write(f"# class={class_name}\n")
        f.write(f"# test_split={split_ratio}\n")
        f.write(f"# seed={seed}\n")
        for rel in rel_paths:
            f.write(f"{rel}\n")


def ensure_real_train_split_manifests(
    real_data_dir: str,
    class_names: List[str],
    test_split: float = 0.1,
    seed: int = 42,
    split_filename: str = "test_data_info.txt",
    force_resplit: bool = False,
):
    if not os.path.isdir(real_data_dir):
        print(f"[WRES-SAFE] real_data_dir not found: {real_data_dir} (skip split manifest generation)")
        return

    if not (0.0 <= test_split < 1.0):
        raise ValueError(f"test_split must be in [0, 1). Got: {test_split}")

    print(f"[WRES-SAFE] Ensuring split manifests in {real_data_dir} (test_split={test_split}, seed={seed})")
    if force_resplit:
        print("[WRES-SAFE] force_resplit=True, existing split files will be overwritten.")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(real_data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WRES-SAFE] class dir not found, skip: {class_dir}")
            continue

        info_path = os.path.join(class_dir, split_filename)
        rel_paths = _list_image_relpaths(class_dir)
        n = len(rel_paths)
        if n == 0:
            print(f"[WRES-SAFE] class {class_name}: no images found")
            continue

        if n == 1 or test_split == 0.0:
            n_test = 0
        else:
            n_test = int(round(n * test_split))
            n_test = max(1, n_test)
            n_test = min(n_test, n - 1)

        if os.path.exists(info_path) and not force_resplit:
            existing = _read_split_file(info_path)
            missing = [rel for rel in existing if rel not in set(rel_paths)]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"[WRES-SAFE] split mismatch in {info_path}: {len(missing)} missing entries "
                    f"(e.g., {preview}). Set --real_force_resplit to regenerate."
                )
            print(
                f"[WRES-SAFE] class {class_name}: total={n}, train={n-len(existing)}, test={len(existing)} [existing]"
            )
            continue

        rng = np.random.default_rng(seed + class_idx * 10007 + 31)
        idxs = np.arange(n)
        rng.shuffle(idxs)
        test_rel_paths = sorted([rel_paths[i] for i in idxs[:n_test]])
        _write_split_file(info_path, class_name, test_rel_paths, test_split, seed)
        print(f"[WRES-SAFE] class {class_name}: total={n}, train={n-n_test}, test={n_test} [generated]")


def _load_class_names(path: str) -> Dict[int, str]:
    names = {}
    if not os.path.exists(path):
        return names
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                idx_str, name = line.split(",", 1)
                names[int(idx_str)] = name
    return names


def _load_class_order(class_order_file: str, num_classes: int, seed: int) -> List[int]:
    if os.path.exists(class_order_file):
        with open(class_order_file, "r") as f:
            raw = f.read().strip()
        class_order = [int(x) for x in raw.split(",")] if raw else []
        print(f"[WRES-SAFE] Loaded class order from: {class_order_file}")
    else:
        print(f"[WRES-SAFE] class_order file not found: {class_order_file}")
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
        print(f"[WRES-SAFE] class_order extended with remaining classes: {rest}")
    elif len(class_order) > num_classes:
        class_order = class_order[:num_classes]
        print(f"[WRES-SAFE] class_order truncated to num_classes={num_classes}")

    if not class_order:
        np.random.seed(seed)
        class_order = list(range(num_classes))

    return class_order


class WRESVLMSafeLoader:
    def __init__(self, args):
        self.args = args
        self.exp_root_dir = os.path.join(self.args.data_dir, self.args.exp_name)
        self.class_order_file = os.path.join(self.exp_root_dir, args.pretrained_model_name, "class_order.txt")
        self.class_names_file = os.path.join(self.exp_root_dir, args.pretrained_model_name, "class_names.txt")

        self.real_data_dir = getattr(self.args, "real_data_dir", "wres_datasets/real_train_dataset")
        self.real_test_split = getattr(self.args, "real_test_split", 0.1)
        self.real_split_file = getattr(self.args, "real_split_file", "test_data_info.txt")
        self.real_force_resplit = getattr(self.args, "real_force_resplit", False)

        ensure_real_train_split_manifests(
            real_data_dir=self.real_data_dir,
            class_names=REAL_NEW_CLASSES,
            test_split=self.real_test_split,
            seed=self.args.seed,
            split_filename=self.real_split_file,
            force_resplit=self.real_force_resplit,
        )

    def _print_class_order(self, class_order: List[int]):
        name_map = _load_class_names(self.class_names_file)
        fallback_real = {i: REAL_NEW_CLASSES[i] for i in range(len(REAL_NEW_CLASSES))}
        for idx in range(self.args.num_classes):
            if idx not in name_map:
                real_idx = idx - self.args.base
                if real_idx in fallback_real:
                    name_map[idx] = fallback_real[real_idx]
                else:
                    name_map[idx] = str(idx)

        base_names = [f"{idx}({name_map.get(idx, idx)})" for idx in class_order[: self.args.base]]
        novel_names = [f"{idx}({name_map.get(idx, idx)})" for idx in class_order[self.args.base :]]

        print(f"\nWRES-SAFE Class order (seed={self.args.seed}): {class_order}")
        print(f"  Base classes: {base_names}")
        print(f"  Novel classes: {novel_names}")

    def _make_loader(self, samples_per_base: int, samples_per_novel: int, samples_per_old_mem: int):
        base = self.args.base
        increment = self.args.increment
        num_classes = self.args.num_classes

        num_labeled = base * samples_per_base
        num_novel_inc = increment * samples_per_novel
        num_known_inc = base * samples_per_old_mem

        class_order = _load_class_order(self.class_order_file, num_classes=num_classes, seed=self.args.seed)
        self._print_class_order(class_order)
        print(
            f"  Sampling config: SAMPLES_PER_BASE={samples_per_base}, "
            f"SAMPLES_PER_NOVEL={samples_per_novel}, SAMPLES_PER_OLD_MEM={samples_per_old_mem}"
        )

        loader = dataloader.StrictPerClassIncrementalLoader(
            data_dir=self.args.data_dir,
            exp_root_dir=self.exp_root_dir,
            pretrained_model_name=self.args.pretrained_model_name,
            base=base,
            increment=increment,
            num_labeled=num_labeled,
            num_novel_inc=num_novel_inc,
            num_known_inc=num_known_inc,
            class_order=class_order,
        )

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode="all")
        test_novel_loader = loader.test_dataloader(mode="novel")
        test_old_loader = loader.test_dataloader(mode="old")
        return train_loader, test_novel_loader, test_old_loader, test_all_loader, class_order

    def makeT1Loader(self):
        return self._make_loader(samples_per_base=1000, samples_per_novel=500, samples_per_old_mem=20)

    def makeT2Loader(self):
        return self._make_loader(samples_per_base=1000, samples_per_novel=500, samples_per_old_mem=20)

    def makeOracleLoader(self):
        num_classes = self.args.num_classes
        samples_per_class = 1000
        class_order = _load_class_order(self.class_order_file, num_classes=num_classes, seed=self.args.seed)
        self._print_class_order(class_order)
        print(f"[WRES-SAFE Oracle] SAMPLES_PER_CLASS={samples_per_class}")

        feature_dir = os.path.join(self.exp_root_dir, self.args.pretrained_model_name)
        train_features = np.load(os.path.join(feature_dir, f"features-{self.args.pretrained_model_name}.npy"))
        train_labels = np.load(os.path.join(feature_dir, f"labels-{self.args.pretrained_model_name}.npy"))
        test_features = np.load(os.path.join(feature_dir, f"test_features-{self.args.pretrained_model_name}.npy"))
        test_labels = np.load(os.path.join(feature_dir, f"test_labels-{self.args.pretrained_model_name}.npy"))
        print(f"[WRES-SAFE Oracle] loaded train={train_features.shape}, test={test_features.shape}")

        sampled_x = []
        sampled_y = []
        for cls in class_order:
            cls_mask = train_labels == cls
            cls_x = train_features[cls_mask]
            cls_y = train_labels[cls_mask]
            n_samples = min(samples_per_class, len(cls_x))
            sampled_x.append(cls_x[:n_samples])
            sampled_y.append(cls_y[:n_samples])

        train_x = np.concatenate(sampled_x, axis=0)
        train_y = np.concatenate(sampled_y, axis=0)

        class DataPointOracle:
            def __init__(self, x, y):
                self._x = x
                self._y = y

        train_data = DataPointOracle(train_x, train_y)
        test_data = DataPointOracle(test_features, test_labels)
        train_loader = iter([train_data])
        test_loader = iter([test_data])
        return train_loader, test_loader, test_loader, test_loader, class_order
