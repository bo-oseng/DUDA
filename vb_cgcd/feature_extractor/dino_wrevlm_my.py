import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from sl_finetuned_model import finetune_dino as finetune_dino_single
from sl_finetuned_model import load_finetuned_model_from_checkpoint as load_finetuned_model_from_checkpoint_single
from sl_finetuned_model_ddp import finetune_dino as finetune_dino_ddp
from sl_finetuned_model_ddp import load_finetuned_model_from_checkpoint as load_finetuned_model_from_checkpoint_ddp
from verify_checkpoint import verify_checkpoint_loading


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _find_clear_class_name(class_names: List[str]) -> Optional[str]:
    clear_candidates = [name for name in class_names if name.lower() == "clear"]
    if len(clear_candidates) > 1:
        raise ValueError(
            f"Multiple clear-like class names found: {clear_candidates}. "
            "Keep exactly one clear class to ensure deterministic class indexing."
        )
    return clear_candidates[0] if clear_candidates else None


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _unpack_transformed_output(transformed):
    if isinstance(transformed, tuple) and len(transformed) == 2:
        return transformed
    return transformed, None


def _normalize_hw(size):
    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError(f"Expected size of length 2, got: {size}")
        return int(size[0]), int(size[1])
    return int(size), int(size)


class RandomResizedCropWithMetadata:
    def __init__(self, image_size: int, mean, std):
        self.random_crop = torchvision.transforms.RandomResizedCrop(image_size)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

    def __call__(self, image):
        orig_width, orig_height = image.size
        top, left, height, width = self.random_crop.get_params(
            image, self.random_crop.scale, self.random_crop.ratio
        )
        out_height, out_width = _normalize_hw(self.random_crop.size)
        cropped = TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [out_height, out_width],
            self.random_crop.interpolation,
        )
        tensor = self.to_tensor(cropped)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        meta = {
            "transform_name": "random_resized_crop",
            "orig_height": int(orig_height),
            "orig_width": int(orig_width),
            "crop_top": int(top),
            "crop_left": int(left),
            "crop_height": int(height),
            "crop_width": int(width),
            "output_height": int(out_height),
            "output_width": int(out_width),
        }
        return tensor, meta


def _unpack_dataset_item(item):
    if isinstance(item, tuple):
        if len(item) == 3:
            return item[0], item[1], item[2]
        if len(item) == 2:
            return item[0], item[1], None
    raise ValueError(f"Unsupported dataset item format: {type(item)}")


class WRESSynInputDataset(Dataset):
    """
    Expected layout:
      root/
        class_a/input/*.png
        class_b/input/*.png
    """

    def __init__(
        self,
        root: str,
        transform=None,
        input_subdir: str = "input",
        class_to_idx: Optional[Dict[str, int]] = None,
        max_samples_per_class: int = 0,
        seed: int = 42,
        allow_missing_classes: bool = False,
    ):
        self.root = root
        self.transform = transform
        self.input_subdir = input_subdir
        self.max_samples_per_class = max_samples_per_class

        if not os.path.isdir(root):
            raise ValueError(f"Dataset root not found: {root}")

        if class_to_idx is None:
            class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            if not class_names:
                raise ValueError(f"No class folders found in: {root}")
            clear_name = _find_clear_class_name(class_names)
            if clear_name is not None:
                class_names = [clear_name] + [name for name in class_names if name != clear_name]
                print(f"[ClassIndex] clear class '{clear_name}' fixed to index 0")
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        else:
            self.class_to_idx = dict(class_to_idx)
            mapped_class_names = sorted(self.class_to_idx.keys())
            clear_name = _find_clear_class_name(mapped_class_names)
            if clear_name is not None and self.class_to_idx[clear_name] != 0:
                raise ValueError(
                    f"clear class '{clear_name}' must be mapped to index 0, "
                    f"but got {self.class_to_idx[clear_name]}."
                )

        self.classes = [name for name, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]
        self.samples = []
        self.targets = []

        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                msg = f"Class folder missing in {root}: {class_name}"
                if allow_missing_classes:
                    print(f"[Warning] {msg}. Skipping.")
                    continue
                raise ValueError(msg)

            scan_root = os.path.join(class_dir, input_subdir)
            if not os.path.isdir(scan_root):
                print(f"[Warning] '{input_subdir}' not found for class '{class_name}'. Falling back to class root.")
                scan_root = class_dir

            image_paths = []
            for current_root, _, files in os.walk(scan_root):
                for fname in files:
                    full_path = os.path.join(current_root, fname)
                    if is_image_file(full_path):
                        image_paths.append(full_path)

            image_paths = sorted(image_paths)
            if not image_paths:
                print(f"[Warning] No images found for class '{class_name}' under: {scan_root}")
                continue

            if self.max_samples_per_class > 0 and len(image_paths) > self.max_samples_per_class:
                rng = np.random.default_rng(seed + self.class_to_idx[class_name])
                selected = rng.choice(len(image_paths), size=self.max_samples_per_class, replace=False)
                image_paths = [image_paths[i] for i in sorted(selected.tolist())]

            class_idx = self.class_to_idx[class_name]
            self.samples.extend((p, class_idx) for p in image_paths)
            self.targets.extend([class_idx] * len(image_paths))
            print(f"Loaded {len(image_paths)} images for class {class_idx} ({class_name})")

        if not self.samples:
            raise ValueError(f"No usable images found in dataset root: {root}")

        print(f"Total samples loaded from {root}: {len(self.samples)}")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        transform_meta = None
        if self.transform is not None:
            image, transform_meta = _unpack_transformed_output(self.transform(image))
        meta = {
            "image_path": os.path.abspath(image_path),
            "rel_path": _normalize_relpath(os.path.relpath(image_path, self.root)),
            "dataset_root": os.path.abspath(self.root),
            "label": int(label),
            "class_name": self.classes[label],
            "source": "syn",
        }
        if transform_meta is not None:
            meta.update(_to_jsonable(transform_meta))
        return image, label, meta


class IndexedSubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int], name: str = "subset"):
        self.dataset = dataset
        self.indices = list(indices)
        self.name = name
        self.classes = getattr(dataset, "classes", None)
        self.class_to_idx = getattr(dataset, "class_to_idx", None)
        base_targets = getattr(dataset, "targets", None)
        if base_targets is not None:
            self.targets = [base_targets[i] for i in self.indices]
        else:
            self.targets = [dataset[i][1] for i in self.indices]

        print(f"[Split] {self.name}: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label, meta = _unpack_dataset_item(self.dataset[self.indices[idx]])
        meta = dict(meta or {})
        meta["subset_name"] = self.name
        meta["subset_index"] = int(idx)
        meta["base_sample_index"] = int(self.indices[idx])
        return image, label, meta


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_labels, class_names):
        self.dataset = dataset
        self.base_labels = set(base_labels)

        print("Filtering dataset for finetuning...")
        print(f"Total samples: {len(dataset)}")
        print(f"Base classes: {[class_names[i] for i in sorted(base_labels)]}")

        if hasattr(dataset, "targets"):
            targets = dataset.targets
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]

        self.indices = [i for i, label in enumerate(targets) if label in self.base_labels]

        print("\nFiltering complete!")
        print(f"  Selected {len(self.indices)} samples from {len(base_labels)} base classes")
        print(f"  Filtered out {len(dataset) - len(self.indices)} samples\n")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label, _ = _unpack_dataset_item(self.dataset[real_idx])
        return {"images": image, "labels": label}


def collate_fn(batch):
    images = []
    labels = []
    meta = []
    has_meta = False

    for item in batch:
        image, label, item_meta = _unpack_dataset_item(item)
        images.append(image)
        labels.append(label)
        meta.append(item_meta)
        has_meta = has_meta or item_meta is not None

    collated = {
        "images": torch.stack(images),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if has_meta:
        collated["meta"] = meta
    return collated


def infer_features_labels(dino, data_loader, features_dir, labels_dir, device, manifest_path=None, feature_split="train"):
    dino.to(device)
    dino.eval()

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    if manifest_path is not None:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    from tqdm import tqdm

    print(f"Extracting features to {features_dir}...")
    row_offset = 0
    manifest_file = open(manifest_path, "w", encoding="utf-8") if manifest_path is not None else None
    try:
        for bidx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Feature extraction"):
            images = batch["images"].to(device)

            with torch.no_grad():
                features = dino(images).pooler_output

            labels_np = batch["labels"].cpu().numpy()
            np.save(os.path.join(features_dir, f"features_{bidx}.npy"), features.cpu().numpy())
            np.save(os.path.join(labels_dir, f"labels_{bidx}.npy"), labels_np)

            if manifest_file is not None:
                batch_meta = batch.get("meta") or [None] * len(labels_np)
                for sample_offset, label in enumerate(labels_np.tolist()):
                    record = dict(batch_meta[sample_offset] or {})
                    record["feature_row"] = int(row_offset + sample_offset)
                    record["feature_split"] = feature_split
                    record["label"] = int(label)
                    manifest_file.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
            row_offset += len(labels_np)
    finally:
        if manifest_file is not None:
            manifest_file.close()


def _normalize_relpath(path: str) -> str:
    return path.replace("\\", "/")


def _read_test_info_file(info_path: str):
    meta = {}
    entries = []
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                payload = line[1:].strip()
                if "=" in payload:
                    k, v = payload.split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            entries.append(line)
    return meta, entries


def _write_test_info_file(
    info_path: str,
    class_name: str,
    rel_paths: List[str],
    test_split: float,
    seed: int,
    pool_size: int,
    max_samples_per_class: int,
):
    with open(info_path, "w") as f:
        f.write("# Auto-generated test split manifest\n")
        f.write(f"# class={class_name}\n")
        f.write(f"# test_split={test_split}\n")
        f.write(f"# seed={seed}\n")
        f.write(f"# pool_size={pool_size}\n")
        f.write(f"# max_samples_per_class={max_samples_per_class}\n")
        for rel in rel_paths:
            f.write(f"{rel}\n")


def split_with_persistent_test_info(
    dataset: WRESSynInputDataset,
    test_split: float,
    seed: int,
    test_info_filename: str,
    max_samples_per_class: int = 0,
    force_resplit: bool = False,
):
    if not (0.0 <= test_split < 1.0):
        raise ValueError(f"--test_split must be in [0, 1). Got: {test_split}")

    train_indices = []
    test_indices = []
    train_index_set = set()
    test_index_set = set()

    print(f"[Split] Persistent per-class split with test ratio={test_split:.4f}")
    print(f"[Split] test info filename: {test_info_filename}")
    if max_samples_per_class > 0:
        print(f"[Split] max_samples_per_class (TOTAL before split): {max_samples_per_class}")
    if force_resplit:
        print("[Split] --force_resplit enabled: existing test_data_info files will be overwritten.")

    for class_name in dataset.classes:
        class_idx = dataset.class_to_idx[class_name]
        class_dir = os.path.join(dataset.root, class_name)
        info_path = os.path.join(class_dir, test_info_filename)

        class_items = []
        rel_to_index = {}
        for sample_idx, (image_path, label) in enumerate(dataset.samples):
            if label != class_idx:
                continue
            rel = _normalize_relpath(os.path.relpath(image_path, class_dir))
            class_items.append((sample_idx, rel))
            rel_to_index[rel] = sample_idx

        n = len(class_items)
        if n == 0:
            print(f"[Split] class {class_idx} ({class_name}): 0 samples (skip)")
            continue

        # Make per-class sampling independent from branch path (existing vs generated)
        # so all DDP ranks reproduce identical pools.
        pool_rng = np.random.default_rng(seed + class_idx * 10007 + 17)
        test_rng = np.random.default_rng(seed + class_idx * 10007 + 29)

        if max_samples_per_class > 0 and n > max_samples_per_class:
            class_indices_full = np.array([sample_idx for sample_idx, _ in class_items], dtype=np.int64)
            pool_rng.shuffle(class_indices_full)
            selected_indices = set(class_indices_full[:max_samples_per_class].tolist())
            pool_items = [(sample_idx, rel) for sample_idx, rel in class_items if sample_idx in selected_indices]
        else:
            pool_items = list(class_items)

        pool_rel_to_index = {rel: sample_idx for sample_idx, rel in pool_items}
        n_pool = len(pool_items)

        if n_pool == 1 or test_split == 0.0:
            expected_n_test = 0
        else:
            expected_n_test = int(round(n_pool * test_split))
            expected_n_test = max(1, expected_n_test)
            expected_n_test = min(expected_n_test, n_pool - 1)

        if os.path.exists(info_path) and not force_resplit:
            meta, rel_paths = _read_test_info_file(info_path)
            missing = [rel for rel in rel_paths if rel not in pool_rel_to_index]
            if missing:
                missing_preview = ", ".join(missing[:5])
                raise ValueError(
                    f"Split info mismatch for class '{class_name}' at {info_path}. "
                    f"{len(missing)} entries are not found in current split pool "
                    f"(e.g., {missing_preview}). Use --force_resplit to regenerate."
                )

            if len(rel_paths) != expected_n_test:
                raise ValueError(
                    f"Split info size mismatch for class '{class_name}' at {info_path}. "
                    f"expected {expected_n_test} test entries but found {len(rel_paths)}. "
                    f"Use --force_resplit to regenerate for current settings."
                )

            meta_cap = int(meta.get("max_samples_per_class", "0")) if "max_samples_per_class" in meta else None
            if meta_cap is not None and meta_cap != max_samples_per_class:
                raise ValueError(
                    f"Split info config mismatch for class '{class_name}' at {info_path}. "
                    f"file max_samples_per_class={meta_cap}, current={max_samples_per_class}. "
                    f"Use --force_resplit to regenerate."
                )

            class_test_indices = [pool_rel_to_index[rel] for rel in rel_paths]
            class_test_indices = sorted(set(class_test_indices))
            source = "existing"
        else:
            class_indices = np.array([sample_idx for sample_idx, _ in pool_items], dtype=np.int64)
            test_rng.shuffle(class_indices)

            if test_split == 0.0 or n_pool == 1:
                n_test = 0
            else:
                n_test = int(round(n_pool * test_split))
                n_test = max(1, n_test)
                n_test = min(n_test, n_pool - 1)

            class_test_indices = class_indices[:n_test].tolist()
            class_test_set = set(class_test_indices)
            rel_paths = sorted([rel for sample_idx, rel in pool_items if sample_idx in class_test_set])
            _write_test_info_file(
                info_path,
                class_name,
                rel_paths,
                test_split,
                seed,
                n_pool,
                max_samples_per_class,
            )
            class_test_indices = sorted(class_test_set)
            source = "generated"

        class_test_set = set(class_test_indices)
        class_train_indices = [sample_idx for sample_idx, _ in pool_items if sample_idx not in class_test_set]

        for idx in class_train_indices:
            if idx not in train_index_set:
                train_index_set.add(idx)
                train_indices.append(idx)
        for idx in class_test_indices:
            if idx not in test_index_set:
                test_index_set.add(idx)
                test_indices.append(idx)

        print(
            f"[Split] class {class_idx} ({class_name}): total={n}, pool={n_pool}, "
            f"train={len(class_train_indices)}, test={len(class_test_indices)} [{source}]"
        )

    if len(train_indices) == 0:
        raise ValueError("Stratified split produced zero training samples. Check dataset and --test_split.")

    return train_indices, test_indices


def generate_class_order(num_classes, base, seed, class_to_idx, novel_class_names=None):
    if base <= 0 or base > num_classes:
        raise ValueError(f"Invalid --labeled_classes ({base}). Must be in [1, {num_classes}]")

    all_indices = list(range(num_classes))
    rng = np.random.default_rng(seed)
    clear_name = _find_clear_class_name(list(class_to_idx.keys()))
    clear_idx = class_to_idx[clear_name] if clear_name is not None else None

    if novel_class_names:
        expected_novel = num_classes - base
        if len(novel_class_names) != expected_novel:
            raise ValueError(
                f"Expected {expected_novel} novel classes, but got {len(novel_class_names)}: {novel_class_names}"
            )

        novel_indices = []
        for name in novel_class_names:
            if name not in class_to_idx:
                raise ValueError(f"Class '{name}' not found in dataset classes: {list(class_to_idx.keys())}")
            novel_indices.append(class_to_idx[name])

        if len(set(novel_indices)) != len(novel_indices):
            raise ValueError(f"Duplicate class found in --novel_classes: {novel_class_names}")

        base_indices = [i for i in all_indices if i not in set(novel_indices)]
        if clear_idx is not None and clear_idx in novel_indices:
            raise ValueError(
                f"clear class '{clear_name}' cannot be in --novel_classes. "
                "clear must remain base so classifier label 0 stays clear."
            )
        if clear_idx is not None and clear_idx in base_indices:
            shuffled_rest = [i for i in base_indices if i != clear_idx]
            rng.shuffle(shuffled_rest)
            base_indices = [clear_idx] + shuffled_rest
        else:
            rng.shuffle(base_indices)
        class_order = base_indices + novel_indices
        return class_order

    if base == num_classes:
        if clear_idx is not None and clear_idx in all_indices:
            return [clear_idx] + [i for i in all_indices if i != clear_idx]
        return all_indices

    rng.shuffle(all_indices)
    if clear_idx is not None and clear_idx in all_indices:
        all_indices = [clear_idx] + [i for i in all_indices if i != clear_idx]
    return all_indices


def merge_npy(features_dir, labels_dir, prefix, model_name, output_dir):
    print(f"Merging .npy files from {features_dir}...")
    feature_files = sorted([os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith(".npy")])
    label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith(".npy")])

    assert len(feature_files) == len(label_files), "Mismatch in number of feature and label files"

    def merged_array(files):
        arrays = [np.load(f) for f in files]
        return np.concatenate(arrays, axis=0)

    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    print(f"Saving merged features to {output_dir}/{model_name}/")
    np.save(f"{output_dir}/{model_name}/{prefix['feature']}-{model_name}.npy", merged_array(feature_files))
    np.save(f"{output_dir}/{model_name}/{prefix['label']}-{model_name}.npy", merged_array(label_files))
    print(f"Merged {len(feature_files)} files successfully!")


def save_class_metadata(exp_root_dir, model_name, classes: List[str], class_order: List[int]):
    class_order_file = os.path.join(exp_root_dir, model_name, "class_order.txt")
    os.makedirs(os.path.dirname(class_order_file), exist_ok=True)
    with open(class_order_file, "w") as f:
        f.write(",".join(map(str, class_order)))
    print(f"\nClass order saved to: {class_order_file}")

    class_names_file = os.path.join(exp_root_dir, model_name, "class_names.txt")
    with open(class_names_file, "w") as f:
        for idx, name in enumerate(classes):
            f.write(f"{idx},{name}\n")
    print(f"Class names saved to: {class_names_file}\n")

    class_mappings_file = os.path.join(exp_root_dir, model_name, "class_mappings.json")
    classifier2name = {str(i): classes[orig_id] for i, orig_id in enumerate(class_order)}
    name2classifier = {name: int(i) for i, name in classifier2name.items()}
    orig2name = {str(i): name for i, name in enumerate(classes)}
    name2orig = {name: i for i, name in enumerate(classes)}

    mapping_data = {
        "classifier2orig": {str(i): int(orig_id) for i, orig_id in enumerate(class_order)},
        "orig2classifier": {str(int(orig_id)): i for i, orig_id in enumerate(class_order)},
        "classifier2name": classifier2name,
        "name2classifier": name2classifier,
        "orig2name": orig2name,
        "name2orig": name2orig,
        "class_order": [int(x) for x in class_order],
    }
    with open(class_mappings_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=4, ensure_ascii=False)
    print(f"Class mappings saved to: {class_mappings_file}\n")


def setup_distributed(args_device: str):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0
    is_main = not distributed or local_rank == 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        run_device = f"cuda:{local_rank}"
    else:
        run_device = args_device

    return distributed, local_rank, is_main, run_device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DINO finetuning + feature extraction on WRES synthetic train dataset"
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--model", default="dino_vitb16", type=str)
    parser.add_argument("--exp_name", type=str, required=True, help="Specific experiment name")
    parser.add_argument("--output_dir", default="wres_datasets", type=str)
    parser.add_argument("--train_root", default="wres_datasets/syn_train_dataset", type=str)
    parser.add_argument("--test_root", default="", type=str)
    parser.add_argument("--test_split", default=0.1, type=float)
    parser.add_argument("--test_info_filename", default="test_data_info.txt", type=str)
    parser.add_argument("--force_resplit", action="store_true")
    parser.add_argument("--input_subdir", default="input", type=str)
    parser.add_argument("--labeled_classes", default=-1, type=int)
    parser.add_argument("--novel_classes", type=str, nargs="+", help="Class names to treat as novel")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--max_samples_per_class", default=0, type=int)
    args = parser.parse_args()

    distributed, local_rank, is_main, run_device = setup_distributed(args.device)
    if is_main:
        print(f"[DDP] distributed={distributed}, local_rank={local_rank}, device={run_device}")

    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = args.img_size  # 224
    transforms = RandomResizedCropWithMetadata(image_size=image_size, mean=mean, std=std)

    full_train_dataset = WRESSynInputDataset(
        root=args.train_root,
        transform=transforms,
        input_subdir=args.input_subdir,
        max_samples_per_class=0,
        seed=args.seed,
    )

    num_classes = len(full_train_dataset.classes)
    if args.labeled_classes <= 0:
        args.labeled_classes = num_classes
    if args.labeled_classes > num_classes:
        raise ValueError(
            f"--labeled_classes ({args.labeled_classes}) cannot exceed detected class count ({num_classes})."
        )

    if args.test_root:
        train_dataset = full_train_dataset
        test_dataset = WRESSynInputDataset(
            root=args.test_root,
            transform=transforms,
            input_subdir=args.input_subdir,
            class_to_idx=full_train_dataset.class_to_idx,
            max_samples_per_class=args.max_samples_per_class,
            seed=args.seed,
            allow_missing_classes=True,
        )
        print("[Info] --test_root provided. Ignoring --test_split.")
    else:
        if distributed:
            if is_main:
                train_indices, test_indices = split_with_persistent_test_info(
                    dataset=full_train_dataset,
                    test_split=args.test_split,
                    seed=args.seed,
                    test_info_filename=args.test_info_filename,
                    max_samples_per_class=args.max_samples_per_class,
                    force_resplit=args.force_resplit,
                )
            dist.barrier()
            if not is_main:
                train_indices, test_indices = split_with_persistent_test_info(
                    dataset=full_train_dataset,
                    test_split=args.test_split,
                    seed=args.seed,
                    test_info_filename=args.test_info_filename,
                    max_samples_per_class=args.max_samples_per_class,
                    force_resplit=False,
                )
        else:
            train_indices, test_indices = split_with_persistent_test_info(
                dataset=full_train_dataset,
                test_split=args.test_split,
                seed=args.seed,
                test_info_filename=args.test_info_filename,
                max_samples_per_class=args.max_samples_per_class,
                force_resplit=args.force_resplit,
            )
        train_dataset = IndexedSubsetDataset(full_train_dataset, train_indices, name="train split")
        if len(test_indices) == 0:
            print("[Warning] test split is empty. Reusing train split for verification/test feature export.")
            test_dataset = train_dataset
        else:
            test_dataset = IndexedSubsetDataset(full_train_dataset, test_indices, name="test split")

    class_order = generate_class_order(
        num_classes=num_classes,
        base=args.labeled_classes,
        seed=args.seed,
        class_to_idx=full_train_dataset.class_to_idx,
        novel_class_names=args.novel_classes,
    )

    idx_to_class = {idx: name for name, idx in full_train_dataset.class_to_idx.items()}
    base_names = [f"{i}({idx_to_class[i]})" for i in class_order[: args.labeled_classes]]
    novel_names = [f"{i}({idx_to_class[i]})" for i in class_order[args.labeled_classes :]]

    print(f"\n{'='*70}")
    print(f"WRES class order (seed={args.seed}): {class_order}")
    print(f"  Base classes ({args.labeled_classes}): {base_names}")
    print(f"  Novel classes ({num_classes - args.labeled_classes}): {novel_names}")
    print(f"{'='*70}\n")

    base_labels = class_order[: args.labeled_classes]
    finetune_dataset = FilteredDataset(train_dataset, base_labels, train_dataset.classes)
    finetune_test_dataset = FilteredDataset(test_dataset, base_labels, test_dataset.classes)
    print(f"[Finetune] train samples={len(finetune_dataset)}, test samples={len(finetune_test_dataset)}")

    exp_root_dir = os.path.join(args.output_dir, args.exp_name)
    model_name = args.model.replace("_", "-") + "-sl"
    checkpoint_dir = os.path.join(exp_root_dir, model_name, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if distributed:
        dino = finetune_dino_ddp(
            finetune_dataset,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=args.model,
            save_dir=checkpoint_dir,
            seed=args.seed,
            test_set=finetune_test_dataset,
        )
    else:
        dino = finetune_dino_single(
            finetune_dataset,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=args.model,
            save_dir=checkpoint_dir,
            seed=args.seed,
            test_set=finetune_test_dataset,
        )

    if distributed:
        dist.barrier()

    if is_main:
        if distributed:
            ckpt_dino = load_finetuned_model_from_checkpoint_ddp(
                checkpoint_dir=checkpoint_dir,
                num_classes=num_classes,
                model_name=args.model,
                device=run_device,
                seed=args.seed,
            )
        else:
            ckpt_dino = load_finetuned_model_from_checkpoint_single(
                checkpoint_dir=checkpoint_dir,
                num_classes=num_classes,
                model_name=args.model,
                device=run_device,
                seed=args.seed,
            )

        verify_checkpoint_loading(
            dino_trained=dino,
            dino_checkpoint=ckpt_dino,
            test_dataset=test_dataset,
            collate_fn=collate_fn,
            device=run_device,
            batch_size=64,
        )

        save_class_metadata(exp_root_dir, model_name, full_train_dataset.classes, class_order)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            shuffle=False,
        )
        train_features_dir = f"{exp_root_dir}/{args.model}_features"
        train_labels_dir = f"{exp_root_dir}/{args.model}_labels"
        train_manifest_path = os.path.join(exp_root_dir, model_name, f"features_manifest-{model_name}.jsonl")
        infer_features_labels(
            dino,
            train_loader,
            train_features_dir,
            train_labels_dir,
            run_device,
            manifest_path=train_manifest_path,
            feature_split="train",
        )
        merge_npy(
            train_features_dir, train_labels_dir, {"feature": "features", "label": "labels"}, model_name, exp_root_dir
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            shuffle=False,
        )
        test_features_dir = f"{exp_root_dir}/{args.model}_test_features"
        test_labels_dir = f"{exp_root_dir}/{args.model}_test_labels"
        test_manifest_path = os.path.join(exp_root_dir, model_name, f"test_features_manifest-{model_name}.jsonl")
        infer_features_labels(
            dino,
            test_loader,
            test_features_dir,
            test_labels_dir,
            run_device,
            manifest_path=test_manifest_path,
            feature_split="test",
        )
        merge_npy(
            test_features_dir,
            test_labels_dir,
            {"feature": "test_features", "label": "test_labels"},
            model_name,
            exp_root_dir,
        )

        print("\nFeature extraction completed!")
        print(f"Output directory: {exp_root_dir}/{model_name}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
