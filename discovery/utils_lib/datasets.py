"""Image datasets, persistent train/test splits, and feature export for degradation discovery."""

import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def feature_transform(image_size=224):
    """Random resized crop with ImageNet normalization, used for training and feature export."""
    return T.Compose(
        [T.RandomResizedCrop(image_size), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )


def list_images(directory):
    """Image paths relative to ``directory`` (recursive, sorted)."""
    paths = []
    for root, _, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.relpath(os.path.join(root, name), directory).replace("\\", "/"))
    return sorted(paths)


class LabeledImageDataset(Dataset):
    """List of (image path, label). Returns (transformed image, label)."""

    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def collate(batch):
    return {"images": torch.stack([image for image, _ in batch]), "labels": torch.tensor([l for _, l in batch])}


# ---------------------------------------------------------------------------
# Synthetic (known) classes
# ---------------------------------------------------------------------------


def synthetic_classes(root):
    """Class folders of ``root`` with ``clear`` fixed to index 0 and the rest sorted."""
    names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if "clear" in names:
        names = ["clear"] + [n for n in names if n != "clear"]
    return names


def synthetic_samples(root, classes, input_subdir="input"):
    samples = []
    for label, name in enumerate(classes):
        scan_root = os.path.join(root, name, input_subdir)
        if not os.path.isdir(scan_root):
            scan_root = os.path.join(root, name)
        paths = [os.path.join(scan_root, p) for p in list_images(scan_root)]
        samples.extend((p, label) for p in paths)
        print(f"[Data] {len(paths)} images for class {label} ({name})")
    return samples


def _read_split_file(path):
    meta, entries = {}, []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].strip().split("=", 1)
                    meta[key.strip()] = value.strip()
                continue
            entries.append(line)
    return meta, entries


def _num_test(n, test_split):
    if n == 1 or test_split == 0.0:
        return 0
    return min(max(1, int(round(n * test_split))), n - 1)


def split_synthetic(root, classes, samples, test_split, seed, max_per_class, split_filename="test_data_info.txt"):
    """Per class, draw a pool of at most ``max_per_class`` images and hold out ``test_split`` of it.

    The held-out images are stored in ``<root>/<class>/<split_filename>`` and reused when present.
    Returns train and test sample indices.
    """
    train_indices, test_indices = [], []
    for label, name in enumerate(classes):
        class_dir = os.path.join(root, name)
        items = [
            (i, os.path.relpath(p, class_dir).replace("\\", "/")) for i, (p, y) in enumerate(samples) if y == label
        ]
        if not items:
            continue
        pool_rng = np.random.default_rng(seed + label * 10007 + 17)
        test_rng = np.random.default_rng(seed + label * 10007 + 29)
        if max_per_class > 0 and len(items) > max_per_class:
            shuffled = np.array([i for i, _ in items], dtype=np.int64)
            pool_rng.shuffle(shuffled)
            selected = set(shuffled[:max_per_class].tolist())
            items = [(i, rel) for i, rel in items if i in selected]
        index_of = {rel: i for i, rel in items}
        n_test = _num_test(len(items), test_split)

        split_path = os.path.join(class_dir, split_filename)
        if os.path.exists(split_path):
            meta, rel_paths = _read_split_file(split_path)
            missing = [rel for rel in rel_paths if rel not in index_of]
            if (
                missing
                or len(rel_paths) != n_test
                or int(meta.get("max_samples_per_class", max_per_class)) != max_per_class
            ):
                raise ValueError(f"{split_path} does not match the current data or settings; delete it to regenerate.")
            class_test = sorted({index_of[rel] for rel in rel_paths})
            source = "existing"
        else:
            shuffled = np.array([i for i, _ in items], dtype=np.int64)
            test_rng.shuffle(shuffled)
            class_test = set(shuffled[:n_test].tolist())
            with open(split_path, "w") as f:
                f.write(f"# Auto-generated test split manifest\n# class={name}\n# test_split={test_split}\n")
                f.write(f"# seed={seed}\n# pool_size={len(items)}\n# max_samples_per_class={max_per_class}\n")
                f.writelines(f"{rel}\n" for rel in sorted(rel for i, rel in items if i in class_test))
            class_test = sorted(class_test)
            source = "generated"
        test_set = set(class_test)
        class_train = [i for i, _ in items if i not in test_set]
        train_indices.extend(class_train)
        test_indices.extend(class_test)
        print(
            f"[Split] class {label} ({name}): pool={len(items)}, train={len(class_train)}, test={len(class_test)} [{source}]"
        )
    return train_indices, test_indices


def save_class_metadata(feature_dir, classes):
    """class_order.txt, class_names.txt, and class_mappings.json for an identity class order."""
    os.makedirs(feature_dir, exist_ok=True)
    order = list(range(len(classes)))
    with open(os.path.join(feature_dir, "class_order.txt"), "w") as f:
        f.write(",".join(map(str, order)))
    with open(os.path.join(feature_dir, "class_names.txt"), "w") as f:
        f.writelines(f"{i},{name}\n" for i, name in enumerate(classes))
    mapping = {
        "classifier2orig": {str(i): i for i in order},
        "orig2classifier": {str(i): i for i in order},
        "classifier2name": {str(i): classes[i] for i in order},
        "name2classifier": {classes[i]: i for i in order},
        "orig2name": {str(i): name for i, name in enumerate(classes)},
        "name2orig": {name: i for i, name in enumerate(classes)},
        "class_order": order,
    }
    with open(os.path.join(feature_dir, "class_mappings.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Real (unlabeled) classes
# ---------------------------------------------------------------------------


def split_real_class(class_dir, class_name, class_offset, test_split, seed, split_filename="test_data_info.txt"):
    """Hold out ``test_split`` of a real class; reuse ``<class_dir>/<split_filename>`` when present.
    Returns (train relative paths, test relative paths)."""
    rel_paths = list_images(class_dir)
    if not rel_paths:
        raise ValueError(f"No images found in {class_dir}")
    split_path = os.path.join(class_dir, split_filename)
    if os.path.exists(split_path):
        _, existing = _read_split_file(split_path)
        missing = [r for r in existing if r not in set(rel_paths)]
        if missing:
            raise ValueError(f"{split_path} lists {len(missing)} missing images; delete it to regenerate.")
        test = sorted(set(existing))
    else:
        rng = np.random.default_rng(seed + class_offset * 10007 + 43)
        order = np.arange(len(rel_paths))
        rng.shuffle(order)
        test = sorted(rel_paths[i] for i in order[: _num_test(len(rel_paths), test_split)])
        with open(split_path, "w") as f:
            f.write(
                f"# Auto-generated split manifest\n# class={class_name}\n# test_split={test_split}\n# seed={seed}\n"
            )
            f.writelines(f"{rel}\n" for rel in test)
    test_set = set(test)
    return [r for r in rel_paths if r not in test_set], test


# ---------------------------------------------------------------------------
# Feature export
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(backbone, loader, device, batch_order="loader"):
    """Pooled DINO features z and labels of every sample.

    ``batch_order="filename"`` concatenates the batches in the lexicographic order of
    ``features_<i>.npy`` (0, 1, 10, 11, ..., 2, ...), the row order of the released synthetic features.
    """
    backbone.to(device).eval()
    features, labels = [], []
    for batch in tqdm(loader, desc="Feature extraction"):
        features.append(backbone(batch["images"].to(device)).pooler_output.cpu().numpy())
        labels.append(batch["labels"].numpy())
    order = list(range(len(features)))
    if batch_order == "filename":
        order.sort(key=lambda i: f"features_{i}.npy")
    return np.concatenate([features[i] for i in order]), np.concatenate([labels[i] for i in order])
