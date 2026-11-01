"""Degradation classes and datasets for paired, unpaired, and real-world evaluation images."""

import json
import os
import random
import re
from dataclasses import dataclass
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tvF

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Known / novel degradation classes
# ---------------------------------------------------------------------------


@dataclass
class DegradationClasses:
    """Known (labeled) and novel (discovered) degradation classes of one stage."""

    deg_map: dict  # class name -> original index
    known: list  # labeled classes available at the initial stage
    novel: list  # classes discovered by VB-CGCD
    orig2classifier: dict  # original index -> classifier index

    def classifier_index(self, name):
        return self.orig2classifier[self.deg_map[name]]


def load_degradation_classes(cfg):
    deg_map = dict(vars(cfg.deg_map))
    with open(cfg.discovery.class_order, "r") as f:
        class_order = [int(x) for x in f.read().strip().split(",")]
    base, inc = cfg.incremental.base_class_num, cfg.incremental.inc_class_num
    idx_to_class = {v: k for k, v in deg_map.items()}
    known = [idx_to_class[i] for i in class_order[:base]]
    novel = [idx_to_class[i] for i in class_order[base : base + inc]]

    with open(cfg.discovery.class_mappings, "r", encoding="utf-8") as f:
        orig2classifier = {int(k): int(v) for k, v in json.load(f)["orig2classifier"].items()}

    print(f"[Classes] known ({len(known)}): {known}")
    print(f"[Classes] novel ({len(novel)}): {novel}")
    print(f"[Classes] orig2classifier: {orig2classifier}")
    return DegradationClasses(deg_map, known, novel, orig2classifier)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def collect_images(path):
    return [
        p
        for p in sorted(glob(os.path.join(path, "*")))
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXTENSIONS
    ]


def load_img(filename):
    return (np.array(Image.open(filename).convert("RGB")) / 255.0).astype(np.float32)


def crop_img(image, base=16):
    h, w = image.shape[0], image.shape[1]
    crop_h, crop_w = h % base, w % base
    return image[crop_h // 2 : h - crop_h + crop_h // 2, crop_w // 2 : w - crop_w + crop_w // 2, :]


def _build_unique_key_map(paths, key_fn):
    key_to_path, duplicated = {}, set()
    for p in paths:
        key = key_fn(os.path.splitext(os.path.basename(p))[0])
        if key in key_to_path:
            duplicated.add(key)
        else:
            key_to_path[key] = p
    for k in duplicated:
        key_to_path.pop(k, None)
    return key_to_path


def _leading_number(stem):
    m = re.match(r"^(\d+)", stem)
    return m.group(1) if m else None


def _match_gt_path(input_path, gt_by_stem, gt_by_first_token, gt_by_leading_num):
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    if input_stem in gt_by_stem:  # exact stem
        return gt_by_stem[input_stem]
    tokens = input_stem.split("_")
    for k in range(len(tokens) - 1, 0, -1):  # prefix token, e.g. im_0001_s100_a04 -> im_0001
        candidate = "_".join(tokens[:k])
        if candidate in gt_by_stem:
            return gt_by_stem[candidate]
    if tokens[0] in gt_by_first_token:  # unique first token, e.g. 0_rain -> 0_clean
        return gt_by_first_token[tokens[0]]
    leading_num = _leading_number(input_stem)  # unique leading number
    if leading_num is not None and leading_num in gt_by_leading_num:
        return gt_by_leading_num[leading_num]
    return None


def build_pairs(gt_dir, input_dir, deg_name):
    gt_paths = collect_images(gt_dir)
    gt_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_paths}
    gt_by_first_token = _build_unique_key_map(gt_paths, lambda s: s.split("_")[0])
    gt_by_leading_num = _build_unique_key_map(gt_paths, _leading_number)
    gt_by_leading_num.pop(None, None)

    paired_gt, paired_input, unmatched = [], [], []
    for in_path in collect_images(input_dir):
        gt_path = _match_gt_path(in_path, gt_by_stem, gt_by_first_token, gt_by_leading_num)
        if gt_path is None:
            unmatched.append(in_path)
            continue
        paired_gt.append(gt_path)
        paired_input.append(in_path)

    if unmatched:
        preview = ", ".join(os.path.basename(x) for x in unmatched[:5])
        print(f"[Data] Warning: {deg_name} has {len(unmatched)} unmatched inputs (first 5: {preview})")
    if not paired_gt:
        raise ValueError(f"[Data] No valid GT-input pairs found for class {deg_name}")
    return paired_gt, paired_input


# ---------------------------------------------------------------------------
# Paired (labeled) data
# ---------------------------------------------------------------------------


class PairedPatchDataset(Dataset):
    """Random 224x224 crops of a GT/input pair with rotation and flip."""

    def __init__(self, gt_paths, input_paths, deg_name, deg_class, patch_size=224):
        assert len(gt_paths) == len(input_paths)
        self.gt_paths = gt_paths
        self.input_paths = input_paths
        self.deg_name = deg_name
        self.deg_class = deg_class
        self.patch_size = patch_size

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt = cv2.imread(self.gt_paths[idx], cv2.IMREAD_COLOR)
        lq = cv2.imread(self.input_paths[idx], cv2.IMREAD_COLOR)
        if gt is None:
            raise ValueError(f"Failed to load GT image: {self.gt_paths[idx]}")
        if lq is None:
            raise ValueError(f"Failed to load input image: {self.input_paths[idx]}")

        pair = torch.stack(
            [tvF.to_image(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)), tvF.to_image(cv2.cvtColor(lq, cv2.COLOR_BGR2RGB))]
        )
        i, j, h, w = v2.RandomCrop.get_params(pair[0], (self.patch_size, self.patch_size))
        pair = pair[:, :, i : i + h, j : j + w]
        rot_k = [0, 3, 2, 1][random.randrange(4)]
        if rot_k > 0:
            pair = torch.rot90(pair, rot_k, dims=[2, 3])
        if random.random() > 0.5:
            pair = pair.flip(2)
        pair = pair.float().div_(255.0)
        return pair[0], pair[1], self.deg_class


def build_paired_dataset(classes, cfg, patch_size):
    """Paired data of the known classes listed in ``train.all_classes``, in that order."""
    datasets = []
    for name in cfg.train.all_classes:
        if name not in classes.known:
            continue
        class_dir = os.path.join(cfg.train.data_root_train, name)
        gt_dir = os.path.join(class_dir, getattr(cfg.train, "gt_subdir", "gt"))
        input_dir = os.path.join(class_dir, getattr(cfg.train, "input_subdir", "input"))
        if not (os.path.isdir(gt_dir) and os.path.isdir(input_dir)):
            print(f"[Data] Skip class '{name}' (missing dirs): gt={gt_dir}, input={input_dir}")
            continue
        gt_paths, input_paths = build_pairs(gt_dir, input_dir, name)
        datasets.append(PairedPatchDataset(gt_paths, input_paths, name, classes.classifier_index(name), patch_size))
        print(f"[Data] Loaded {len(datasets[-1])} paired images for known class: {name}")
    if not datasets:
        raise ValueError("No paired training data found.")
    combined = ConcatDataset(datasets)
    print(f"[Data] Total paired samples: {len(combined)}")
    return combined


def dataset_labels(dataset):
    """Per-sample class labels of a ConcatDataset (or Subset) of PairedPatchDataset."""
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset_labels(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    labels = []
    for sub in dataset.datasets:
        labels.extend([int(sub.deg_class)] * len(sub))
    return np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Unpaired (unlabeled) data with the reliable bank
# ---------------------------------------------------------------------------


class StrongAugmentation:
    """Semi-UIR strong augmentation for the student view."""

    def __init__(self):
        self.color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
        self.random_grayscale = transforms.RandomGrayscale(p=0.2)

    def __call__(self, image):
        strong = image
        if random.random() < 0.8:
            strong = self.color_jitter(strong)
        strong = self.random_grayscale(strong)
        if random.random() < 0.5:
            kernel_size = int(random.random() * 4.95)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            strong = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))(strong)
        return strong


class UnpairedBankDataset(Dataset):
    """Unpaired patch with its weak view, strong view, and reliable-bank pseudo-target.

    Returns (weak, strong, pseudo_target, pseudo_target_path).
    """

    def __init__(self, input_paths, bank_paths, deg_name, deg_class, fine_size=224):
        assert len(input_paths) == len(bank_paths)
        self.input_paths = input_paths
        self.bank_paths = bank_paths
        self.deg_name = deg_name
        self.deg_class = deg_class
        self.fine_size = int(fine_size)
        self.to_tensor = transforms.ToTensor()
        self.strong_aug = StrongAugmentation()

    def __len__(self):
        return len(self.input_paths)

    def _load(self, path):
        return Image.open(path).convert("RGB").resize((self.fine_size, self.fine_size), Image.Resampling.LANCZOS)

    def _blank(self):
        return Image.fromarray(np.zeros((self.fine_size, self.fine_size, 3), dtype=np.uint8))

    def __getitem__(self, idx):
        bank_path = self.bank_paths[idx]
        weak = self._load(self.input_paths[idx])
        strong = self.strong_aug(weak.copy())
        if os.path.exists(bank_path):
            try:
                pseudo = self._load(bank_path)
            except (OSError, ValueError):
                pseudo = self._blank()
        else:
            pseudo = self._blank()
        return self.to_tensor(weak), self.to_tensor(strong), self.to_tensor(pseudo), bank_path


def build_unpaired_dataset(classes, patches_dir, bank_dir, fine_size=224):
    """Unpaired patches of the novel classes: ``patches_dir/<class>/*``, bank at ``bank_dir/<class>/*``."""
    datasets = []
    for name in classes.novel:
        patch_dir = os.path.join(patches_dir, name)
        if not os.path.isdir(patch_dir):
            print(f"[Data] Warning: novel class patch dir not found: {patch_dir}")
            continue
        input_paths = collect_images(patch_dir)
        if not input_paths:
            print(f"[Data] Warning: no unpaired patches found for novel class: {name}")
            continue
        class_bank_dir = os.path.join(bank_dir, name)
        os.makedirs(class_bank_dir, exist_ok=True)
        bank_paths = [os.path.join(class_bank_dir, os.path.basename(p)) for p in input_paths]
        datasets.append(UnpairedBankDataset(input_paths, bank_paths, name, classes.classifier_index(name), fine_size))
        print(f"[Data] Loaded {len(input_paths)} unpaired patches for novel class: {name} (bank: {class_bank_dir})")
    if not datasets:
        raise ValueError("No unpaired patches found for the novel classes.")
    combined = ConcatDataset(datasets)
    print(f"[Data] Total unpaired patches: {len(combined)}")
    return combined


# ---------------------------------------------------------------------------
# Real-world evaluation data
# ---------------------------------------------------------------------------


class RealImageDataset(Dataset):
    """Full-resolution real image cropped to a multiple of 16. Returns (image, path)."""

    def __init__(self, paths, name="real_eval"):
        self.paths = paths
        self.name = name
        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return self.to_tensor(crop_img(load_img(path)).astype(np.float32)), path


def build_real_eval_datasets(root, set_names):
    """List of (dataset, set_name) for ``root/<set_name>/*``."""
    datasets = []
    for name in set_names:
        set_dir = os.path.join(root, name)
        if not os.path.isdir(set_dir):
            print(f"[Data] Warning: real eval set not found: {set_dir}")
            continue
        paths = collect_images(set_dir)
        if not paths:
            print(f"[Data] Warning: no images found in eval set: {set_dir}")
            continue
        datasets.append((RealImageDataset(paths, name=name), name))
        print(f"[Data] Loaded real eval set: {name} ({len(paths)} images)")
    return datasets


# ---------------------------------------------------------------------------
# Sampling for the supervised stage
# ---------------------------------------------------------------------------


def cap_per_class(dataset, labels, max_samples_per_class, seed):
    """Randomly keep at most ``max_samples_per_class`` samples of each class."""
    rng = np.random.default_rng(seed)
    selected = []
    for cls_id in sorted(np.unique(labels).tolist()):
        indices = np.where(labels == cls_id)[0]
        if len(indices) > max_samples_per_class:
            indices = np.sort(rng.choice(indices, size=max_samples_per_class, replace=False))
        selected.extend(indices.tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    print(f"[Data] Per-class cap {max_samples_per_class}: {len(labels)} -> {len(selected)} samples")
    return torch.utils.data.Subset(dataset, selected.tolist()), labels[selected]


class ClassBalancedDistributedSampler(torch.utils.data.Sampler):
    """Samples with replacement, each class with equal probability, sharded across ranks."""

    def __init__(self, labels, num_replicas, rank, seed=42):
        counts = {}
        for cls_id in labels.tolist():
            counts[int(cls_id)] = counts.get(int(cls_id), 0) + 1
        self.weights = torch.as_tensor(
            np.asarray([1.0 / counts[int(c)] for c in labels.tolist()], dtype=np.float64), dtype=torch.double
        )
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = max(1, int(np.ceil(len(labels) / float(num_replicas))))
        self.total_size = self.num_samples * num_replicas
        print(f"[Data] Class-balanced sampling, class counts: {counts}")

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(self.weights, self.total_size, True, generator=g).tolist()
        return iter(sampled[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
