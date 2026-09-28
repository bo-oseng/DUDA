import json
import os
import random
import re
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tvF

from utils_lib.utils import load_img
from utils_lib.utils_incremental import get_old_new_classes


# Global variables to be initialized from config
DEG_MAP = None
OLD_CLASSES = None
NEW_CLASSES = None
ORIG2CLASSIFIER = None  # Mapping from original DEG_MAP indices to classifier output indices


def init_from_config(config):
    """
    Initialize global DEG_MAP, OLD_CLASSES, NEW_CLASSES, ORIG2CLASSIFIER from config.
    """
    global DEG_MAP, OLD_CLASSES, NEW_CLASSES, ORIG2CLASSIFIER

    DEG_MAP = config["deg_map"]
    old_classes, new_classes, _, _ = get_old_new_classes(config)
    OLD_CLASSES = old_classes
    NEW_CLASSES = new_classes

    if "cgcd" in config and "class_mappings" in config["cgcd"]:
        class_mappings_path = config["cgcd"]["class_mappings"]
        if os.path.exists(class_mappings_path):
            with open(class_mappings_path, "r", encoding="utf-8") as f:
                class_mappings = json.load(f)
            ORIG2CLASSIFIER = {int(k): int(v) for k, v in class_mappings["orig2classifier"].items()}
            print(f"\n[WRES Dataset Config] Loaded ORIG2CLASSIFIER mapping from {class_mappings_path}:")
            print(f"  {ORIG2CLASSIFIER}")
        else:
            print(f"\n[WRES Dataset Config] Warning: class_mappings file not found at {class_mappings_path}")
            ORIG2CLASSIFIER = {i: i for i in range(len(DEG_MAP))}
            print("  Using identity mapping")
    else:
        ORIG2CLASSIFIER = {i: i for i in range(len(DEG_MAP))}
        print("\n[WRES Dataset Config] No class_mappings found, using identity mapping")

    print(f"\n[WRES Dataset Config] Initialized from config:")
    print(f"  DEG_MAP: {len(DEG_MAP)} classes")
    print(f"  OLD_CLASSES ({len(OLD_CLASSES)}): {OLD_CLASSES}")
    print(f"  NEW_CLASSES ({len(NEW_CLASSES)}): {NEW_CLASSES}\n")


def crop_img(image, base=16):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2 : h - crop_h + crop_h // 2, crop_w // 2 : w - crop_w + crop_w // 2, :]


class RefDegImage(Dataset):
    """
    Validation/Test dataset of paired GT(LQ target) and input(LQ source).
    """

    def __init__(self, hq_img_paths, lq_img_paths, val=False, name="test", deg_name="unknown", deg_class=0):
        assert len(hq_img_paths) == len(lq_img_paths)
        self.hq_paths = hq_img_paths
        self.lq_paths = lq_img_paths
        self.totensor = torchvision.transforms.ToTensor()
        self.val = val
        self.name = name
        self.degradation = deg_name
        self.deg_class = deg_class

    def __len__(self):
        return len(self.hq_paths)

    def __getitem__(self, idx):
        hq_path = self.hq_paths[idx]
        lq_path = self.lq_paths[idx]

        hq_image = load_img(hq_path)
        lq_image = load_img(lq_path)

        if self.val:
            hq_image = crop_img(hq_image)
            lq_image = crop_img(lq_image)

        hq_image = self.totensor(hq_image.astype(np.float32))
        lq_image = self.totensor(lq_image.astype(np.float32))

        return hq_image, lq_image, hq_path


class TrainLabeled(Dataset):
    def __init__(self, hq_img_paths, lq_img_paths, deg_name, deg_class, patch_size=224):
        assert len(hq_img_paths) == len(lq_img_paths)
        self.hq_paths = hq_img_paths
        self.lq_paths = lq_img_paths
        self.deg_name = deg_name
        self.deg_class = deg_class
        self.patch_size = patch_size

    def __len__(self):
        return len(self.hq_paths)

    def __getitem__(self, idx):
        hq_image = cv2.imread(self.hq_paths[idx], cv2.IMREAD_COLOR)
        lq_image = cv2.imread(self.lq_paths[idx], cv2.IMREAD_COLOR)
        if hq_image is None:
            raise ValueError(f"Failed to load GT image: {self.hq_paths[idx]}")
        if lq_image is None:
            raise ValueError(f"Failed to load input image: {self.lq_paths[idx]}")

        pair = torch.stack(
            [
                tvF.to_image(cv2.cvtColor(hq_image, cv2.COLOR_BGR2RGB)),
                tvF.to_image(cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB)),
            ]
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


def _collect_images(path):
    return sorted(glob(os.path.join(path, "*")))


def _build_unique_key_map(paths, key_fn):
    key_to_path = {}
    duplicated = set()
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        key = key_fn(stem)
        if key in key_to_path:
            duplicated.add(key)
        else:
            key_to_path[key] = p
    for k in duplicated:
        key_to_path.pop(k, None)
    return key_to_path


def _match_gt_path(input_path, gt_by_stem, gt_by_first_token, gt_by_leading_num):
    input_stem = os.path.splitext(os.path.basename(input_path))[0]

    # 1) Exact stem match
    if input_stem in gt_by_stem:
        return gt_by_stem[input_stem]

    # 2) Prefix-token match (e.g., im_0001_s100_a04 -> im_0001)
    tokens = input_stem.split("_")
    for k in range(len(tokens) - 1, 0, -1):
        candidate = "_".join(tokens[:k])
        if candidate in gt_by_stem:
            return gt_by_stem[candidate]

    # 3) First token unique match (e.g., 0_rain -> 0_clean)
    first_token = tokens[0]
    if first_token in gt_by_first_token:
        return gt_by_first_token[first_token]

    # 4) Leading numeric unique match
    m = re.match(r"^(\d+)", input_stem)
    if m:
        leading_num = m.group(1)
        if leading_num in gt_by_leading_num:
            return gt_by_leading_num[leading_num]

    return None


def _build_pairs(hq_path, lq_path, deg_name):
    gt_img_paths = _collect_images(hq_path)
    input_img_paths = _collect_images(lq_path)

    gt_by_stem = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in gt_img_paths
    }
    gt_by_first_token = _build_unique_key_map(gt_img_paths, lambda s: s.split("_")[0])
    gt_by_leading_num = _build_unique_key_map(
        gt_img_paths,
        lambda s: re.match(r"^(\d+)", s).group(1) if re.match(r"^(\d+)", s) else None,
    )
    gt_by_leading_num.pop(None, None)

    paired_hq = []
    paired_lq = []
    unmatched = []

    for in_path in input_img_paths:
        gt_path = _match_gt_path(in_path, gt_by_stem, gt_by_first_token, gt_by_leading_num)
        if gt_path is None:
            unmatched.append(in_path)
            continue
        paired_hq.append(gt_path)
        paired_lq.append(in_path)

    if unmatched:
        preview = ", ".join(os.path.basename(x) for x in unmatched[:5])
        print(
            f"[WRES] Warning: {deg_name} has {len(unmatched)} unmatched inputs "
            f"(first 5: {preview})"
        )

    if len(paired_hq) == 0:
        raise ValueError(f"[WRES] No valid GT-input pairs found for class {deg_name}")

    return paired_hq, paired_lq


def create_train_datasets(train_data_list, patch_size=224):
    datasets = []
    for data_info in train_data_list:
        hq_path, lq_path, deg_name = data_info

        if NEW_CLASSES is not None and deg_name in NEW_CLASSES:
            print(f"Skipping NEW class for baseline training: {deg_name}")
            continue

        hq_img_paths, lq_img_paths = _build_pairs(hq_path, lq_path, deg_name)

        if len(hq_img_paths) == 0:
            print(f"[WRES] Skip empty class {deg_name}: gt={hq_path}, input={lq_path}")
            continue

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]
        dataset = TrainLabeled(hq_img_paths, lq_img_paths, deg_name, deg_class, patch_size)
        datasets.append(dataset)
        print(f"Loaded {len(dataset)} images for OLD class: {deg_name}")

    if len(datasets) > 0:
        combined = ConcatDataset(datasets)
        print(f"Total training samples (OLD classes only): {len(combined)}")
        return combined
    raise ValueError("No training datasets found after filtering!")


def create_val_datasets(val_data_list):
    datasets = []
    for data_info in val_data_list:
        hq_path, lq_path, deg_name = data_info
        if NEW_CLASSES is not None and deg_name in NEW_CLASSES:
            continue

        hq_img_paths, lq_img_paths = _build_pairs(hq_path, lq_path, deg_name)
        if len(hq_img_paths) == 0:
            continue

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]
        val_dataset = RefDegImage(hq_img_paths, lq_img_paths, val=True, name=deg_name, deg_class=deg_class)
        datasets.append((val_dataset, deg_name))
        print(f"Loaded validation for OLD class: {deg_name}")

    return datasets


def create_stage0_val_datasets(val_data_list):
    """
    Stage 0 validation set creation.
    Keeps OLD/NEW labeling identical to datasets_my behavior.
    """
    datasets = []

    if NEW_CLASSES is None or OLD_CLASSES is None:
        print("[Warning] NEW_CLASSES/OLD_CLASSES not initialized, treating all as OLD")
        return create_val_datasets(val_data_list)

    for data_info in val_data_list:
        hq_path, lq_path, deg_name = data_info

        hq_img_paths, lq_img_paths = _build_pairs(hq_path, lq_path, deg_name)
        if len(hq_img_paths) == 0:
            continue

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]
        val_dataset = RefDegImage(hq_img_paths, lq_img_paths, val=True, name=deg_name, deg_class=deg_class)
        datasets.append((val_dataset, deg_name))

        label = "NEW (zero-shot)" if deg_name in NEW_CLASSES else "OLD"
        print(f"Loaded validation for {label} class: {deg_name}")

    return datasets
