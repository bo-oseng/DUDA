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
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as tvF

from utils_lib.utils import load_img
from utils_lib.utils_incremental import get_old_new_classes


# Global variables to be initialized from config
DEG_MAP = None
OLD_CLASSES = None
NEW_CLASSES = None
ORIG2CLASSIFIER = None  # Mapping from original DEG_MAP indices to classifier output indices

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


class TrainUnlabeled(Dataset):
    """
    Unlabeled patch dataset with pseudo-label file paths.
    """

    def __init__(self, lq_patch_paths, pseudo_patch_paths, deg_name, deg_class):
        assert len(lq_patch_paths) == len(pseudo_patch_paths)
        self.lq_paths = lq_patch_paths
        self.pseudo_paths = pseudo_patch_paths
        self.degradation = deg_name
        self.deg_class = deg_class

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq_path = self.lq_paths[idx]
        pseudo_path = self.pseudo_paths[idx]

        lq_image = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        if lq_image is None:
            raise ValueError(f"Failed to load LQ patch: {lq_path}")
        lq_image = cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if os.path.exists(pseudo_path):
            pseudo_image = cv2.imread(pseudo_path, cv2.IMREAD_COLOR)
            if pseudo_image is None:
                pseudo_image = np.zeros_like(lq_image)
            else:
                pseudo_image = cv2.cvtColor(pseudo_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            pseudo_image = np.zeros_like(lq_image)

        lq_tensor = torch.from_numpy(lq_image.transpose(2, 0, 1))
        pseudo_tensor = torch.from_numpy(pseudo_image.transpose(2, 0, 1))

        return lq_tensor, pseudo_tensor, pseudo_path


class RealEvalImage(Dataset):
    """
    Real-world evaluation dataset (unpaired): returns only LQ image tensor and path.
    """

    def __init__(self, lq_img_paths, name="real_eval"):
        self.lq_paths = lq_img_paths
        self.name = name
        self.totensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq_path = self.lq_paths[idx]
        lq_image = load_img(lq_path)
        lq_image = crop_img(lq_image)
        lq_tensor = self.totensor(lq_image.astype(np.float32))
        return lq_tensor, lq_path


def _collect_images(path):
    candidates = sorted(glob(os.path.join(path, "*")))
    out = []
    for p in candidates:
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in IMG_EXTENSIONS:
            out.append(p)
    return out


def _ensure_min_size(img, patch_size):
    h, w = img.shape[:2]
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return img
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)


def _patch_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    end_pos = length - patch_size
    if positions[-1] != end_pos:
        positions.append(end_pos)
    return positions


def _extract_patches_with_full_coverage(image, patch_size, stride):
    image = _ensure_min_size(image, patch_size)
    h, w = image.shape[:2]
    ys = _patch_positions(h, patch_size, stride)
    xs = _patch_positions(w, patch_size, stride)

    patches = []
    for y in ys:
        for x in xs:
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append((patch, y, x))
    return patches


def _build_or_update_patch_cache_for_class(
    src_class_dir,
    patch_class_dir,
    patch_size,
    stride,
    force_patchify=False,
    max_patches_per_image=0,
    seed=42,
    show_progress=False,
    progress_desc=None,
):
    os.makedirs(patch_class_dir, exist_ok=True)
    src_images = _collect_images(src_class_dir)

    generated = 0
    reused_existing_images = 0
    reused_existing_patches = 0
    removed_old_patches = 0
    replaced_mismatch_patches = 0

    pbar = None
    if show_progress:
        desc = progress_desc or f"[Patchify] {os.path.basename(src_class_dir)}"
        pbar = tqdm(total=len(src_images), desc=desc, dynamic_ncols=True, leave=False)

    for img_idx, src_path in enumerate(src_images):
        stem = os.path.splitext(os.path.basename(src_path))[0]
        prefix = f"{stem}__id{img_idx:06d}"
        existing = glob(os.path.join(patch_class_dir, f"{prefix}__y*_x*.png"))

        if force_patchify and len(existing) > 0:
            for old_path in existing:
                try:
                    os.remove(old_path)
                    removed_old_patches += 1
                except OSError:
                    pass
            existing = []

        existing_names = {os.path.basename(p) for p in existing}
        if len(existing_names) > 0:
            reused_existing_images += 1
            reused_existing_patches += len(existing_names)

        image = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        patches = _extract_patches_with_full_coverage(image, patch_size=patch_size, stride=stride)
        if max_patches_per_image and max_patches_per_image > 0 and len(patches) > max_patches_per_image:
            rng = random.Random(seed + img_idx)
            sampled_indices = sorted(rng.sample(range(len(patches)), k=max_patches_per_image))
            patches = [patches[i] for i in sampled_indices]

        for patch, y, x in patches:
            patch_name = f"{prefix}__y{y:05d}_x{x:05d}.png"
            out_path = os.path.join(patch_class_dir, patch_name)
            if patch_name in existing_names or os.path.exists(out_path):
                old_patch = cv2.imread(out_path, cv2.IMREAD_COLOR)
                if old_patch is not None and old_patch.shape[0] == patch_size and old_patch.shape[1] == patch_size:
                    continue
                replaced_mismatch_patches += 1
            cv2.imwrite(out_path, patch)
            generated += 1

        if pbar is not None:
            pbar.update(1)
            if (img_idx + 1) % 50 == 0 or (img_idx + 1) == len(src_images):
                pbar.set_postfix(
                    {
                        "gen": generated,
                        "reused_img": reused_existing_images,
                        "repl": replaced_mismatch_patches,
                    }
                )

    if pbar is not None:
        pbar.close()

    return {
        "source_images": len(src_images),
        "generated_patches": generated,
        "reused_existing_images": reused_existing_images,
        "reused_existing_patches": reused_existing_patches,
        "removed_old_patches": removed_old_patches,
        "replaced_mismatch_patches": replaced_mismatch_patches,
    }


def _filter_invalid_patch_files(patch_paths, patch_size):
    valid_paths = []
    removed_cnt = 0
    for p in patch_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] != patch_size or img.shape[1] != patch_size:
            try:
                os.remove(p)
            except OSError:
                pass
            removed_cnt += 1
            continue
        valid_paths.append(p)
    return valid_paths, removed_cnt


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


def create_labeled_dataset(train_data_list, patch_size=256):
    return create_train_datasets(train_data_list, patch_size=patch_size)


def create_stage1_unlabeled_dataset(
    real_train_root,
    lq_patches_dir,
    pseudo_patches_dir,
    patch_size=224,
    patch_stride=None,
    force_patchify=False,
    max_patches_per_image=0,
    seed=42,
    build_cache=True,
    clean_invalid_patches=True,
    show_progress=False,
    skip_cache_if_exists=True,
):
    """
    Create unlabeled dataset for continual stages.
    - Source images: real_train_root/<NEW_CLASS>/*
    - Patch cache: lq_patches_dir/<NEW_CLASS>/*.png (incrementally generated)
    - Pseudo labels: pseudo_patches_dir/<NEW_CLASS>/*.png
    """
    if NEW_CLASSES is None:
        raise ValueError("NEW_CLASSES not initialized. Call init_from_config() first!")

    if patch_stride is None:
        patch_stride = max(1, patch_size // 2)

    datasets = []
    if build_cache:
        print(
            f"[WRES] Building unlabeled patch cache: patch_size={patch_size}, stride={patch_stride}, "
            f"force_patchify={force_patchify}, max_patches_per_image={max_patches_per_image}"
        )
    else:
        print("[WRES] Reusing existing unlabeled patch cache (build_cache=False)")

    for deg_name in NEW_CLASSES:
        src_class_dir = os.path.join(real_train_root, deg_name)
        patch_class_dir = os.path.join(lq_patches_dir, deg_name)
        pseudo_class_dir = os.path.join(pseudo_patches_dir, deg_name)

        if not os.path.isdir(src_class_dir):
            print(f"[WRES] Warning: NEW class source dir not found: {src_class_dir}")
            continue

        lq_patch_paths = []
        if build_cache and (not force_patchify) and skip_cache_if_exists:
            existing_class_patches = sorted(glob(os.path.join(patch_class_dir, "*.png")))
            if len(existing_class_patches) > 0:
                lq_patch_paths = existing_class_patches
                print(
                    f"[WRES] {deg_name}: existing patch cache found ({len(lq_patch_paths)} patches), "
                    f"skip patchify (force_patchify=False)"
                )

        if len(lq_patch_paths) == 0:
            if build_cache:
                stats = _build_or_update_patch_cache_for_class(
                    src_class_dir=src_class_dir,
                    patch_class_dir=patch_class_dir,
                    patch_size=patch_size,
                    stride=patch_stride,
                    force_patchify=force_patchify,
                    max_patches_per_image=max_patches_per_image,
                    seed=seed,
                    show_progress=show_progress,
                    progress_desc=f"[Patchify:{deg_name}]",
                )
                print(
                    f"[WRES] {deg_name}: source={stats['source_images']}, generated={stats['generated_patches']}, "
                    f"reused_images={stats['reused_existing_images']}, "
                    f"reused_patches={stats['reused_existing_patches']}, "
                    f"replaced_mismatch={stats['replaced_mismatch_patches']}, "
                    f"removed={stats['removed_old_patches']}"
                )
            lq_patch_paths = sorted(glob(os.path.join(patch_class_dir, "*.png")))

        if clean_invalid_patches and len(lq_patch_paths) > 0:
            lq_patch_paths, removed_invalid = _filter_invalid_patch_files(lq_patch_paths, patch_size=patch_size)
            if removed_invalid > 0:
                print(f"[WRES] {deg_name}: removed {removed_invalid} invalid/corrupted patches")
                if build_cache:
                    refill_stats = _build_or_update_patch_cache_for_class(
                        src_class_dir=src_class_dir,
                        patch_class_dir=patch_class_dir,
                        patch_size=patch_size,
                        stride=patch_stride,
                        force_patchify=False,
                        max_patches_per_image=max_patches_per_image,
                        seed=seed,
                        show_progress=show_progress,
                        progress_desc=f"[Patchify-refill:{deg_name}]",
                    )
                    lq_patch_paths = sorted(glob(os.path.join(patch_class_dir, "*.png")))
                    lq_patch_paths, removed_invalid2 = _filter_invalid_patch_files(
                        lq_patch_paths, patch_size=patch_size
                    )
                    if removed_invalid2 > 0:
                        print(
                            f"[WRES] {deg_name}: warning - {removed_invalid2} invalid patches still remained after refill"
                        )
                    print(
                        f"[WRES] {deg_name}: refill generated={refill_stats['generated_patches']}, "
                        f"reused={refill_stats['reused_existing_patches']}"
                    )

        if len(lq_patch_paths) == 0:
            print(f"[WRES] Warning: no patches found for NEW class: {deg_name} ({patch_class_dir})")
            continue

        os.makedirs(pseudo_class_dir, exist_ok=True)
        pseudo_patch_paths = [os.path.join(pseudo_class_dir, os.path.basename(p)) for p in lq_patch_paths]

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]
        dataset = TrainUnlabeled(lq_patch_paths, pseudo_patch_paths, deg_name, deg_class)
        datasets.append(dataset)
        print(
            f"[WRES] Loaded {len(dataset)} UNLABELED patches for NEW class: {deg_name}\n"
            f"  LQ patches: {patch_class_dir}\n"
            f"  Pseudo patches: {pseudo_class_dir}"
        )

    if len(datasets) == 0:
        raise ValueError("No stage unlabeled datasets found for NEW classes.")

    combined = ConcatDataset(datasets)
    print(f"[WRES] Total stage UNLABELED patches: {len(combined)}")
    return combined


def create_stage1_val_datasets(val_data_list):
    """
    Optional paired validation for OLD+NEW classes.
    """
    datasets = []
    if NEW_CLASSES is None or OLD_CLASSES is None:
        raise ValueError("NEW_CLASSES/OLD_CLASSES not initialized. Call init_from_config() first!")

    valid_classes = set(OLD_CLASSES + NEW_CLASSES)
    for data_info in val_data_list:
        hq_path, lq_path, deg_name = data_info
        if deg_name not in valid_classes:
            continue

        try:
            hq_img_paths, lq_img_paths = _build_pairs(hq_path, lq_path, deg_name)
        except Exception as e:
            print(f"[WRES] Skip val class {deg_name}: {e}")
            continue

        if len(hq_img_paths) == 0:
            continue

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]
        val_dataset = RefDegImage(hq_img_paths, lq_img_paths, val=True, name=deg_name, deg_class=deg_class)
        datasets.append((val_dataset, deg_name))
        label = "NEW" if deg_name in NEW_CLASSES else "OLD"
        print(f"Loaded validation for {label} class: {deg_name}")

    return datasets


def create_real_eval_datasets(real_eval_root, eval_set_names):
    """
    Unpaired real-world evaluation datasets.
    """
    datasets = []
    for set_name in eval_set_names:
        set_dir = os.path.join(real_eval_root, set_name)
        if not os.path.isdir(set_dir):
            print(f"[WRES] Warning: real eval set not found: {set_dir}")
            continue
        lq_img_paths = _collect_images(set_dir)
        if len(lq_img_paths) == 0:
            print(f"[WRES] Warning: no images found in eval set: {set_dir}")
            continue
        datasets.append((RealEvalImage(lq_img_paths, name=set_name), set_name))
        print(f"[WRES] Loaded real eval set: {set_name} ({len(lq_img_paths)} images)")
    return datasets
