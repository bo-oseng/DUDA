import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import datasets_wres_continual as base_data


STRONG_AUG_VARIANT = "default"


def init_from_config(config):
    global STRONG_AUG_VARIANT
    base_data.init_from_config(config)
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    STRONG_AUG_VARIANT = str(train_cfg.get("semiuir_strong_aug_variant", "default"))
    print(f"[WRES-SemiUIR] strong aug variant: {STRONG_AUG_VARIANT}")
    return None


def _pil_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    if hasattr(Image, "LANCZOS"):
        return Image.LANCZOS
    return Image.ANTIALIAS


class SemiUIRStrongAug:
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.25,
        )
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


def build_semiuir_strong_aug(variant=None):
    chosen = str(variant or STRONG_AUG_VARIANT or "default").strip().lower()
    if chosen != "default":
        raise ValueError(f"Unsupported augmentation: {chosen}; use default.")
    return SemiUIRStrongAug()


class TrainUnlabeledSemiUIR(Dataset):
    """
    Unlabeled image dataset for Semi-UIR style training.

    Returns:
        weak_tensor: resized original input
        strong_tensor: strongly augmented view
        pseudo_tensor: current pseudo image (or zeros if missing)
        pseudo_path: pseudo-bank path to read/write
    """

    def __init__(self, lq_img_paths, pseudo_img_paths, deg_name, deg_class, fine_size=224):
        assert len(lq_img_paths) == len(pseudo_img_paths)
        self.lq_paths = lq_img_paths
        self.pseudo_paths = pseudo_img_paths
        self.degradation = deg_name
        self.deg_class = deg_class
        self.fine_size = int(fine_size)
        self.resample = _pil_resample()
        self.to_tensor = ToTensor()
        self.strong_aug = build_semiuir_strong_aug()

    def __len__(self):
        return len(self.lq_paths)

    def _load_and_resize_rgb(self, path):
        image = Image.open(path).convert("RGB")
        return image.resize((self.fine_size, self.fine_size), self.resample)

    def __getitem__(self, idx):
        lq_path = self.lq_paths[idx]
        pseudo_path = self.pseudo_paths[idx]

        weak_image = self._load_and_resize_rgb(lq_path)
        strong_image = self.strong_aug(weak_image.copy())

        if os.path.exists(pseudo_path):
            try:
                pseudo_image = self._load_and_resize_rgb(pseudo_path)
            except (OSError, ValueError):
                pseudo_image = Image.fromarray(np.zeros((self.fine_size, self.fine_size, 3), dtype=np.uint8))
        else:
            pseudo_image = Image.fromarray(np.zeros((self.fine_size, self.fine_size, 3), dtype=np.uint8))

        weak_tensor = self.to_tensor(weak_image)
        strong_tensor = self.to_tensor(strong_image)
        pseudo_tensor = self.to_tensor(pseudo_image)
        return weak_tensor, strong_tensor, pseudo_tensor, pseudo_path


class TrainUnlabeledSemiUIRNoAug(Dataset):
    """
    Unlabeled image dataset for Semi-UIR style training.

    Returns:
        weak_tensor: resized original input
        strong_tensor: strongly augmented view
        pseudo_tensor: current pseudo image (or zeros if missing)
        pseudo_path: pseudo-bank path to read/write
    """

    def __init__(self, lq_img_paths, pseudo_img_paths, deg_name, deg_class, fine_size=224):
        assert len(lq_img_paths) == len(pseudo_img_paths)
        self.lq_paths = lq_img_paths
        self.pseudo_paths = pseudo_img_paths
        self.degradation = deg_name
        self.deg_class = deg_class
        self.fine_size = int(fine_size)
        self.resample = _pil_resample()
        self.to_tensor = ToTensor()
        self.strong_aug = SemiUIRStrongAug()

    def __len__(self):
        return len(self.lq_paths)

    def _load_and_resize_rgb(self, path):
        image = Image.open(path).convert("RGB")
        return image.resize((self.fine_size, self.fine_size), self.resample)

    def __getitem__(self, idx):
        lq_path = self.lq_paths[idx]
        pseudo_path = self.pseudo_paths[idx]

        weak_image = self._load_and_resize_rgb(lq_path)
        strong_image = weak_image.copy()

        if os.path.exists(pseudo_path):
            try:
                pseudo_image = self._load_and_resize_rgb(pseudo_path)
            except (OSError, ValueError):
                pseudo_image = Image.fromarray(np.zeros((self.fine_size, self.fine_size, 3), dtype=np.uint8))
        else:
            pseudo_image = Image.fromarray(np.zeros((self.fine_size, self.fine_size, 3), dtype=np.uint8))

        weak_tensor = self.to_tensor(weak_image)
        strong_tensor = self.to_tensor(strong_image)
        pseudo_tensor = self.to_tensor(pseudo_image)
        return weak_tensor, strong_tensor, pseudo_tensor, pseudo_path


def create_stage1_unlabeled_dataset(
    real_train_root,
    pseudo_patches_dir,
    fine_size=224,
):
    """
    Create image-based unlabeled dataset for continual stages.

    - Source images: real_train_root/<NEW_CLASS>/*
    - Pseudo labels: pseudo_patches_dir/<NEW_CLASS>/*
    """
    if base_data.NEW_CLASSES is None:
        raise ValueError("NEW_CLASSES not initialized. Call init_from_config() first!")

    datasets = []
    for deg_name in base_data.NEW_CLASSES:
        src_class_dir = os.path.join(real_train_root, deg_name)
        pseudo_class_dir = os.path.join(pseudo_patches_dir, deg_name)

        if not os.path.isdir(src_class_dir):
            print(f"[WRES-SemiUIR] Warning: NEW class source dir not found: {src_class_dir}")
            continue

        lq_img_paths = base_data._collect_images(src_class_dir)
        if len(lq_img_paths) == 0:
            print(f"[WRES-SemiUIR] Warning: no unlabeled images found for NEW class: {deg_name}")
            continue

        os.makedirs(pseudo_class_dir, exist_ok=True)
        pseudo_img_paths = [os.path.join(pseudo_class_dir, os.path.basename(path)) for path in lq_img_paths]

        deg_class_orig = base_data.DEG_MAP[deg_name]
        deg_class = base_data.ORIG2CLASSIFIER[deg_class_orig]
        dataset = TrainUnlabeledSemiUIR(
            lq_img_paths=lq_img_paths,
            pseudo_img_paths=pseudo_img_paths,
            deg_name=deg_name,
            deg_class=deg_class,
            fine_size=fine_size,
        )
        datasets.append(dataset)
        print(
            f"[WRES-SemiUIR] Loaded {len(dataset)} UNLABELED images for NEW class: {deg_name}\n"
            f"  Images: {src_class_dir}\n"
            f"  Pseudo bank: {pseudo_class_dir}"
        )

    if len(datasets) == 0:
        raise ValueError("No stage unlabeled image datasets found for NEW classes.")

    combined = ConcatDataset(datasets)
    print(f"[WRES-SemiUIR] Total stage UNLABELED images: {len(combined)}")
    return combined
