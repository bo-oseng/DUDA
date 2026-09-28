import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import torch
import random
import cv2
import json
import os
from glob import glob
import tqdm
from utils_lib.utils import load_img, modcrop
from utils_lib.utils_incremental import get_old_new_classes
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


# Global variables to be initialized from config
DEG_MAP = None
OLD_CLASSES = None
NEW_CLASSES = None
ORIG2CLASSIFIER = None  # Mapping from original DEG_MAP indices to classifier output indices


def init_from_config(config):
    """
    Initialize global DEG_MAP, OLD_CLASSES, NEW_CLASSES, ORIG2CLASSIFIER from config.
    Should be called once at the beginning of training/evaluation.
    """
    global DEG_MAP, OLD_CLASSES, NEW_CLASSES, ORIG2CLASSIFIER

    DEG_MAP = config["deg_map"]
    old_classes, new_classes, _, _ = get_old_new_classes(config)
    OLD_CLASSES = old_classes
    NEW_CLASSES = new_classes

    # Load orig2classifier mapping from class_mappings.json if available
    if "cgcd" in config and "class_mappings" in config["cgcd"]:
        class_mappings_path = config["cgcd"]["class_mappings"]
        if os.path.exists(class_mappings_path):
            with open(class_mappings_path, "r") as f:
                class_mappings = json.load(f)
            # Convert string keys to integers
            ORIG2CLASSIFIER = {int(k): int(v) for k, v in class_mappings["orig2classifier"].items()}
            print(f"\n[Dataset Config] Loaded ORIG2CLASSIFIER mapping from {class_mappings_path}:")
            print(f"  {ORIG2CLASSIFIER}")
        else:
            print(f"\n[Dataset Config] Warning: class_mappings file not found at {class_mappings_path}")
            ORIG2CLASSIFIER = {i: i for i in range(len(DEG_MAP))}
            print(f"  Using identity mapping")
    else:
        # If no class_mappings, assume identity mapping
        ORIG2CLASSIFIER = {i: i for i in range(len(DEG_MAP))}
        print(f"\n[Dataset Config] No class_mappings found, using identity mapping")

    print(f"\n[Dataset Config] Initialized from config:")
    print(f"  DEG_MAP: {len(DEG_MAP)} classes")
    print(f"  OLD_CLASSES ({len(OLD_CLASSES)}): {OLD_CLASSES}")
    print(f"  NEW_CLASSES ({len(NEW_CLASSES)}): {NEW_CLASSES}\n")


def rotate(img, rotate_index):
    if rotate_index == 0:
        return img
    if rotate_index == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate_index == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotate_index == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate_index == 4:
        return cv2.flip(img, 0)  # 상하 반전
    if rotate_index == 5:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 0)
    if rotate_index == 6:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_180), 0)
    if rotate_index == 7:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 0)
    return img


def get_deg_name(path):
    """
    Get the degradation name from the path.
    Priority order: check compound degradations first, then single ones.
    """
    # Convert to lowercase for case-insensitive matching
    path_lower = path.lower()

    # Check compound degradations first (order matters!)
    if "low_haze_snow" in path_lower:
        return "low_haze_snow"
    elif "low_haze_rain" in path_lower:
        return "low_haze_rain"
    elif "haze_snow" in path_lower:
        return "haze_snow"
    elif "haze_rain" in path_lower:
        return "haze_rain"
    elif "low_snow" in path_lower:
        return "low_snow"
    elif "low_rain" in path_lower:
        return "low_rain"
    elif "low_haze" in path_lower:
        return "low_haze"
    # Single degradations
    elif "haze" in path_lower or "sots" in path_lower or "reside" in path_lower:
        return "haze"
    elif "rain" in path_lower:
        return "rain"
    elif "snow" in path_lower:
        return "snow"
    elif "low" in path_lower or "lol" in path_lower:
        return "low"
    else:
        # Default fallback
        return "haze"


def crop_img(image, base=16):
    """
    Mod crop the image to ensure the dimension is divisible by base. Also done by SwinIR, Restormer and others.
    """
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2 : h - crop_h + crop_h // 2, crop_w // 2 : w - crop_w + crop_w // 2, :]


################# DATASETS


class RefDegImage(Dataset):
    """
    Dataset for Image Restoration having low-quality image and the reference image.
    Tasks: synthetic denoising, deblurring, super-res, etc.
    """

    def __init__(
        self, hq_img_paths, lq_img_paths, augmentations=None, val=False, name="test", deg_name="noise", deg_class=0
    ):

        assert len(hq_img_paths) == len(lq_img_paths)

        self.hq_paths = hq_img_paths
        self.lq_paths = lq_img_paths
        self.totensor = torchvision.transforms.ToTensor()
        self.val = val
        self.augs = augmentations
        self.name = name
        self.degradation = deg_name
        self.deg_class = deg_class

        if self.val:
            self.augs = None  # No augmentations during validation/test

    def __len__(self):
        return len(self.hq_paths)

    def __getitem__(self, idx):
        hq_path = self.hq_paths[idx]
        lq_path = self.lq_paths[idx]

        hq_image = load_img(hq_path)
        lq_image = load_img(lq_path)

        if self.val:
            # if an image has an odd number dimension we trim for example from [321, 189] to [320, 188].
            hq_image = crop_img(hq_image)
            lq_image = crop_img(lq_image)

        hq_image = self.totensor(hq_image.astype(np.float32))
        lq_image = self.totensor(lq_image.astype(np.float32))

        return hq_image, lq_image, hq_path


def create_testsets(testsets, debug=False):
    """
    Given a list of testsets create pytorch datasets for each.
    The method requires the paths to references and noisy images.
    """
    assert len(testsets) > 0

    if debug:
        print(20 * "****")
        print("Creating Testsets", len(testsets))

    datasets = []
    for testdt in testsets:

        path_hq, path_lq = testdt[0], testdt[1]
        if debug:
            print(path_hq, path_lq)

        if ("denoising" in path_hq) or ("jpeg" in path_hq):
            dataset_name = path_hq.split("/")[-1]
            dataset_sigma = path_lq.split("/")[-1].split("_")[-1].split(".")[0]
            dataset_name = dataset_name + f"_{dataset_sigma}"
        elif "Rain" in path_hq:
            if "Rain100L" in path_hq:
                dataset_name = "Rain100L"
            else:
                dataset_name = path_hq.split("/")[3]

        elif ("gopro" in path_hq) or ("GoPro" in path_hq):
            dataset_name = "GoPro"
        elif "LOL" in path_hq:
            dataset_name = "LOL"
        elif "SOTS" in path_hq:
            dataset_name = "SOTS"
        elif "fiveK" in path_hq:
            dataset_name = "MIT5K"
        else:
            assert False, f"{path_hq} - unknown dataset"

        hq_img_paths = sorted(glob(os.path.join(path_hq, "*")))
        lq_img_paths = sorted(glob(os.path.join(path_lq, "*")))

        if "SOTS" in path_hq:
            # Haze removal SOTS test dataset
            dataset_name = "SOTS"
            hq_img_paths = sorted(glob(os.path.join(path_hq, "*.jpg")))
            assert len(hq_img_paths) == 500

            lq_img_paths = [file.replace("GT", "IN") for file in hq_img_paths]

        if "fiveK" in path_hq:
            dataset_name = "MIT5K"
            testf = "test-data/mit5k/test.txt"
            f = open(testf, "r")
            test_ids = f.readlines()
            test_ids = [x.strip() for x in test_ids]
            f.close()
            hq_img_paths = [os.path.join(path_hq, f"{x}.jpg") for x in test_ids]
            lq_img_paths = [x.replace("expertC", "input") for x in hq_img_paths]
            assert len(hq_img_paths) == 498

        if "gopro" in path_hq:
            assert len(hq_img_paths) == 1111

        if "LOL" in path_hq:
            assert len(hq_img_paths) == 15

        assert len(hq_img_paths) == len(lq_img_paths)

        deg_name = get_deg_name(path_hq)
        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]  # Map to classifier output index

        valdts = RefDegImage(
            hq_img_paths=hq_img_paths,
            lq_img_paths=lq_img_paths,
            val=True,
            name=dataset_name,
            deg_name=deg_name,
            deg_class=deg_class,
        )

        datasets.append(valdts)

    assert len(datasets) == len(testsets)
    print(20 * "****")

    return datasets


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


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
        hq_image = cv2.imread(self.hq_paths[idx])
        lq_image = cv2.imread(self.lq_paths[idx])

        # Stack HQ+LQ → [2, C, H, W] uint8
        pair = torch.stack([
            F.to_image(cv2.cvtColor(hq_image, cv2.COLOR_BGR2RGB)),
            F.to_image(cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB)),
        ])

        # Same random crop for both
        i, j, h, w = v2.RandomCrop.get_params(pair[0], (self.patch_size, self.patch_size))
        pair = pair[:, :, i : i + h, j : j + w]

        # Same random augmentation for both (4 rotations × optional vflip)
        rot_k = [0, 3, 2, 1][random.randrange(4)]
        if rot_k > 0:
            pair = torch.rot90(pair, rot_k, dims=[2, 3])
        if random.random() > 0.5:
            pair = pair.flip(2)

        pair = pair.float().div_(255.0)

        return pair[0], pair[1], self.deg_class


class TrainContrastive(Dataset):
    """
    OneRestore style contrastive dataset.
    Directly returns (pos, inp, neg, inp_deg_class) for contrastive training.

    For each sample:
    - Loads GT + all degradation versions from the same image
    - Applies same random crop & augmentation to all
    - Randomly selects one degradation as input (inp)
    - Remaining degradations become negatives (neg)

    Returns:
        pos: [C, H, W] - GT image
        inp: [C, H, W] - randomly selected degraded image
        neg: [N_deg-1, C, H, W] - other degraded images
        inp_deg_class: scalar tensor - classifier index of inp
    """

    def __init__(self, data_root, gt_folder="clear", deg_types=None, patch_size=256):
        self.data_root = data_root
        self.gt_folder = gt_folder
        self.patch_size = patch_size

        # Use OLD_CLASSES if deg_types not specified, exclude clear
        if deg_types is None:
            if OLD_CLASSES is None:
                raise ValueError("OLD_CLASSES not initialized. Call init_from_config() first!")
            self.deg_types = [c for c in OLD_CLASSES if c != gt_folder]
        else:
            self.deg_types = [c for c in deg_types if c != gt_folder]

        # Pre-compute classifier indices for each degradation type
        self.deg_classifier_indices = []
        for deg_type in self.deg_types:
            orig_idx = DEG_MAP[deg_type]
            self.deg_classifier_indices.append(ORIG2CLASSIFIER.get(orig_idx, orig_idx))

        # File list based on GT folder
        gt_dir = os.path.join(data_root, gt_folder)
        if not os.path.exists(gt_dir):
            raise ValueError(f"GT folder not found: {gt_dir}")
        self.file_list = sorted([f for f in os.listdir(gt_dir) if is_image_file(f)])

        # Validate degradation folders exist
        for deg_type in self.deg_types:
            deg_dir = os.path.join(data_root, deg_type)
            if not os.path.exists(deg_dir):
                raise ValueError(f"Degradation folder not found: {deg_dir}")

        print(
            f"[TrainContrastive] {len(self.file_list)} images, " f"{len(self.deg_types)} deg types: {self.deg_types}"
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        n_deg = len(self.deg_types)

        # 1. Load GT + all degradation images → tensor
        gt_path = os.path.join(self.data_root, self.gt_folder, filename)
        gt_img = cv2.imread(gt_path)
        if gt_img is None:
            raise ValueError(f"Failed to load GT image: {gt_path}")

        all_imgs = [cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)]
        for deg_type in self.deg_types:
            deg_path = os.path.join(self.data_root, deg_type, filename)
            deg_img = cv2.imread(deg_path)
            if deg_img is None:
                raise ValueError(f"Failed to load: {deg_path}")
            all_imgs.append(cv2.cvtColor(deg_img, cv2.COLOR_BGR2RGB))

        # 2. Stack to single tensor: [1+N_deg, C, H, W] uint8
        all_tensors = torch.stack([F.to_image(img) for img in all_imgs])

        # 3. Same random crop for all
        i, j, h, w = v2.RandomCrop.get_params(all_tensors[0], (self.patch_size, self.patch_size))
        all_tensors = all_tensors[:, :, i : i + h, j : j + w]

        # 4. Same random augmentation for all (4 rotations × optional vflip = 8 modes)
        rot_k = [0, 3, 2, 1][random.randrange(4)]
        if rot_k > 0:
            all_tensors = torch.rot90(all_tensors, rot_k, dims=[2, 3])
        if random.random() > 0.5:
            all_tensors = all_tensors.flip(2)

        # 5. To float [0, 1] and split
        all_tensors = all_tensors.float().div_(255.0)
        pos = all_tensors[0]
        deg_tensors = all_tensors[1:]

        # 6. Random select one as input, rest as negatives
        inp_idx = random.randint(0, n_deg - 1)
        inp = deg_tensors[inp_idx]
        inp_deg_class = torch.tensor(self.deg_classifier_indices[inp_idx], dtype=torch.long)
        neg = torch.cat([deg_tensors[:inp_idx], deg_tensors[inp_idx + 1 :]], dim=0)

        return pos, inp, neg, inp_deg_class


def create_contrastive_dataset(data_root, patch_size=256, deg_types=None):
    """
    Create a contrastive dataset for OneRestore training.

    Args:
        data_root: Root directory (e.g., data/images/train)
        patch_size: Size of random crop patches
        deg_types: List of degradation types. If None, uses OLD_CLASSES.

    Returns:
        TrainContrastive dataset
    """
    dataset = TrainContrastive(data_root=data_root, gt_folder="clear", deg_types=deg_types, patch_size=patch_size)
    print(f"Created contrastive dataset with {len(dataset)} samples")
    return dataset


class TrainUnlabeled(Dataset):
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

        # Load pre-extracted patches (PNG format, cv2: BGR -> RGB)
        lq_image = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        if lq_image is None:
            raise ValueError(f"Failed to load LQ image: {lq_path}")
        lq_image = cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if os.path.exists(pseudo_path):
            pseudo_image = cv2.imread(pseudo_path, cv2.IMREAD_COLOR)
            if pseudo_image is None:
                # File exists but failed to load - might be corrupted, or race condition
                # Fallback to zeros or raise error? Raising error is safer for debugging but for robust init maybe zero
                print(f"[Warning] Failed to load existing pseudo label: {pseudo_path}, using zeros")
                pseudo_image = np.zeros_like(lq_image)
            else:
                pseudo_image = cv2.cvtColor(pseudo_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            # File does not exist (e.g. during initialization phase)
            # Create dummy pseudo label (zeros)
            pseudo_image = np.zeros_like(lq_image)

        # Convert to tensor [C, H, W] - no augmentation here
        unpaired_data = torch.from_numpy(lq_image.transpose(2, 0, 1))
        pseudo_list = torch.from_numpy(pseudo_image.transpose(2, 0, 1))

        return unpaired_data, pseudo_list, pseudo_path


def create_train_datasets(train_data_list, patch_size=224):
    datasets = []
    for data_info in train_data_list:
        hq_path, lq_path, deg_name = data_info

        # [변경] New 클래스인 경우 학습 데이터셋에서 제외
        if deg_name in NEW_CLASSES:
            print(f"Skipping NEW class for baseline training: {deg_name}")
            continue

        hq_img_paths = sorted(glob(os.path.join(hq_path, "*")))
        lq_img_paths = sorted(glob(os.path.join(lq_path, "*")))

        if len(hq_img_paths) == 0 or len(lq_img_paths) == 0:
            continue
        assert len(hq_img_paths) == len(lq_img_paths)

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]  # Map to classifier output index
        dataset = TrainLabeled(hq_img_paths, lq_img_paths, deg_name, deg_class, patch_size)
        datasets.append(dataset)
        print(f"Loaded {len(dataset)} images for OLD class: {deg_name}")

    if len(datasets) > 0:
        combined = ConcatDataset(datasets)
        print(f"Total training samples (OLD classes only): {len(combined)}")
        return combined
    else:
        raise ValueError("No training datasets found after filtering!")


def create_val_datasets(val_data_list):
    """
    Create validation datasets for training stages (Stage 0 uses create_stage0_val_datasets instead).
    This is kept for backward compatibility with other training scripts.
    """
    datasets = []
    for data_info in val_data_list:
        hq_path, lq_path, deg_name = data_info
        if deg_name in NEW_CLASSES:
            continue

        hq_img_paths = sorted(glob(os.path.join(hq_path, "*")))
        lq_img_paths = sorted(glob(os.path.join(lq_path, "*")))

        if len(hq_img_paths) == 0 or len(lq_img_paths) == 0:
            continue
        assert len(hq_img_paths) == len(lq_img_paths)

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]  # Map to classifier output index
        val_dataset = RefDegImage(hq_img_paths, lq_img_paths, val=True, name=deg_name, deg_class=deg_class)
        datasets.append((val_dataset, deg_name))
        print(f"Loaded validation for OLD class: {deg_name}")

    return datasets


def create_stage0_val_datasets(val_data_list):
    """
    Create validation datasets for Stage 0 baseline training.
    Includes ALL classes:
    - OLD classes: for validation metrics (trained)
    - NEW classes: for zero-shot evaluation (not seen during training)
    """
    datasets = []

    if NEW_CLASSES is None or OLD_CLASSES is None:
        print("[Warning] NEW_CLASSES/OLD_CLASSES not initialized, treating all as OLD")
        # Fallback to original behavior if not initialized
        return create_val_datasets(val_data_list)

    for data_info in val_data_list:
        hq_path, lq_path, deg_name = data_info

        hq_img_paths = sorted(glob(os.path.join(hq_path, "*")))
        lq_img_paths = sorted(glob(os.path.join(lq_path, "*")))

        if len(hq_img_paths) == 0 or len(lq_img_paths) == 0:
            continue
        assert len(hq_img_paths) == len(lq_img_paths)

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]  # Map to classifier output index
        val_dataset = RefDegImage(hq_img_paths, lq_img_paths, val=True, name=deg_name, deg_class=deg_class)
        datasets.append((val_dataset, deg_name))

        # NEW vs OLD label (dynamically determined from config)
        label = "NEW (zero-shot)" if deg_name in NEW_CLASSES else "OLD"
        print(f"Loaded validation for {label} class: {deg_name}")

    return datasets


def create_labeled_dataset(train_data_list, patch_size=256):
    datasets = []
    for data_info in train_data_list:
        hq_path, lq_path, deg_name = data_info

        # [변경] New 클래스인 경우 학습 데이터셋에서 제외
        if deg_name in NEW_CLASSES:
            print(f"Skipping NEW class for baseline training: {deg_name}")
            continue

        hq_img_paths = sorted(glob(os.path.join(hq_path, "*")))
        lq_img_paths = sorted(glob(os.path.join(lq_path, "*")))

        if len(hq_img_paths) == 0 or len(lq_img_paths) == 0:
            continue
        assert len(hq_img_paths) == len(lq_img_paths)

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]  # Map to classifier output index
        dataset = TrainLabeled(hq_img_paths, lq_img_paths, deg_name, deg_class, patch_size)
        datasets.append(dataset)
        print(f"Loaded {len(dataset)} images for OLD class: {deg_name}")

    if len(datasets) > 0:
        combined = ConcatDataset(datasets)
        print(f"Total training samples (OLD classes only): {len(combined)}")
        return combined
    else:
        raise ValueError("No training datasets found after filtering!")


def create_stage1_unlabeled_dataset(lq_patches_dir, pseudo_patches_dir):
    """
    Create unlabeled dataset for Stage 1 incremental learning.
    Uses NEW_CLASSES from config (no hardcoding).
    """
    datasets = []

    # Use NEW_CLASSES from config (dynamically determined based on incremental stage)
    if NEW_CLASSES is None:
        raise ValueError("NEW_CLASSES not initialized. Call init_from_config() first!")

    for deg_name in NEW_CLASSES:

        lq_deg_dir = os.path.join(lq_patches_dir, deg_name)
        if not os.path.exists(lq_deg_dir):
            print(f"Warning: LQ patches directory not found: {lq_deg_dir}")
            continue

        lq_patch_paths = sorted(glob(os.path.join(lq_deg_dir, "*.png")))

        if len(lq_patch_paths) == 0:
            print(f"Warning: No LQ patches found in {lq_deg_dir}")
            continue

        pseudo_deg_dir = os.path.join(pseudo_patches_dir, deg_name)

        # Create directory if it doesn't exist (for initialization)
        if not os.path.exists(pseudo_deg_dir):
            print(f"Creating pseudo label directory: {pseudo_deg_dir}")
            os.makedirs(pseudo_deg_dir, exist_ok=True)

        pseudo_patch_paths = []
        for lq_patch_path in lq_patch_paths:
            basename = os.path.basename(lq_patch_path)
            pseudo_patch_path = os.path.join(pseudo_deg_dir, basename)
            pseudo_patch_paths.append(pseudo_patch_path)

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]  # Map to classifier output index
        dataset = TrainUnlabeled(lq_patch_paths, pseudo_patch_paths, deg_name, deg_class)
        datasets.append(dataset)
        print(f"Loaded {len(dataset)} UNLABELED patches for NEW class: {deg_name}")
        print(f"  LQ patches: {lq_deg_dir}")
        print(f"  Pseudo patches: {pseudo_deg_dir}")

    if len(datasets) > 0:
        combined = ConcatDataset(datasets)
        print(f"Total Stage 1 UNLABELED patches: {len(combined)}")
        return combined
    else:
        raise ValueError("No Stage 1 unlabeled datasets found!")


def create_stage1_val_datasets(val_data_list):
    """
    Create validation datasets for Stage 1 incremental learning.
    Uses NEW_CLASSES from config to determine OLD vs NEW labels.

    IMPORTANT: Only includes OLD + NEW classes (current stage).
    Future stage classes are excluded from validation.
    """
    datasets = []

    if NEW_CLASSES is None or OLD_CLASSES is None:
        raise ValueError("NEW_CLASSES/OLD_CLASSES not initialized. Call init_from_config() first!")

    # Define which classes to validate: OLD + NEW (current stage only)
    valid_classes = set(OLD_CLASSES + NEW_CLASSES)

    for data_info in val_data_list:
        hq_path, lq_path, deg_name = data_info

        # Skip future stage classes (not in OLD or NEW)
        if deg_name not in valid_classes:
            continue

        hq_img_paths = sorted(glob(os.path.join(hq_path, "*")))
        lq_img_paths = sorted(glob(os.path.join(lq_path, "*")))

        if len(hq_img_paths) == 0 or len(lq_img_paths) == 0:
            continue
        assert len(hq_img_paths) == len(lq_img_paths)

        deg_class_orig = DEG_MAP[deg_name]
        deg_class = ORIG2CLASSIFIER[deg_class_orig]  # Map to classifier output index
        val_dataset = RefDegImage(hq_img_paths, lq_img_paths, val=True, name=deg_name, deg_class=deg_class)
        datasets.append((val_dataset, deg_name))

        # NEW vs OLD label (dynamically determined from config)
        label = "NEW" if deg_name in NEW_CLASSES else "OLD"
        print(f"Loaded validation for {label} class: {deg_name}")

    return datasets
