import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sl_finetuned_model import load_finetuned_model_from_checkpoint as load_ckpt_single
from sl_finetuned_model_ddp import load_finetuned_model_from_checkpoint as load_ckpt_ddp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def normalize_relpath(path: str) -> str:
    return path.replace("\\", "/")


def list_rel_images(class_dir: str) -> List[str]:
    rel_paths = []
    for root, _, files in os.walk(class_dir):
        for fname in files:
            full = os.path.join(root, fname)
            if not is_image_file(full):
                continue
            rel_paths.append(normalize_relpath(os.path.relpath(full, class_dir)))
    return sorted(rel_paths)


def read_split_file(path: str) -> List[str]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def write_split_file(path: str, class_name: str, rel_paths: List[str], split: float, seed: int):
    with open(path, "w") as f:
        f.write("# Auto-generated split manifest\n")
        f.write(f"# class={class_name}\n")
        f.write(f"# test_split={split}\n")
        f.write(f"# seed={seed}\n")
        for rel in rel_paths:
            f.write(f"{rel}\n")


def ensure_split_file(
    class_dir: str,
    class_name: str,
    split_filename: str,
    test_split: float,
    seed: int,
    class_offset: int,
    force_resplit: bool,
) -> List[str]:
    rel_paths = list_rel_images(class_dir)
    if not rel_paths:
        raise ValueError(f"No images found in {class_dir}")

    info_path = os.path.join(class_dir, split_filename)
    if os.path.exists(info_path) and not force_resplit:
        existing = read_split_file(info_path)
        missing = [r for r in existing if r not in set(rel_paths)]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Split mismatch in {info_path}: {len(missing)} missing paths "
                f"(e.g., {preview}). Use --force_resplit."
            )
        return sorted(set(existing))

    n = len(rel_paths)
    if n == 1 or test_split == 0.0:
        n_test = 0
    else:
        n_test = int(round(n * test_split))
        n_test = max(1, n_test)
        n_test = min(n_test, n - 1)

    rng = np.random.default_rng(seed + class_offset * 10007 + 43)
    idx = np.arange(n)
    rng.shuffle(idx)
    test_rel = sorted([rel_paths[i] for i in idx[:n_test]])
    write_split_file(info_path, class_name, test_rel, test_split, seed)
    return test_rel


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
        cropped = torchvision.transforms.functional.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [out_height, out_width],
            self.random_crop.interpolation,
        )
        tensor = self.to_tensor(cropped)
        tensor = torchvision.transforms.functional.normalize(tensor, mean=self.mean, std=self.std)
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


def load_manifest_jsonl(path: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_manifest_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")


def build_placeholder_manifest(labels: np.ndarray, split_name: str, label_to_name: Dict[int, str]):
    return [
        {
            "feature_row": int(i),
            "feature_split": split_name,
            "label": int(label),
            "class_name": label_to_name.get(int(label), f"class_{int(label)}"),
            "source": "unknown",
            "metadata_missing": True,
        }
        for i, label in enumerate(labels.tolist())
    ]


def remove_target_manifest_entries(records: List[Dict[str, Any]], target_labels: List[int]):
    target_set = {int(y) for y in target_labels}
    return [record for record in records if int(record.get("label", -1)) not in target_set]


def reindex_manifest(records: List[Dict[str, Any]], start: int = 0):
    reindexed = []
    for offset, record in enumerate(records):
        item = dict(record)
        item["feature_row"] = int(start + offset)
        reindexed.append(item)
    return reindexed


class RealClassSplitDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], transform=None):
        self.samples = samples
        self.targets = [int(sample["label"]) for sample in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        transform_meta = None
        if self.transform is not None:
            image, transform_meta = _unpack_transformed_output(self.transform(image))
        meta = {
            "image_path": os.path.abspath(image_path),
            "rel_path": sample.get("rel_path"),
            "label": int(sample["label"]),
            "class_name": sample.get("class_name"),
            "source": sample.get("source", "real"),
            "feature_split": sample.get("feature_split"),
        }
        if transform_meta is not None:
            meta.update(_to_jsonable(transform_meta))
        return image, int(sample["label"]), meta


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


def infer_to_arrays(model, loader, device):
    model.to(device)
    model.eval()

    feat_chunks = []
    label_chunks = []
    manifest = []
    row_offset = 0
    from tqdm import tqdm

    for batch in tqdm(loader, desc="Extract real features"):
        images = batch["images"].to(device)
        labels = batch["labels"].cpu().numpy()
        with torch.no_grad():
            feat = model(images).pooler_output.cpu().numpy()
        feat_chunks.append(feat)
        label_chunks.append(labels)

        batch_meta = batch.get("meta") or [None] * len(labels)
        for sample_offset, label in enumerate(labels.tolist()):
            record = dict(batch_meta[sample_offset] or {})
            record["feature_row"] = int(row_offset + sample_offset)
            record["label"] = int(label)
            manifest.append(_to_jsonable(record))
        row_offset += len(labels)

    if not feat_chunks:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.concatenate(feat_chunks, axis=0), np.concatenate(label_chunks, axis=0), manifest


def parse_model_arg(pretrained_model_name: str) -> str:
    name = pretrained_model_name
    if name.endswith("-sl"):
        name = name[:-3]
    return name.replace("-", "_")


def load_class_names(class_names_file: str) -> Dict[int, str]:
    out = {}
    if not os.path.exists(class_names_file):
        return out
    with open(class_names_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, cls = line.split(",", 1)
            out[int(idx_str)] = cls
    return out


def save_class_names(class_names_file: str, idx_to_name: Dict[int, str]):
    with open(class_names_file, "w") as f:
        for idx in sorted(idx_to_name):
            f.write(f"{idx},{idx_to_name[idx]}\n")


def load_class_order(path: str) -> List[int]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        raw = f.read().strip()
    return [int(x) for x in raw.split(",")] if raw else []


def save_class_order(path: str, order: List[int]):
    with open(path, "w") as f:
        f.write(",".join(map(str, order)))


def resolve_real_class_dir(real_root: str, class_name: str, input_subdir: str) -> str:
    class_dir = os.path.join(real_root, class_name)
    if input_subdir:
        class_dir = os.path.join(class_dir, input_subdir)
    return class_dir


def read_training_config_num_classes(checkpoint_dir: str) -> int:
    cfg_path = os.path.join(checkpoint_dir, "training_config.txt")
    if not os.path.exists(cfg_path):
        return 0
    with open(cfg_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("num_classes:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return 0
    return 0


def remove_target_labels(features: np.ndarray, labels: np.ndarray, target_labels: List[int]):
    target_set = set(target_labels)
    keep = np.array([y not in target_set for y in labels], dtype=bool)
    return features[keep], labels[keep]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append WRES real classes to existing syn feature npy files")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--img_size", default=224, type=int)

    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--output_dir", default="wres_datasets", type=str)
    parser.add_argument("--pretrained_model_name", default="dinov3-vitl16-sl", type=str)
    parser.add_argument("--real_root", default="wres_datasets/real_train_dataset", type=str)
    parser.add_argument(
        "--real_input_subdir",
        default="",
        type=str,
        help="Optional subdirectory inside each real class folder to scan, e.g. LQ for UDC Train/Poled/LQ.",
    )
    parser.add_argument(
        "--real_classes",
        nargs="+",
        default=["RainReal", "SnowReal", "UnannotatedHazyImages"],
        type=str,
    )
    parser.add_argument(
        "--real_start_label",
        default=-1,
        type=int,
        help="Start label for real classes. If negative, auto-set to current synthetic class count.",
    )
    parser.add_argument(
        "--explicit_train_dir",
        default="",
        type=str,
        help="Use this directory directly as train data. Requires exactly one --real_classes value.",
    )
    parser.add_argument(
        "--explicit_test_dir",
        default="",
        type=str,
        help="Use this directory directly as test data instead of creating a random split.",
    )
    parser.add_argument("--split_filename", default="test_data_info.txt", type=str)
    parser.add_argument("--test_split", default=0.1, type=float)
    parser.add_argument("--force_resplit", action="store_true")
    parser.add_argument("--backup_original", action="store_true")
    args = parser.parse_args()

    if bool(args.explicit_train_dir) != bool(args.explicit_test_dir):
        parser.error("--explicit_train_dir and --explicit_test_dir must be provided together.")
    if args.explicit_train_dir and len(args.real_classes) != 1:
        parser.error("Explicit train/test directories require exactly one --real_classes value.")

    if args.seed != 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    exp_root = os.path.join(args.output_dir, args.exp_name)
    model_dir = os.path.join(exp_root, args.pretrained_model_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoint")

    features_path = os.path.join(model_dir, f"features-{args.pretrained_model_name}.npy")
    labels_path = os.path.join(model_dir, f"labels-{args.pretrained_model_name}.npy")
    test_features_path = os.path.join(model_dir, f"test_features-{args.pretrained_model_name}.npy")
    test_labels_path = os.path.join(model_dir, f"test_labels-{args.pretrained_model_name}.npy")
    train_manifest_path = os.path.join(model_dir, f"features_manifest-{args.pretrained_model_name}.jsonl")
    test_manifest_path = os.path.join(model_dir, f"test_features_manifest-{args.pretrained_model_name}.jsonl")
    class_order_file = os.path.join(model_dir, "class_order.txt")
    class_names_file = os.path.join(model_dir, "class_names.txt")

    for p in [features_path, labels_path, test_features_path, test_labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required npy file not found: {p}")

    base_names = load_class_names(class_names_file)
    existing_num_classes = len(base_names) if base_names else (np.load(labels_path).max().item() + 1)
    checkpoint_num_classes = read_training_config_num_classes(checkpoint_dir)
    base_num_classes = checkpoint_num_classes if checkpoint_num_classes > 0 else existing_num_classes
    if checkpoint_num_classes > 0 and checkpoint_num_classes != existing_num_classes:
        print(
            f"[Auto] Using checkpoint num_classes={checkpoint_num_classes} "
            f"instead of existing feature/class_names count={existing_num_classes}."
        )
    model_arg = parse_model_arg(args.pretrained_model_name)

    if args.real_start_label < 0:
        args.real_start_label = int(base_num_classes)
        print(f"[Auto] --real_start_label was not set. " f"Using base class count: {args.real_start_label}")
    if args.real_start_label < int(base_num_classes):
        raise ValueError(
            f"--real_start_label ({args.real_start_label}) overlaps with existing base labels "
            f"[0, {int(base_num_classes) - 1}]. Set --real_start_label >= {int(base_num_classes)}."
        )

    print(f"Loading checkpoint from: {checkpoint_dir}")
    # DINOv3 checkpoint uses a different module naming (o_proj/up_proj/down_proj).
    # Reuse ddp loader logic for robust target-module mapping.
    if "dinov3" in model_arg:
        print("Using DINOv3-compatible checkpoint loader (sl_finetuned_model_ddp).")
        dino = load_ckpt_ddp(
            checkpoint_dir=checkpoint_dir,
            num_classes=base_num_classes,
            model_name=model_arg,
            device=args.device,
            seed=args.seed,
        )
    else:
        dino = load_ckpt_single(
            checkpoint_dir=checkpoint_dir,
            num_classes=base_num_classes,
            model_name=model_arg,
            device=args.device,
            seed=args.seed,
        )

    #! 오류가 있엇음 interpolation, crop_pct 포함 버전(20260323)
    # interpolation = 3
    # crop_pct = 0.875
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # transform = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.Resize(int(args.img_size / crop_pct), interpolation),
    #         torchvision.transforms.CenterCrop(args.img_size),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    #     ]
    # )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = args.img_size  # 224
    transform = RandomResizedCropWithMetadata(image_size=image_size, mean=mean, std=std)

    train_samples = []
    test_samples = []
    label_to_name = dict(base_names)
    target_labels = []

    for i, cls_name in enumerate(args.real_classes):
        label = args.real_start_label + i
        target_labels.append(label)
        label_to_name[label] = cls_name

        if args.explicit_train_dir:
            train_class_dir = args.explicit_train_dir
            test_class_dir = args.explicit_test_dir
            if not os.path.isdir(train_class_dir):
                raise FileNotFoundError(f"Explicit train directory not found: {train_class_dir}")
            if not os.path.isdir(test_class_dir):
                raise FileNotFoundError(f"Explicit test directory not found: {test_class_dir}")
            rel_train = list_rel_images(train_class_dir)
            rel_test = list_rel_images(test_class_dir)
            if not rel_train:
                raise ValueError(f"No train images found in {train_class_dir}")
            if not rel_test:
                raise ValueError(f"No test images found in {test_class_dir}")
        else:
            train_class_dir = resolve_real_class_dir(args.real_root, cls_name, args.real_input_subdir)
            test_class_dir = train_class_dir
            if not os.path.isdir(train_class_dir):
                raise FileNotFoundError(f"Real class directory not found: {train_class_dir}")

            rel_all = list_rel_images(train_class_dir)
            rel_test = ensure_split_file(
                class_dir=train_class_dir,
                class_name=cls_name,
                split_filename=args.split_filename,
                test_split=args.test_split,
                seed=args.seed,
                class_offset=i,
                force_resplit=args.force_resplit,
            )
            rel_test_set = set(rel_test)
            rel_train = [r for r in rel_all if r not in rel_test_set]

        train_samples.extend(
            {
                "image_path": os.path.join(train_class_dir, r),
                "rel_path": r,
                "label": label,
                "class_name": cls_name,
                "feature_split": "train",
                "source": "real",
            }
            for r in rel_train
        )
        test_samples.extend(
            {
                "image_path": os.path.join(test_class_dir, r),
                "rel_path": r,
                "label": label,
                "class_name": cls_name,
                "feature_split": "test",
                "source": "real",
            }
            for r in rel_test
        )
        print(f"[{cls_name}] label={label}, train={len(rel_train)}, test={len(rel_test)}")

    real_train_ds = RealClassSplitDataset(train_samples, transform=transform)
    real_test_ds = RealClassSplitDataset(test_samples, transform=transform)

    real_train_loader = DataLoader(
        real_train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
    )
    real_test_loader = DataLoader(
        real_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
    )

    real_train_feat, real_train_label, real_train_manifest = infer_to_arrays(dino, real_train_loader, args.device)
    real_test_feat, real_test_label, real_test_manifest = infer_to_arrays(dino, real_test_loader, args.device)

    syn_feat = np.load(features_path)
    syn_label = np.load(labels_path)
    syn_test_feat = np.load(test_features_path)
    syn_test_label = np.load(test_labels_path)

    base_train_manifest = load_manifest_jsonl(train_manifest_path)
    if base_train_manifest is None or len(base_train_manifest) != len(syn_feat):
        print(
            f"[Warning] Missing or mismatched train manifest at {train_manifest_path}. "
            "Synthetic rows will keep placeholder metadata until base features are re-exported."
        )
        base_train_manifest = build_placeholder_manifest(syn_label, "train", label_to_name)
    base_test_manifest = load_manifest_jsonl(test_manifest_path)
    if base_test_manifest is None or len(base_test_manifest) != len(syn_test_feat):
        print(
            f"[Warning] Missing or mismatched test manifest at {test_manifest_path}. "
            "Synthetic rows will keep placeholder metadata until base features are re-exported."
        )
        base_test_manifest = build_placeholder_manifest(syn_test_label, "test", label_to_name)

    # Idempotent merge: remove existing target labels first, then append.
    syn_feat, syn_label = remove_target_labels(syn_feat, syn_label, target_labels)
    syn_test_feat, syn_test_label = remove_target_labels(syn_test_feat, syn_test_label, target_labels)

    base_train_manifest = reindex_manifest(remove_target_manifest_entries(base_train_manifest, target_labels), start=0)
    base_test_manifest = reindex_manifest(remove_target_manifest_entries(base_test_manifest, target_labels), start=0)
    real_train_manifest = reindex_manifest(real_train_manifest, start=len(base_train_manifest))
    real_test_manifest = reindex_manifest(real_test_manifest, start=len(base_test_manifest))

    merged_train_feat = np.concatenate([syn_feat, real_train_feat], axis=0)
    merged_train_label = np.concatenate([syn_label, real_train_label], axis=0)
    merged_test_feat = np.concatenate([syn_test_feat, real_test_feat], axis=0)
    merged_test_label = np.concatenate([syn_test_label, real_test_label], axis=0)
    merged_train_manifest = base_train_manifest + real_train_manifest
    merged_test_manifest = base_test_manifest + real_test_manifest

    if args.backup_original:
        for p in [features_path, labels_path, test_features_path, test_labels_path]:
            bak = p + ".bak_before_real_merge"
            if not os.path.exists(bak):
                os.rename(p, bak)
                print(f"Backup created: {bak}")
            else:
                print(f"Backup exists, keep: {bak}")

    np.save(features_path, merged_train_feat)
    np.save(labels_path, merged_train_label)
    np.save(test_features_path, merged_test_feat)
    np.save(test_labels_path, merged_test_label)
    save_manifest_jsonl(train_manifest_path, merged_train_manifest)
    save_manifest_jsonl(test_manifest_path, merged_test_manifest)

    # Update class names and class order.
    save_class_names(class_names_file, label_to_name)
    class_order = load_class_order(class_order_file)
    for y in target_labels:
        if y not in class_order:
            class_order.append(y)
    save_class_order(class_order_file, class_order)

    print("\nReal feature append completed.")
    print(f"Train: {merged_train_feat.shape}, labels={np.unique(merged_train_label, return_counts=True)}")
    print(f"Test:  {merged_test_feat.shape}, labels={np.unique(merged_test_label, return_counts=True)}")
    print(f"Updated class_order: {class_order}")
    print(f"Updated class_names: {class_names_file}")
    print(f"Updated train manifest: {train_manifest_path}")
    print(f"Updated test manifest: {test_manifest_path}")
