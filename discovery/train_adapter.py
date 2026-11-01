"""Fine-tune the DINOv3 BOFT adapter on the known (synthetic) degradation classes and export features.

Writes ``<output_dir>/<exp_name>/<model>-sl/``:
    checkpoint/                         adapter, base-layer biases, and classification head
    features-*.npy, labels-*.npy        training-split features z and labels
    test_features-*.npy, test_labels-*  held-out features and labels
    class_order.txt, class_names.txt, class_mappings.json
The held-out images of each class are listed in ``<train_root>/<class>/test_data_info.txt`` and
reused when that file exists.

Usage (from discovery/):
    torchrun --nproc_per_node=3 train_adapter.py --train_root <syn_train_dataset> --output_dir <dir>
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from utils_lib.datasets import (
    LabeledImageDataset,
    collate,
    extract_features,
    feature_transform,
    save_class_metadata,
    split_synthetic,
    synthetic_classes,
    synthetic_samples,
)
from utils_lib.dino_adapter import finetune_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train_root", required=True, help="Directory with <class>/input/* for the known classes")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exp_name", default="wres_with_clear")
    parser.add_argument("--model", default="dinov3_vitl16")
    parser.add_argument("--input_subdir", default="input")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--max_samples_per_class", type=int, default=900, help="Images drawn per class before the split"
    )
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = f"cuda:{local_rank}"
    is_main = local_rank == 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform = feature_transform(224)
    classes = synthetic_classes(args.train_root)
    samples = synthetic_samples(args.train_root, classes, args.input_subdir)
    # Rank 0 writes missing split files first; the other ranks then read them.
    if is_main:
        train_idx, test_idx = split_synthetic(
            args.train_root, classes, samples, args.test_split, args.seed, args.max_samples_per_class
        )
    dist.barrier()
    if not is_main:
        train_idx, test_idx = split_synthetic(
            args.train_root, classes, samples, args.test_split, args.seed, args.max_samples_per_class
        )
    train_set = LabeledImageDataset([samples[i] for i in train_idx], transform)
    test_set = LabeledImageDataset([samples[i] for i in test_idx], transform) if test_idx else train_set
    print(f"Classes: {classes}, train samples: {len(train_set)}, test samples: {len(test_set)}")

    feature_dir = os.path.join(args.output_dir, args.exp_name, args.model.replace("_", "-") + "-sl")
    backbone = finetune_adapter(
        train_set,
        num_classes=len(classes),
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model,
        save_dir=os.path.join(feature_dir, "checkpoint"),
        seed=args.seed,
        test_set=test_set,
    )

    if is_main:
        save_class_metadata(feature_dir, classes)
        suffix = os.path.basename(feature_dir)
        for prefix, dataset in (("", train_set), ("test_", test_set)):
            loader = DataLoader(
                dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers, shuffle=False
            )
            features, labels = extract_features(backbone, loader, device, batch_order="filename")
            np.save(os.path.join(feature_dir, f"{prefix}features-{suffix}.npy"), features)
            np.save(os.path.join(feature_dir, f"{prefix}labels-{suffix}.npy"), labels)
            print(f"Saved {prefix}features: {features.shape}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
