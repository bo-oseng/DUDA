"""Append features of the unlabeled real-world classes to the synthetic feature files.

Each real class receives a new label (``--real_start_label``, ...), and ``test_split`` of its images
is held out (listed in ``<real_root>/<class>/test_data_info.txt``, reused when present).
Running the script again replaces the previously appended real-class features.

Usage (from discovery/):
    python extract_real_features.py --feature_root <output_dir of train_adapter.py> --real_root <real_train_dataset>
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils_lib.datasets import LabeledImageDataset, collate, extract_features, feature_transform, split_real_class
from utils_lib.dino_adapter import load_adapter_backbone


def read_num_classes(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "training_config.txt")) as f:
        for line in f:
            if line.startswith("num_classes:"):
                return int(line.split(":", 1)[1])
    raise ValueError(f"num_classes not found in {checkpoint_dir}/training_config.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--exp_name", default="wres_with_clear")
    parser.add_argument("--model", default="dinov3-vitl16-sl")
    parser.add_argument("--real_root", required=True)
    parser.add_argument("--real_classes", nargs="+", default=["RainReal", "SnowReal", "UnannotatedHazyImages"])
    parser.add_argument("--real_start_label", type=int, default=None, help="Default: number of known classes")
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    feature_dir = os.path.join(args.feature_root, args.exp_name, args.model)
    checkpoint_dir = os.path.join(feature_dir, "checkpoint")

    def path(name):
        return os.path.join(feature_dir, f"{name}-{args.model}.npy")

    num_known = read_num_classes(checkpoint_dir)
    start = num_known if args.real_start_label is None else args.real_start_label
    if start < num_known:
        raise ValueError(f"--real_start_label ({start}) overlaps the known labels [0, {num_known - 1}]")
    backbone = load_adapter_backbone(
        checkpoint_dir, num_known, args.model[: -len("-sl")].replace("-", "_"), args.device, args.seed
    )

    transform = feature_transform(224)
    train_samples, test_samples, new_labels = [], [], []
    for i, name in enumerate(args.real_classes):
        label = start + i
        new_labels.append(label)
        class_dir = os.path.join(args.real_root, name)
        train_rel, test_rel = split_real_class(class_dir, name, i, args.test_split, args.seed)
        train_samples += [(os.path.join(class_dir, r), label) for r in train_rel]
        test_samples += [(os.path.join(class_dir, r), label) for r in test_rel]
        print(f"[{name}] label={label}, train={len(train_rel)}, test={len(test_rel)}")

    real = {}
    for split, samples in (("train", train_samples), ("test", test_samples)):
        loader = DataLoader(
            LabeledImageDataset(samples, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        real[split] = extract_features(backbone, loader, args.device)

    for prefix, split in (("", "train"), ("test_", "test")):
        features, labels = np.load(path(f"{prefix}features")), np.load(path(f"{prefix}labels"))
        keep = ~np.isin(labels, new_labels)  # drop real features appended by a previous run
        np.save(path(f"{prefix}features"), np.concatenate([features[keep], real[split][0]]))
        np.save(path(f"{prefix}labels"), np.concatenate([labels[keep], real[split][1]]))

    names_path, order_path = os.path.join(feature_dir, "class_names.txt"), os.path.join(feature_dir, "class_order.txt")
    with open(names_path) as f:
        names = {int(i): n for i, n in (line.strip().split(",", 1) for line in f if "," in line)}
    names.update({label: name for label, name in zip(new_labels, args.real_classes)})
    with open(names_path, "w") as f:
        f.writelines(f"{i},{names[i]}\n" for i in sorted(names))
    with open(order_path) as f:
        order = [int(x) for x in f.read().strip().split(",") if x]
    order += [label for label in new_labels if label not in order]
    with open(order_path, "w") as f:
        f.write(",".join(map(str, order)))
    print(f"Appended real features; class order: {order}")


if __name__ == "__main__":
    main()
