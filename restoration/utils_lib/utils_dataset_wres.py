"""
Utility functions for WRES paired dataset path generation.

Expected structure:
  data_root/
    CLASS_A/
      input/*.png
      gt/*.png
    CLASS_B/
      input/*.png
      gt/*.png
"""

import os

from .utils_incremental import get_old_new_classes


def _discover_classes(data_root, input_subdir="input", gt_subdir="gt"):
    classes = []
    if not os.path.isdir(data_root):
        return classes

    for name in sorted(os.listdir(data_root)):
        class_dir = os.path.join(data_root, name)
        if not os.path.isdir(class_dir):
            continue
        in_dir = os.path.join(class_dir, input_subdir)
        gt_dir = os.path.join(class_dir, gt_subdir)
        if os.path.isdir(in_dir) and os.path.isdir(gt_dir):
            classes.append(name)
    return classes


def generate_train_val_data(config, split="train"):
    if split == "train":
        data_root = config["train"]["data_root_train"]
    elif split == "val":
        data_root = config["train"]["data_root_val"]
    else:
        raise ValueError(f"Unknown split: {split}")

    input_subdir = config["train"].get("input_subdir", "input")
    gt_subdir = config["train"].get("gt_subdir", "gt")

    all_classes = config["train"].get("all_classes", [])
    if not all_classes:
        all_classes = _discover_classes(data_root, input_subdir=input_subdir, gt_subdir=gt_subdir)
        print(f"[WRES] Auto-discovered classes from {data_root}: {all_classes}")

    data_list = []
    for class_name in all_classes:
        class_dir = os.path.join(data_root, class_name)
        degraded_path = os.path.join(class_dir, input_subdir)
        clean_path = os.path.join(class_dir, gt_subdir)

        if not os.path.isdir(clean_path) or not os.path.isdir(degraded_path):
            print(
                f"[WRES] Skip class '{class_name}' (missing dirs): gt={clean_path}, input={degraded_path}"
            )
            continue

        data_list.append([clean_path, degraded_path, class_name])

    return data_list


def filter_old_new_data(data_list, old_classes, new_classes, mode="old"):
    if mode == "old":
        return [item for item in data_list if item[2] in old_classes]
    if mode == "new":
        return [item for item in data_list if item[2] in new_classes]
    if mode == "all":
        return data_list
    raise ValueError(f"Unknown mode: {mode}")


def get_train_val_data_for_stage(config):
    old_classes, new_classes, _, _ = get_old_new_classes(config)

    train_data_all = generate_train_val_data(config, split="train")
    val_data_all = generate_train_val_data(config, split="val")

    train_data_old = filter_old_new_data(train_data_all, old_classes, new_classes, mode="old")
    train_data_new = filter_old_new_data(train_data_all, old_classes, new_classes, mode="new")

    print(f"\n[WRES Dataset Filtering]")
    print(f"  Train root: {config['train']['data_root_train']}")
    print(f"  Val root:   {config['train']['data_root_val']}")
    print(f"  Train (OLD/Labeled): {len(train_data_old)} classes -> {[x[2] for x in train_data_old]}")
    print(f"  Train (NEW/Unlabeled): {len(train_data_new)} classes -> {[x[2] for x in train_data_new]}")
    print(f"  Val (ALL): {len(val_data_all)} classes -> {[x[2] for x in val_data_all]}\n")

    return train_data_old, train_data_new, val_data_all
