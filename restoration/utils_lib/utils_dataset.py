"""
Utility functions for automatic dataset path generation.
"""

import os
from .utils_incremental import get_old_new_classes


def generate_train_val_data(config, split="train"):
    if split == "train":
        data_root = config["train"]["data_root_train"]
    elif split == "val":
        data_root = config["train"]["data_root_val"]
    else:
        raise ValueError(f"Unknown split: {split}")

    all_classes = config["train"]["all_classes"]
    clean_path = os.path.join(data_root, "clear")

    data_list = []
    for class_name in all_classes:
        degraded_path = os.path.join(data_root, class_name)
        data_list.append([clean_path, degraded_path, class_name])

    return data_list


def filter_old_new_data(data_list, old_classes, new_classes, mode="old"):
    if mode == "old":
        return [item for item in data_list if item[2] in old_classes]
    elif mode == "new":
        return [item for item in data_list if item[2] in new_classes]
    elif mode == "all":
        return data_list
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_train_val_data_for_stage(config):
    # Get OLD/NEW class lists
    old_classes, new_classes, old_order_indices, new_order_indices = get_old_new_classes(config)

    # Generate all data paths
    train_data_all = generate_train_val_data(config, split="train")
    val_data_all = generate_train_val_data(config, split="val")

    # Filter train data: only OLD classes for labeled training
    train_data_old = filter_old_new_data(train_data_all, old_classes, new_classes, mode="old")
    train_data_new = filter_old_new_data(train_data_all, old_classes, new_classes, mode="new")

    # Validation: use ALL classes to measure both OLD (forgetting) and NEW (learning)
    # No filtering needed for validation

    print(f"\n[Dataset Filtering]")
    print(f"  Train (OLD/Labeled): {len(train_data_old)} classes")
    print(f"  Train (NEW/Unlabeled): {len(train_data_new)} classes (not used in this function)")
    print(f"  Val (ALL): {len(val_data_all)} classes\n")

    return train_data_old, train_data_new, val_data_all


if __name__ == "__main__":
    # Test
    import yaml
    from datasets_my import init_from_config

    config_path = "configs/train_wcgcd_stage1_lora_debug.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize dataset module
    init_from_config(config)

    # Generate data lists
    train_old, train_new, val_all = get_train_val_data_for_stage(config)

    print("=" * 60)
    print("Train Data (OLD classes - Labeled):")
    print("=" * 60)
    for item in train_old:
        print(f"  {item}")

    print("\n" + "=" * 60)
    print("Train Data (NEW classes - Unlabeled):")
    print("=" * 60)
    for item in train_new:
        print(f"  {item}")

    print("\n" + "=" * 60)
    print("Val Data (ALL classes):")
    print("=" * 60)
    for item in val_all:
        print(f"  {item}")
