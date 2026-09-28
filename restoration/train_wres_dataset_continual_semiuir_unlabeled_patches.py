"""
Semi-UIR-style continual training entrypoint that uses prebuilt real_train_patches for unlabeled data.

This wrapper reuses train_wres_dataset_continual_semiuir_resume_mid_epoch.py but swaps the unlabeled dataset
builder so NEW-class data is read from lq_patches_dir/<class>/*.png instead of raw real images.
"""

import os

import numpy as np
import train_wres_dataset_continual as base
import train_wres_dataset_continual_semiuir_resume_mid_epoch as base_train
import datasets_wres_continual_semiuir as semi_data


def _extract_labels_from_dataset(dataset):
    if hasattr(dataset, "datasets"):
        labels = []
        for sub_dataset in dataset.datasets:
            if not hasattr(sub_dataset, "deg_class"):
                raise AttributeError("Expected sub-dataset to have 'deg_class' for capped labeled sampling.")
            labels.extend([int(sub_dataset.deg_class)] * len(sub_dataset))
        return np.asarray(labels, dtype=np.int64)

    if hasattr(dataset, "deg_class"):
        return np.full(len(dataset), int(dataset.deg_class), dtype=np.int64)

    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset)}")


def _apply_per_class_cap(dataset, labels, max_samples_per_class, seed):
    if max_samples_per_class is None or max_samples_per_class <= 0:
        return dataset, labels, None

    rng = np.random.default_rng(seed)
    selected_indices = []
    before = {}
    after = {}

    for cls_id in sorted(np.unique(labels).tolist()):
        cls_indices = np.where(labels == cls_id)[0]
        before[int(cls_id)] = int(len(cls_indices))

        if len(cls_indices) > max_samples_per_class:
            cls_indices = rng.choice(cls_indices, size=max_samples_per_class, replace=False)
            cls_indices = np.sort(cls_indices)

        after[int(cls_id)] = int(len(cls_indices))
        selected_indices.extend(cls_indices.tolist())

    selected_indices = np.asarray(sorted(selected_indices), dtype=np.int64)
    capped_dataset = base_train.Subset(dataset, selected_indices.tolist())
    capped_labels = labels[selected_indices]

    stats = {
        "original_total": int(len(labels)),
        "capped_total": int(len(capped_labels)),
        "before": before,
        "after": after,
        "max_samples_per_class": int(max_samples_per_class),
    }
    return capped_dataset, capped_labels, stats


def create_stage1_unlabeled_patch_dataset(
    lq_patches_dir, pseudo_patches_dir, fine_size=224, unlabeled_patch_dir=None
):
    if semi_data.base_data.NEW_CLASSES is None:
        raise ValueError("NEW_CLASSES not initialized. Call init_from_config() first!")
    if unlabeled_patch_dir and len(semi_data.base_data.NEW_CLASSES) != 1:
        raise ValueError("train.unlabeled_patch_dir requires exactly one NEW class.")

    datasets = []
    for deg_name in semi_data.base_data.NEW_CLASSES:
        patch_class_dir = unlabeled_patch_dir or os.path.join(lq_patches_dir, deg_name)
        pseudo_class_dir = os.path.join(pseudo_patches_dir, deg_name)

        if not os.path.isdir(patch_class_dir):
            print(f"[WRES-SemiUIR-PatchUL] Warning: NEW class patch dir not found: {patch_class_dir}")
            continue

        lq_patch_paths = semi_data.base_data._collect_images(patch_class_dir)
        if len(lq_patch_paths) == 0:
            print(f"[WRES-SemiUIR-PatchUL] Warning: no unlabeled patches found for NEW class: {deg_name}")
            continue

        os.makedirs(pseudo_class_dir, exist_ok=True)
        pseudo_patch_paths = [os.path.join(pseudo_class_dir, os.path.basename(path)) for path in lq_patch_paths]

        deg_class_orig = semi_data.base_data.DEG_MAP[deg_name]
        deg_class = semi_data.base_data.ORIG2CLASSIFIER[deg_class_orig]
        dataset = semi_data.TrainUnlabeledSemiUIR(
            lq_img_paths=lq_patch_paths,
            pseudo_img_paths=pseudo_patch_paths,
            deg_name=deg_name,
            deg_class=deg_class,
            fine_size=fine_size,
        )
        datasets.append(dataset)
        print(
            f"[WRES-SemiUIR-PatchUL] Loaded {len(dataset)} UNLABELED patches for NEW class: {deg_name}\n"
            f"  Patches: {patch_class_dir}\n"
            f"  Pseudo bank: {pseudo_class_dir}"
        )

    if len(datasets) == 0:
        raise ValueError("No stage unlabeled patch datasets found for NEW classes.")

    combined = semi_data.ConcatDataset(datasets)
    print(f"[WRES-SemiUIR-PatchUL] Total stage UNLABELED patches: {len(combined)}")
    return combined


def build_dataloaders_patch_unlabeled(cfg, rank, world_size):
    config_dict = {
        "deg_map": vars(cfg.deg_map),
        "incremental": {
            "stage": cfg.incremental.stage,
            "inc_class_num": cfg.incremental.inc_class_num,
            "base_class_num": cfg.incremental.base_class_num,
        },
        "cgcd": {
            "class_order": cfg.cgcd.class_order,
            "class_mappings": cfg.cgcd.class_mappings,
        },
        "train": {
            "data_root_train": cfg.train.data_root_train,
            "data_root_val": cfg.train.data_root_val,
            "all_classes": cfg.train.all_classes,
            "input_subdir": getattr(cfg.train, "input_subdir", "input"),
            "gt_subdir": getattr(cfg.train, "gt_subdir", "gt"),
            "semiuir_strong_aug_variant": getattr(cfg.train, "semiuir_strong_aug_variant", "default"),
        },
    }
    semi_data.init_from_config(config_dict)
    if rank == 0:
        base.print_incremental_info(config_dict)

    train_data_old, _, val_data_all = base.get_train_val_data_for_stage(config_dict)

    labeled_dataset = base.create_labeled_dataset(train_data_old, cfg.train.patch_size)
    max_samples_per_class = int(getattr(cfg.train, "max_samples_per_class", 0) or 0)
    if max_samples_per_class > 0:
        labeled_labels = _extract_labels_from_dataset(labeled_dataset)
        labeled_dataset, labeled_labels, cap_stats = _apply_per_class_cap(
            labeled_dataset,
            labeled_labels,
            max_samples_per_class=max_samples_per_class,
            seed=cfg.train.seed,
        )
        if rank == 0 and cap_stats is not None:
            print("\n" + "=" * 50)
            print("[WRES-SemiUIR-PatchUL] Labeled per-class cap applied")
            print("=" * 50)
            print(f"  max_samples_per_class: {cap_stats['max_samples_per_class']}")
            print(f"  total: {cap_stats['original_total']} -> {cap_stats['capped_total']}")
            print(f"  class counts (before): {cap_stats['before']}")
            print(f"  class counts (after):  {cap_stats['after']}")

    unlabeled_dataset = create_stage1_unlabeled_patch_dataset(
        lq_patches_dir=cfg.train.lq_patches_dir,
        pseudo_patches_dir=cfg.train.pseudo_patches_dir,
        fine_size=int(getattr(cfg.train, "unlabeled_fine_size", cfg.train.patch_size)),
        unlabeled_patch_dir=getattr(cfg.train, "unlabeled_patch_dir", None),
    )

    use_validation = bool(getattr(cfg.train, "use_validation", False))
    if use_validation:
        val_datasets = base.create_stage1_val_datasets(val_data_all)
    else:
        val_datasets = []
        if rank == 0:
            print("[WRES-SemiUIR-PatchUL] Validation disabled (train.use_validation=False)")

    use_real_eval = bool(getattr(cfg.train, "use_real_eval", False))
    if use_real_eval:
        real_eval_datasets = base.create_real_eval_datasets(
            getattr(cfg.train, "real_eval_root", ""),
            list(getattr(cfg.train, "real_eval_sets", [])),
        )
    else:
        real_eval_datasets = []
        if rank == 0:
            print("[WRES-SemiUIR-PatchUL] Real eval disabled (train.use_real_eval=False)")

    labeled_sampler = base_train.DistributedSampler(
        labeled_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed
    )
    unlabeled_sampler = base_train.DistributedSampler(
        unlabeled_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed
    )

    pin_memory = bool(getattr(cfg.train, "pin_memory", True))
    prefetch_factor = int(getattr(cfg.train, "prefetch_factor", 4))
    persistent_workers = bool(getattr(cfg.train, "persistent_workers", True))

    loader_kwargs = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if cfg.train.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        loader_kwargs["persistent_workers"] = persistent_workers

    if rank == 0:
        print(
            "[WRES-SemiUIR-PatchUL] Train loader config: "
            f"batch_size={loader_kwargs['batch_size']}, "
            f"num_workers={loader_kwargs['num_workers']}, "
            f"pin_memory={loader_kwargs['pin_memory']}, "
            f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'N/A')}, "
            f"persistent_workers={loader_kwargs.get('persistent_workers', False)}"
        )

    labeled_loader = base_train.DataLoader(labeled_dataset, sampler=labeled_sampler, **loader_kwargs)
    unlabeled_loader = base_train.DataLoader(unlabeled_dataset, sampler=unlabeled_sampler, **loader_kwargs)

    return (
        labeled_loader,
        unlabeled_loader,
        val_datasets,
        real_eval_datasets,
        labeled_sampler,
        unlabeled_sampler,
    )


base_train.build_dataloaders = build_dataloaders_patch_unlabeled


if __name__ == "__main__":
    print("[WRES-SemiUIR-PatchUL] Using prebuilt unlabeled patches from train.lq_patches_dir.")
    base_train.main()
