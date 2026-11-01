"""Prepare the WRES directory layout, unpaired patches, or the paper monitoring subset."""

import argparse
import os
from pathlib import Path

from preprocess_unpaired_patches import patchify_directory

ROOT = Path(__file__).resolve().parents[1]
REAL_CLASSES = ("RainReal", "SnowReal", "UnannotatedHazyImages")
PAIRED_CLASSES = ("OTS", "Outdoor_Rain", "RainDrop", "SPA", "Snow100k")
EVAL_SETS = ("RealRain_2320", "RTTS", "Snow100K_R")


def create_layout(data_root):
    directories = [data_root / "syn_train_dataset/clear/gt"]
    directories += [data_root / "syn_train_dataset" / name / split for name in PAIRED_CLASSES for split in ("input", "gt")]
    directories += [data_root / split / name for split in ("real_train_dataset", "real_train_patches") for name in REAL_CLASSES]
    directories += [data_root / split / name for split in ("real_eval_dataset", "real_eval_small_dataset") for name in EVAL_SETS]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directories ready: {data_root}")


def prepare_patches(data_root, num_workers):
    # Preflight every class before writing any patches. Indexing restarts per class,
    # matching the existing paper patch filenames and dataset ordering.
    for name in REAL_CLASSES:
        source = data_root / "real_train_dataset" / name
        target = data_root / "real_train_patches" / name
        if not source.is_dir():
            raise ValueError(f"Missing unpaired image directory: {source}")
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Patch directory is not empty: {target}; keep existing patches or choose another data root")
    summaries = {}
    for name in REAL_CLASSES:
        summary = patchify_directory(
            data_root / "real_train_dataset" / name,
            data_root / "real_train_patches" / name,
            patch_size=224,
            stride=224,
            num_workers=num_workers,
        )
        summaries[name] = summary
        print(f"[{name}] {summary}")
        if summary["failed_images"]:
            raise RuntimeError(f"Patch extraction failed for {name}; see the image errors above")
    return summaries


def prepare_eval_subset(data_root):
    pending = []
    for name in EVAL_SETS:
        manifest = ROOT / "datasets/manifests/real_eval_small" / f"{name}.txt"
        for filename in manifest.read_text().splitlines():
            source = data_root / "real_eval_dataset" / name / filename
            target = data_root / "real_eval_small_dataset" / name / filename
            if not source.is_file():
                raise ValueError(f"Monitoring image missing from full evaluation set: {source}")
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            if target.exists() or target.is_symlink():
                raise ValueError(f"Monitoring file already exists; refusing to replace it: {target}")
            pending.append((source, target))
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(source, target.parent))
    print(f"Linked {len(pending)} monitoring images from the full evaluation sets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "patches", "eval-subset"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", ROOT / "datasets/wres_datasets")))
    parser.add_argument("--num-workers", type=int, default=4, help="Worker processes for patch extraction (default: 4)")
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    data_root = args.data_root.expanduser().resolve()
    if args.action == "init":
        create_layout(data_root)
    elif args.action == "patches":
        prepare_patches(data_root, args.num_workers)
    else:
        prepare_eval_subset(data_root)


if __name__ == "__main__":
    main()
