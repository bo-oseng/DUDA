"""OneRestore + CGCD training and evaluation utilities."""

import argparse
import builtins
import csv
import datetime
import json
import os
from pathlib import Path

import pyiqa
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
from tqdm import tqdm

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def install_qalign_compat():
    # Q-Align's cached HF module references older transformers symbols as globals.
    try:
        from transformers.cache_utils import Cache
        from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    except Exception:
        return
    builtins.Cache = Cache
    builtins.BaseModelOutputWithPast = BaseModelOutputWithPast
    builtins.CausalLMOutputWithPast = CausalLMOutputWithPast


def resolve_device(device_arg, local_rank, use_ddp):
    if use_ddp:
        if device_arg.startswith("cuda"):
            return torch.device(f"cuda:{local_rank}")
        raise ValueError("DDP mode requires a CUDA device.")
    return torch.device(device_arg)


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = read_image(str(path)).float() / 255.0
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]
        return image, str(path)


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_sets(results_root, model_tag, set_names):
    model_root = results_root / model_tag
    if not model_root.exists():
        raise FileNotFoundError(f"Model result directory not found: {model_root}")

    if set_names:
        names = [name.strip() for name in str(set_names).split(",") if name.strip()]
    else:
        names = sorted([p.name for p in model_root.iterdir() if p.is_dir() and (p / "output").is_dir()])

    resolved = []
    for name in names:
        output_dir = model_root / name / "output"
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory not found for set '{name}': {output_dir}")
        paths = sorted([p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])
        if not paths:
            raise FileNotFoundError(f"No image files found in: {output_dir}")
        resolved.append((name, output_dir, paths))
    return resolved


@torch.no_grad()
def evaluate_saved_qalign(discovered_sets, metric, device, rank, world_size, use_ddp, num_workers):
    all_set_names = [set_name for set_name, _, _ in discovered_sets]

    if rank == 0:
        print(f"\n{'=' * 80}")
        print("[SavedQAlign] Metrics: Q-Align")
        print(f"{'=' * 80}")

    local_results = {}
    local_sample_records = {}

    for set_name, output_dir, image_paths in discovered_sets:
        total_count = len(image_paths)
        if use_ddp:
            local_indices = list(range(rank, total_count, world_size))
        else:
            local_indices = list(range(total_count))

        dataset = ImagePathDataset(image_paths)
        subset = Subset(dataset, local_indices)
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        qalign_sum = 0.0
        sample_records = []
        iterator = tqdm(
            loader,
            desc=f"[GPU {rank}][SavedQAlign] {set_name:<18} ({len(local_indices)}/{total_count})",
            leave=False,
            position=rank,
            ncols=100,
        )

        for local_pos, batch in enumerate(iterator):
            image = batch[0].to(device, non_blocking=True)
            path = batch[1][0]
            global_idx = local_indices[local_pos]

            score = float(metric(image).reshape(-1)[0].item())
            qalign_sum += score
            sample_records.append(
                (
                    int(global_idx),
                    {
                        "set": set_name,
                        "file": Path(path).name,
                        "qalign": score,
                    },
                )
            )

        local_count = len(local_indices)
        local_mean = qalign_sum / local_count if local_count > 0 else 0.0
        local_results[set_name] = {
            "count_total": total_count,
            "count_local": local_count,
            "qalign_sum": qalign_sum,
            "output_dir": str(output_dir),
        }
        local_sample_records[set_name] = sample_records
        print(
            f"[GPU {rank}][SavedQAlign] {set_name:20s} - Q-Align: {local_mean:.4f} ({local_count}/{total_count} imgs)"
        )

    if use_ddp:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        gathered_sample_records = [None] * world_size
        dist.all_gather_object(gathered_sample_records, local_sample_records)
    else:
        gathered_results = [local_results]
        gathered_sample_records = [local_sample_records]

    if rank != 0:
        return None

    merged = {}
    per_sample_rows = []
    avg_sum = 0.0
    valid_count = 0

    print(f"\n{'=' * 80}")
    print("[SavedQAlign] Per-set Results")
    print(f"{'=' * 80}")

    for set_name in all_set_names:
        count_total = 0
        count_local = 0
        qalign_sum_total = 0.0
        output_dir = None
        sample_items = []

        for result_dict in gathered_results:
            if not result_dict or set_name not in result_dict:
                continue
            row = result_dict[set_name]
            count_total = max(count_total, int(row.get("count_total", 0)))
            count_local += int(row.get("count_local", 0))
            qalign_sum_total += float(row.get("qalign_sum", 0.0))
            output_dir = output_dir or row.get("output_dir")

        for record_dict in gathered_sample_records:
            if not record_dict or set_name not in record_dict:
                continue
            sample_items.extend(record_dict[set_name])

        if count_total <= 0:
            continue

        if count_local != count_total:
            print(f"[SavedQAlign][Warning] {set_name}: aggregated {count_local} samples but expected {count_total}.")

        mean_qalign = qalign_sum_total / float(count_local if count_local > 0 else 1)
        merged[set_name] = {
            "qalign": mean_qalign,
            "count": count_total,
            "output_dir": output_dir,
        }
        avg_sum += mean_qalign
        valid_count += 1

        sample_items.sort(key=lambda x: x[0])
        per_sample_rows.extend([record for _, record in sample_items])

        print(f"{set_name:20s} Q-Align: {mean_qalign:.4f} ({count_total} imgs)")

    avg_qalign = avg_sum / valid_count if valid_count > 0 else 0.0

    print(f"{'-' * 80}")
    print(f"{'SavedQAlign Avg':20s} Q-Align: {avg_qalign:.4f}")
    print(f"{'=' * 80}")

    return {
        "model": "teacher",
        "metric_names": ["qalign"],
        "metric_labels": {"qalign": "Q-Align"},
        "per_set": merged,
        "average": {"qalign": avg_qalign},
        "per_sample_rows": per_sample_rows,
    }


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Evaluate Q-Align from already-saved restored real-eval images only.")
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root result dir that contains <model_tag>/<set_name>/output",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="teacher",
        help="Model subdir under results_root (default: teacher)",
    )
    parser.add_argument(
        "--sets",
        type=str,
        default=None,
        help="Comma-separated set names. Default: auto-discover all sets with output dirs",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers per rank")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for single-process mode; DDP uses local CUDA rank",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="saved_qalign",
        help="Output filename suffix before _metrics.json/.csv and _per_sample.csv",
    )
    args = parser.parse_args()

    rank, world_size, local_rank, use_ddp = setup_ddp()
    device = resolve_device(args.device, local_rank, use_ddp)
    results_root = Path(args.results_root)

    install_qalign_compat()
    metric = pyiqa.create_metric("qalign", as_loss=False, device=str(device))
    discovered_sets = discover_sets(results_root, args.model_tag, args.sets)

    if rank == 0:
        print(f"\n{'=' * 80}")
        print("Evaluation Mode - Saved Real Q-Align")
        print(f"{'=' * 80}")
        print(f"Results root: {results_root}")
        print(f"Model tag: {args.model_tag}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Sets: {', '.join(name for name, _, _ in discovered_sets)}")
        print(f"Suffix: {args.suffix}")
        print(f"{'=' * 80}\n")

    summary = evaluate_saved_qalign(
        discovered_sets=discovered_sets,
        metric=metric,
        device=device,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
        num_workers=args.num_workers,
    )

    if rank == 0 and summary is not None:
        summary["model"] = args.model_tag
        per_set = summary["per_set"]
        avg_qalign = float(summary["average"]["qalign"])
        total_count = sum(int(stats["count"]) for stats in per_set.values())

        json_path = results_root / f"{args.model_tag}_{args.suffix}_metrics.json"
        csv_path = results_root / f"{args.model_tag}_{args.suffix}_metrics.csv"
        per_sample_csv_path = results_root / f"{args.model_tag}_{args.suffix}_per_sample.csv"

        save_json(
            json_path,
            {
                "model": args.model_tag,
                "metric_names": ["qalign"],
                "metric_labels": {"qalign": "Q-Align"},
                "per_set": per_set,
                "average": {"qalign": avg_qalign},
            },
        )
        save_csv(
            csv_path,
            [
                {
                    "model": args.model_tag,
                    "set": set_name,
                    "count": stats["count"],
                    "qalign": stats["qalign"],
                }
                for set_name, stats in per_set.items()
            ]
            + [
                {
                    "model": args.model_tag,
                    "set": "AVG",
                    "count": total_count,
                    "qalign": avg_qalign,
                }
            ],
            ["model", "set", "count", "qalign"],
        )
        save_csv(per_sample_csv_path, summary["per_sample_rows"], ["set", "file", "qalign"])

        print(f"Saved summary JSON: {json_path}")
        print(f"Saved summary CSV: {csv_path}")
        print(f"Saved per-sample CSV: {per_sample_csv_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
