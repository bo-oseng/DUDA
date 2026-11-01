"""Evaluate a DUDA checkpoint on real-world sets with MUSIQ, CLIP-IQA, and LIQE (Table 1).

Restored images are saved under ``<output>/<model>/<set>/output/`` for the separate Q-Align
evaluation (evaluate_qalign.py). The overall score is the unweighted mean over the sets.

Usage (from restoration/):
    torchrun --nproc_per_node=3 evaluate.py --config <config> --checkpoint <ckpt> --output <dir>
"""

import argparse
import csv
import datetime
import json
import os

import pyiqa
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from utils_lib.checkpoint import load_for_inference
from utils_lib.common import amp_dtype_from_cfg, enable_tf32, load_config, seed_everything
from utils_lib.datasets import build_real_eval_datasets
from utils_lib.descriptor import DiscoveryInstanceDescriptor
from utils_lib.dino import load_degradation_encoder
from utils_lib.onerestore import OneRestore
from utils_lib.real_eval import dino_resize

METRICS = [("musiq", "MUSIQ"), ("clipiqa", "CLIP-IQA"), ("liqe", "LIQE")]


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def prepare_for_metric(metric_name, image):
    """LIQE expects the shorter side to be at least 224 pixels."""
    if metric_name == "liqe":
        _, _, h, w = image.shape
        if min(h, w) < 224:
            scale = 224.0 / float(min(h, w))
            size = (max(224, int(round(h * scale))), max(224, int(round(w * scale))))
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    return image


@torch.no_grad()
def evaluate(model, dino, descriptor, datasets, device, rank, world_size, use_ddp, amp_dtype, args, model_tag):
    model.eval()
    model.set_runtime_resize_fallback(True)
    metrics = {name: pyiqa.create_metric(name, as_loss=False, device=device) for name, _ in METRICS}

    local_sums, local_records = {}, {}
    for dataset, set_name in datasets:
        indices = list(range(rank, len(dataset), world_size)) if use_ddp else list(range(len(dataset)))
        loader = DataLoader(
            Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
        save_dir = os.path.join(args.output, model_tag, set_name, "output")
        os.makedirs(save_dir, exist_ok=True)
        sums = {name: 0.0 for name, _ in METRICS}
        records = []
        desc = f"[GPU {rank}] {set_name:<18} ({len(indices)}/{len(dataset)})"
        for position, (image, path) in enumerate(tqdm(loader, desc=desc, leave=False, position=rank, ncols=100)):
            image, path = image.to(device, non_blocking=True), path[0]
            os.environ["ONERESTORE_DEBUG_SAMPLE"] = os.path.basename(path)
            try:
                with autocast("cuda", dtype=amp_dtype):
                    embedding, _ = descriptor(dino(dino_resize(image)).pooler_output)
                    restored = model(image, embedding)
            finally:
                os.environ.pop("ONERESTORE_DEBUG_SAMPLE", None)
            restored = torch.clamp(restored, 0, 1)

            # Match the saved-PNG evaluation used for the paper's per-image records.
            metric_image = (restored * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).float() / 255.0
            record = {"set": set_name, "file": os.path.basename(path)}
            for name, _ in METRICS:
                record[name] = float(metrics[name](prepare_for_metric(name, metric_image)).squeeze().item())
                sums[name] += record[name]
            records.append((indices[position], record))
            save_image(restored[0], os.path.join(save_dir, f"{os.path.splitext(os.path.basename(path))[0]}.png"))
        local_sums[set_name] = {"count": len(indices), "total": len(dataset), **sums}
        local_records[set_name] = records
        line = ", ".join(f"{label}: {sums[name] / max(len(indices), 1):.4f}" for name, label in METRICS)
        print(f"[GPU {rank}] {set_name:20s} - {line} ({len(indices)}/{len(dataset)} imgs)")

    gathered_sums, gathered_records = [local_sums], [local_records]
    if use_ddp:
        gathered_sums, gathered_records = [None] * world_size, [None] * world_size
        dist.all_gather_object(gathered_sums, local_sums)
        dist.all_gather_object(gathered_records, local_records)
    if rank != 0:
        return None

    per_set, rows = {}, []
    for _, set_name in datasets:
        count = sum(g[set_name]["count"] for g in gathered_sums)
        per_set[set_name] = {"count": count}
        per_set[set_name].update({name: sum(g[set_name][name] for g in gathered_sums) / count for name, _ in METRICS})
        items = sorted((item for g in gathered_records for item in g[set_name]), key=lambda item: item[0])
        rows.extend(record for _, record in items)
    average = {name: sum(stats[name] for stats in per_set.values()) / len(per_set) for name, _ in METRICS}
    return {
        "model": model_tag,
        "metric_names": [name for name, _ in METRICS],
        "metric_labels": dict(METRICS),
        "num_sets": len(per_set),
        "per_set": per_set,
        "average": average,
    }, rows


def save_results(summary, rows, output_dir, model_tag):
    names = summary["metric_names"]
    with open(os.path.join(output_dir, f"{model_tag}_real_eval_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, f"{model_tag}_real_eval_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "set", "count"] + names)
        writer.writeheader()
        for set_name, stats in summary["per_set"].items():
            writer.writerow({"model": model_tag, "set": set_name, **stats})
        total = sum(stats["count"] for stats in summary["per_set"].values())
        writer.writerow({"model": model_tag, "set": "AVG", "count": total, **summary["average"]})
    with open(os.path.join(output_dir, f"{model_tag}_metrics_per_sample.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["set", "file"] + names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Directory for restored images and metrics")
    parser.add_argument("--model", default="teacher", choices=["teacher", "student"])
    parser.add_argument("--real_eval_root", default=None, help="Default: train.real_eval_root of the config")
    parser.add_argument("--real_eval_sets", default=None, help="Comma-separated; default: train.real_eval_sets")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rank, world_size, local_rank, use_ddp = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    enable_tf32()
    _, amp_dtype = amp_dtype_from_cfg(cfg)
    seed_everything(cfg.train.seed + rank)

    root = args.real_eval_root or cfg.train.real_eval_root
    set_names = args.real_eval_sets.split(",") if args.real_eval_sets else list(cfg.train.real_eval_sets)
    datasets = build_real_eval_datasets(root, set_names)

    dino = load_degradation_encoder(
        cfg.discovery.dino_checkpoint, cfg.discovery.dino_num_classes, cfg.discovery.dino_model, device
    )
    descriptor = DiscoveryInstanceDescriptor(
        cluster_dir=cfg.discovery.cluster_dir,
        output_dim=cfg.discovery.descriptor_dim,
        stage=cfg.incremental.stage,
        pca_path=cfg.discovery.pca_path,
        scaler_path=cfg.discovery.scaler_path,
    ).to(device)
    model = OneRestore(channel=cfg.model.width).to(device)
    epoch = load_for_inference(args.checkpoint, model, descriptor, use_student=args.model == "student")
    descriptor.eval()
    if rank == 0:
        print(f"Checkpoint: {args.checkpoint} (epoch {epoch}), model: {args.model}, sets: {set_names}")

    result = evaluate(model, dino, descriptor, datasets, device, rank, world_size, use_ddp, amp_dtype, args, args.model)
    if rank == 0:
        summary, rows = result
        save_results(summary, rows, args.output, args.model)
        print(f"\n{'=' * 80}\n[{args.model}] Per-set results\n{'=' * 80}")
        for set_name, stats in summary["per_set"].items():
            line = ", ".join(f"{label}: {stats[name]:.4f}" for name, label in METRICS)
            print(f"{set_name:24s} {line} ({stats['count']} imgs)")
        line = ", ".join(f"{label}: {summary['average'][name]:.4f}" for name, label in METRICS)
        print(f"{'-' * 80}\n{'Average':24s} {line}\n{'=' * 80}")
        print(f"Saved results to {args.output}")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
