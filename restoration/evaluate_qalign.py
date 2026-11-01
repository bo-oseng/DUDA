"""Q-Align scores of restored images saved by evaluate.py (Table 1).

Runs in a separate environment (see requirements-qalign.txt) and depends only on PyIQA.
The overall score is the unweighted mean over the sets.

Usage (from restoration/):
    torchrun --nproc_per_node=1 evaluate_qalign.py --results_root <dir used by evaluate.py>
"""

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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def install_qalign_compat():
    """Q-Align's remote model code refers to these transformers symbols as globals."""
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

    builtins.Cache = Cache
    builtins.BaseModelOutputWithPast = BaseModelOutputWithPast
    builtins.CausalLMOutputWithPast = CausalLMOutputWithPast


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = read_image(str(self.paths[index])).float() / 255.0
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image[:3], str(self.paths[index])


def find_sets(results_root, model_tag, set_names):
    model_root = results_root / model_tag
    names = (
        set_names.split(",") if set_names else sorted(p.name for p in model_root.iterdir() if (p / "output").is_dir())
    )
    sets = []
    for name in names:
        output_dir = model_root / name / "output"
        paths = sorted(p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise FileNotFoundError(f"No restored images found in {output_dir}")
        sets.append((name, paths))
    return sets


@torch.no_grad()
def score_sets(sets, metric, device, rank, world_size, use_ddp, num_workers):
    local = {}
    for set_name, paths in sets:
        indices = list(range(rank, len(paths), world_size)) if use_ddp else list(range(len(paths)))
        loader = DataLoader(
            Subset(ImageDataset(paths), indices),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        records = []
        for position, (image, path) in enumerate(
            tqdm(loader, desc=f"[GPU {rank}] {set_name}", position=rank, ncols=100)
        ):
            score = float(metric(image.to(device, non_blocking=True)).reshape(-1)[0].item())
            records.append((indices[position], {"set": set_name, "file": Path(path[0]).name, "qalign": score}))
        local[set_name] = records
        print(
            f"[GPU {rank}] {set_name:20s} - Q-Align: {sum(r['qalign'] for _, r in records) / max(len(records), 1):.4f}"
        )

    gathered = [local]
    if use_ddp:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    if rank != 0:
        return None

    per_set, rows = {}, []
    for set_name, paths in sets:
        items = sorted((item for g in gathered for item in g[set_name]), key=lambda item: item[0])
        records = [record for _, record in items]
        per_set[set_name] = {"qalign": sum(r["qalign"] for r in records) / len(records), "count": len(paths)}
        rows.extend(records)
    average = sum(stats["qalign"] for stats in per_set.values()) / len(per_set)
    return per_set, average, rows


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results_root", required=True, help="Directory containing <model>/<set>/output")
    parser.add_argument("--model", default="teacher")
    parser.add_argument("--sets", default=None, help="Comma-separated set names; default: all sets found")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    rank, world_size, local_rank, use_ddp = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    results_root = Path(args.results_root)
    install_qalign_compat()
    metric = pyiqa.create_metric("qalign", as_loss=False, device=str(device))
    sets = find_sets(results_root, args.model, args.sets)

    result = score_sets(sets, metric, device, rank, world_size, use_ddp, args.num_workers)
    if rank == 0:
        per_set, average, rows = result
        prefix = results_root / f"{args.model}_qalign"
        summary = {"model": args.model, "per_set": per_set, "average": {"qalign": average}}
        Path(f"{prefix}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with open(f"{prefix}_per_sample.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["set", "file", "qalign"])
            writer.writeheader()
            writer.writerows(rows)
        for set_name, stats in per_set.items():
            print(f"{set_name:20s} Q-Align: {stats['qalign']:.4f} ({stats['count']} imgs)")
        print(f"{'Average':20s} Q-Align: {average:.4f}\nSaved results to {prefix}_metrics.json")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
