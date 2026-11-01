"""No-reference IQA on real-world images during training (monitoring only).

Each set is sharded across ranks and the per-set means are averaged without weighting.
The metrics and data loaders are created at every call, as in the original training runs.
"""

import os
import sys
import time

import pyiqa
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from tqdm import tqdm

# DINO input: resize to 224 and ImageNet normalization.
dino_resize = v2.Compose([v2.Resize([224, 224]), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _shard_loader(dataset, rank, world_size, num_workers=2):
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True
    )
    return loader, indices


@torch.no_grad()
def evaluate_real_iqa(
    model, dino_model, descriptor, datasets, device, writer, rank, world_size, amp_dtype, tb_step, label, tb_prefix
):
    """MUSIQ and CLIP-IQA of the restored images. Used by the semi-supervised stage."""
    if not datasets:
        return 0.0
    model.eval()
    model.set_runtime_resize_fallback(True)
    dino_model.eval()
    descriptor.eval()
    metrics = [
        ("musiq", "MUSIQ", pyiqa.create_metric("musiq", as_loss=False, device=device)),
        ("clipiqa", "CLIP-IQA", pyiqa.create_metric("clipiqa", as_loss=False, device=device)),
    ]

    local = {}
    for dataset, set_name in datasets:
        loader, indices = _shard_loader(dataset, rank, world_size)
        sums = {name: 0.0 for name, _, _ in metrics}
        count = 0
        iterator = loader
        if rank == 0 and sys.stderr.isatty():
            iterator = tqdm(loader, desc=f"[{tb_prefix}] {set_name:<18}", leave=False, ncols=100)
        for image, path in iterator:
            image = image.to(device, non_blocking=True)
            count += int(image.shape[0])
            os.environ["ONERESTORE_DEBUG_SAMPLE"] = os.path.basename(path[0])
            try:
                with autocast("cuda", dtype=amp_dtype):
                    embedding, _ = descriptor(dino_model(dino_resize(image)).pooler_output)
                    restored = model(image, embedding)
            finally:
                os.environ.pop("ONERESTORE_DEBUG_SAMPLE", None)
            restored = torch.clamp(restored, 0, 1)
            for name, _, metric in metrics:
                sums[name] += float(metric(restored).squeeze().item())
        local[set_name] = {"count": count, "total": len(dataset), **{f"{n}_sum": s for n, s in sums.items()}}
        line = ", ".join(f"{d}: {sums[n] / max(count, 1):.4f}" for n, d, _ in metrics)
        print(f"[GPU {rank}][{tb_prefix}] {set_name:20s} - {line} ({count}/{len(dataset)} imgs)")

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    model.set_runtime_resize_fallback(False)
    if rank != 0:
        return 0.0

    print(f"\n{'=' * 70}\n[{tb_prefix}] {label} Metrics (MUSIQ / CLIP-IQA)\n{'=' * 70}")
    averages = {name: 0.0 for name, _, _ in metrics}
    num_sets = 0
    for _, set_name in datasets:
        count = sum(r[set_name]["count"] for r in gathered if r and set_name in r)
        if count <= 0:
            continue
        means = {
            n: sum(r[set_name][f"{n}_sum"] for r in gathered if r and set_name in r) / count for n, _, _ in metrics
        }
        total = max(r[set_name]["total"] for r in gathered if r and set_name in r)
        print(
            f"{set_name:24s} " + ", ".join(f"{d}: {means[n]:.4f}" for n, d, _ in metrics) + f" ({count}/{total} imgs)"
        )
        for name, _, _ in metrics:
            averages[name] += means[name]
            if writer is not None:
                writer.add_scalar(f"{tb_prefix}_{set_name}/{name.upper()}", means[name], tb_step)
        num_sets += 1
    averages = {name: value / max(num_sets, 1) for name, value in averages.items()}
    print(f"{'-' * 70}\n{(tb_prefix + ' Avg'):24s} " + ", ".join(f"{d}: {averages[n]:.4f}" for n, d, _ in metrics))
    print(f"{'=' * 70}\n")
    if writer is not None:
        for name in averages:
            writer.add_scalar(f"{tb_prefix}/Avg_{name.upper()}", averages[name], tb_step)
    return averages["musiq"]


@torch.no_grad()
def evaluate_real_musiq(
    model, dino_model, descriptor, datasets, device, epoch, writer, rank, world_size, amp_dtype, timeout_seconds=3600
):
    """MUSIQ of the restored images. Used by the supervised stage; stops early after ``timeout_seconds``."""
    if not datasets:
        return 0.0
    model.eval()
    dino_model.eval()
    descriptor.eval()
    musiq = pyiqa.create_metric("musiq", as_loss=False, device=device)
    start = time.monotonic()
    timed_out = False
    results = {}

    for dataset, set_name in datasets:
        if timeout_seconds > 0:
            flag = torch.tensor([int(time.monotonic() - start >= timeout_seconds)], device=device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if int(flag.item()) > 0:
                timed_out = True
                if rank == 0:
                    print(f"[RealEval] Timeout reached before set '{set_name}', skip remaining sets.")
                break
        loader, _ = _shard_loader(dataset, rank, world_size)
        stats = torch.zeros(2, device=device, dtype=torch.float64)  # MUSIQ sum, image count
        iterator = tqdm(loader, desc=f"[RealEval] {set_name:<18}", leave=False, ncols=110) if rank == 0 else loader
        for image, _ in iterator:
            image = image.to(device, non_blocking=True)
            with autocast("cuda", dtype=amp_dtype):
                embedding, _ = descriptor(dino_model(dino_resize(image)).pooler_output)
                restored = model(image, embedding)
            restored = torch.clamp(restored, 0, 1)
            stats[0] += musiq(restored).mean().double()
            stats[1] += float(restored.shape[0])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        results[set_name] = (stats[0] / stats[1]).item() if stats[1] > 0 else 0.0
        if rank == 0 and writer:
            writer.add_scalar(f"RealVal_{set_name}/MUSIQ", results[set_name], epoch)

    flag = torch.tensor([int(timed_out)], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if rank != 0:
        return 0.0
    for set_name, value in results.items():
        print(f"[RealEval] Epoch {epoch} {set_name:24s} MUSIQ: {value:.4f}")
    average = sum(results.values()) / len(results) if results else 0.0
    print(f"[RealEval] Epoch {epoch} Avg MUSIQ: {average:.4f}" + (" (stopped by timeout)" if bool(flag.item()) else ""))
    if writer:
        writer.add_scalar("RealVal/Avg_MUSIQ", average, epoch)
    return average
