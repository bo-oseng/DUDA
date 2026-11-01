"""Supervised initialization of DUDA on paired data of the known degradations.

The restoration model and the descriptor (built from the base VB-CGCD clusters) are trained
with an L1 loss on random 224x224 crops. The resulting checkpoint initializes the student and
the teacher of the semi-supervised adaptation (train_semi_supervised.py).

Usage (from restoration/):
    torchrun --nproc_per_node=3 train_supervised.py --config <config> --exp_name <name>
"""

import argparse
import datetime
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from tqdm import tqdm

from utils_lib import checkpoint as ckpt
from utils_lib.common import (
    amp_dtype_from_cfg,
    batch_psnr,
    count_params,
    enable_tf32,
    load_config,
    seed_everything,
    setup_ddp,
)
from utils_lib.datasets import (
    ClassBalancedDistributedSampler,
    build_paired_dataset,
    build_real_eval_datasets,
    cap_per_class,
    dataset_labels,
    load_degradation_classes,
)
from utils_lib.descriptor import DiscoveryInstanceDescriptor
from utils_lib.dino import load_degradation_encoder
from utils_lib.onerestore import OneRestore
from utils_lib.real_eval import evaluate_real_musiq

# Training crops are already 224x224, so DINO only needs ImageNet normalization.
dino_normalize = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def train_one_epoch(
    model, dino, descriptor, loader, optimizer, scaler, device, epoch, cfg, writer, global_step, rank, amp_dtype
):
    model.train()
    descriptor.train()
    dino.eval()
    criterion_l1 = nn.L1Loss()
    criterion_ce = nn.CrossEntropyLoss()
    transfer_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.train.epochs}") if rank == 0 else loader
    sums = {"loss": 0.0, "l1": 0.0, "ce": 0.0, "psnr": 0.0}
    for i, (gt, lq, label) in enumerate(pbar):
        with torch.cuda.stream(transfer_stream):
            gt = gt.to(device, non_blocking=True)
            lq = lq.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
        compute_stream.wait_stream(transfer_stream)

        with torch.no_grad():
            z = dino(dino_normalize(lq)).pooler_output
        with autocast("cuda", dtype=amp_dtype):
            embedding, log_likelihood = descriptor(z)
            restored = model(lq, embedding)
            loss_l1 = criterion_l1(restored, gt)
            with torch.no_grad():  # classification is monitored only
                loss_ce = criterion_ce(log_likelihood, label)
            loss = cfg.train.l1_weight * loss_l1

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(descriptor.parameters(), max_norm=1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        with torch.no_grad():
            psnr = batch_psnr(gt, restored).mean()
        values = {"loss": loss.item(), "l1": loss_l1.item(), "ce": loss_ce.item(), "psnr": psnr.item()}
        for key, value in values.items():
            sums[key] += value
        if rank == 0 and (global_step + 1) % 10 == 0:
            pbar.set_description(
                f"Epoch {epoch}/{cfg.train.epochs} - loss: {sums['loss']/(i+1):.4f}, "
                f"ce: {sums['ce']/(i+1):.4f}, psnr: {sums['psnr']/(i+1):.2f}"
            )
        if rank == 0 and (i + 1) % cfg.train.print_freq == 0:
            writer.add_scalar("Train/Loss", values["loss"], global_step)
            writer.add_scalar("Train/L1_Loss", values["l1"], global_step)
            writer.add_scalar("Train/CE_Loss", values["ce"], global_step)
            writer.add_scalar("Train/PSNR", values["psnr"], global_step)
        global_step += 1

    n = len(loader)
    if rank == 0:
        print(f"\n[Epoch {epoch}] Loss: {sums['loss']/n:.4f}, CE: {sums['ce']/n:.4f}, PSNR: {sums['psnr']/n:.2f} dB")
    return global_step, sums["loss"] / n


def train(cfg):
    rank, world_size, local_rank = setup_ddp(timeout=datetime.timedelta(seconds=3600))
    device = torch.device(f"cuda:{local_rank}")
    enable_tf32()
    use_bf16, amp_dtype = amp_dtype_from_cfg(cfg)
    seed_everything(cfg.train.seed + rank)
    save_dir = os.path.join(cfg.train.save_dir, cfg.exp_name)
    writer = None
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(cfg.train.log_dir, cfg.exp_name))
        print(f"World size: {world_size}, AMP dtype: {'BF16' if use_bf16 else 'FP16'}")

    classes = load_degradation_classes(cfg)
    dataset = build_paired_dataset(classes, cfg, patch_size=224)
    labels = dataset_labels(dataset)
    dataset, labels = cap_per_class(dataset, labels, cfg.train.max_samples_per_class, cfg.train.seed)
    real_eval_datasets = build_real_eval_datasets(cfg.train.real_eval_root, cfg.train.real_eval_sets)
    sampler = ClassBalancedDistributedSampler(labels, world_size, rank, seed=cfg.train.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    if rank == 0:
        print(f"Train batches per GPU: {len(loader)}")

    dino = load_degradation_encoder(
        cfg.discovery.dino_checkpoint, cfg.discovery.dino_num_classes, cfg.discovery.dino_model, device
    )
    descriptor = DiscoveryInstanceDescriptor(
        cluster_dir=cfg.discovery.cluster_dir,
        output_dim=cfg.discovery.descriptor_dim,
        stage=0,
        pca_path=cfg.discovery.pca_path,
    ).to(device)
    model = OneRestore(channel=cfg.model.width).to(device)
    model.set_runtime_resize_fallback(True)  # large real-eval images
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    descriptor = DDP(descriptor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if rank == 0:
        print(f"Trainable parameters: OneRestore {count_params(model):,}, descriptor {count_params(descriptor):,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(descriptor.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    scaler = None if use_bf16 else GradScaler("cuda")

    start_epoch = 1
    if cfg.train.resume:
        start_epoch = ckpt.resume_supervised(cfg.train.resume, model.module, descriptor.module, optimizer, scheduler)

    def real_eval(epoch):
        evaluate_real_musiq(
            model.module,
            dino,
            descriptor,
            real_eval_datasets,
            device,
            epoch,
            writer,
            rank,
            world_size,
            amp_dtype,
            cfg.train.real_eval_timeout_seconds,
        )

    real_eval(0)  # IQA before training
    global_step = 0
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        sampler.set_epoch(epoch)
        if rank == 0:
            print(
                f"\n{'=' * 50}\nEpoch {epoch}/{cfg.train.epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}\n{'=' * 50}"
            )
        global_step, train_loss = train_one_epoch(
            model, dino, descriptor, loader, optimizer, scaler, device, epoch, cfg, writer, global_step, rank, amp_dtype
        )
        scheduler.step()
        if rank == 0:
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Train/Loss", train_loss, epoch)
        if epoch % cfg.train.real_eval_freq == 0:
            real_eval(epoch)
        if rank == 0:
            if epoch % cfg.train.save_freq == 0 or epoch == cfg.train.epochs:
                ckpt.save_supervised(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
                    model.module,
                    descriptor.module,
                    optimizer,
                    scheduler,
                    epoch,
                )
            ckpt.save_supervised(
                os.path.join(save_dir, "checkpoint_latest.pth"),
                model.module,
                descriptor.module,
                optimizer,
                scheduler,
                epoch,
            )
        dist.barrier()

    if rank == 0:
        print("\nTraining completed!")
        writer.close()
    dist.destroy_process_group()


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.exp_name = args.exp_name
    train(cfg)


if __name__ == "__main__":
    main()
