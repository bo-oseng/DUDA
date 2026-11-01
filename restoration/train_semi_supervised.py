"""Semi-supervised adaptation of DUDA (Sec. 3).

The student and teacher start from the supervised model. Each step uses a batch of paired
synthetic data and a batch of unpaired real patches:

    L_total = L_sup + omega_u * L_con
    L_sup   = L_struct(y_hat, y) + lambda_per * L_per(y_hat, y) + lambda_ce * L_ce(l(z), c_gt)
    L_con   = L_1(y_hat_u, y^B)

where y^B is the pseudo-target stored in the reliable bank, updated by MUSIQ and the
discovery-adaptive score. The teacher is the EMA of the student.

Usage (from restoration/):
    torchrun --nproc_per_node=3 train_semi_supervised.py --config <config> --exp_name <name>
"""

import argparse
import os
import warnings
from datetime import datetime

import pyiqa
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.models import vgg16
from torchvision.utils import make_grid
from tqdm import tqdm

from utils_lib import checkpoint as ckpt
from utils_lib.common import (
    amp_dtype_from_cfg,
    batch_psnr,
    configure_torch_extensions_dir,
    enable_tf32,
    load_config,
    seed_everything,
    setup_ddp,
)
from utils_lib.datasets import (
    build_paired_dataset,
    build_real_eval_datasets,
    build_unpaired_dataset,
    collect_images,
    load_degradation_classes,
)
from utils_lib.descriptor import DiscoveryInstanceDescriptor
from utils_lib.dino import load_degradation_encoder
from utils_lib.losses import StructureLoss, VGGPerceptualLoss
from utils_lib.mean_teacher import consistency_weight, freeze, update_teacher
from utils_lib.onerestore import OneRestore
from utils_lib.real_eval import dino_resize, evaluate_real_iqa
from utils_lib.reliable_bank import update_reliable_bank

warnings.filterwarnings("ignore")
configure_torch_extensions_dir()


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------


def build_loaders(cfg, classes, rank, world_size):
    paired = build_paired_dataset(classes, cfg, cfg.train.patch_size)
    unpaired = build_unpaired_dataset(
        classes, cfg.train.unpaired_patches_dir, cfg.train.reliable_bank_dir, cfg.train.patch_size
    )
    kwargs = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=cfg.train.prefetch_factor,
        persistent_workers=True,
    )
    paired_sampler = DistributedSampler(paired, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed)
    unpaired_sampler = DistributedSampler(
        unpaired, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed
    )
    paired_loader = DataLoader(paired, sampler=paired_sampler, **kwargs)
    unpaired_loader = DataLoader(unpaired, sampler=unpaired_sampler, **kwargs)
    return paired_loader, unpaired_loader, paired_sampler, unpaired_sampler


def build_models(cfg, device):
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
    student = OneRestore(channel=cfg.model.width).to(device)
    teacher = OneRestore(channel=cfg.model.width).to(device)
    freeze(teacher)
    return dino, descriptor, student, teacher


_BANK_LOG_SAMPLES = {}


@torch.no_grad()
def log_bank_samples(writer, cfg, classes, global_step):
    """Write 20 fixed reliable-bank entries per novel class to TensorBoard."""
    cache = _BANK_LOG_SAMPLES
    for name in classes.novel:
        bank_dir = os.path.join(cfg.train.reliable_bank_dir, name)
        if name not in cache:
            files = [os.path.basename(p) for p in collect_images(os.path.join(cfg.train.unpaired_patches_dir, name))]
            if not os.path.isdir(bank_dir) or not files:
                continue
            n = min(20, len(files))
            step = (len(files) - 1) / float(n - 1) if n > 1 else 0.0
            cache[name] = [files[int(round(i * step))] for i in range(n)]
        images = []
        for file_name in cache[name]:
            path = os.path.join(bank_dir, file_name)
            try:
                images.append(read_image(path).float() / 255.0)
            except (OSError, RuntimeError, ValueError):
                images.append(torch.zeros((3, 224, 224), dtype=torch.float32))
        writer.add_image(
            f"PseudoLabel/{name}", make_grid(images, nrow=max(1, len(images) // 4), padding=2), global_step
        )


def next_batch(iterator, loader, cycle, name):
    try:
        return next(iterator), iterator
    except StopIteration:
        if not cycle:
            raise RuntimeError(f"{name} exhausted before the end of the epoch.")
        iterator = iter(loader)
        return next(iterator), iterator


def to_device(paired_batch, unpaired_batch, device, stream):
    with torch.cuda.stream(stream):
        gt, lq, label = (t.to(device, non_blocking=True) for t in paired_batch)
        weak, strong, bank = (t.to(device, non_blocking=True) for t in unpaired_batch[:3])
    return gt, lq, label, weak, strong, bank, unpaired_batch[3]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(ctx, epoch, global_step, skip_steps=0):
    cfg, rank, device = ctx["cfg"], ctx["rank"], ctx["device"]
    student, teacher, dino, descriptor = ctx["student"], ctx["teacher"], ctx["dino"], ctx["descriptor"]
    optimizer, scaler, writer = ctx["optimizer"], ctx["scaler"], ctx["writer"]
    paired_loader, unpaired_loader = ctx["paired_loader"], ctx["unpaired_loader"]

    teacher.eval()
    freeze(teacher)
    student.train()
    descriptor.train()
    criterion_l1 = nn.L1Loss()
    criterion_ce = nn.CrossEntropyLoss()
    transfer_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    # 016 uses the unpaired loader to define an epoch; cycle the paired loader.
    num_steps = len(unpaired_loader)
    cycle_paired, cycle_unpaired = len(paired_loader) < num_steps, False
    paired_iter, unpaired_iter = iter(paired_loader), iter(unpaired_loader)
    steps = range(skip_steps, num_steps)
    pbar = tqdm(steps, desc=f"Epoch {epoch}", total=num_steps, initial=skip_steps, dynamic_ncols=True, mininterval=0.5)
    pbar = pbar if rank == 0 else steps

    omega_u = consistency_weight(epoch, cfg.train.consistency_weight, cfg.train.rampup_epoch)
    for _ in range(skip_steps):  # resume in the middle of an epoch
        _, paired_iter = next_batch(paired_iter, paired_loader, cycle_paired, "paired loader")
        _, unpaired_iter = next_batch(unpaired_iter, unpaired_loader, cycle_unpaired, "unpaired loader")
    paired_batch, paired_iter = next_batch(paired_iter, paired_loader, cycle_paired, "paired loader")
    unpaired_batch, unpaired_iter = next_batch(unpaired_iter, unpaired_loader, cycle_unpaired, "unpaired loader")
    batch = to_device(paired_batch, unpaired_batch, device, transfer_stream)

    sums = {"sup": 0.0, "con": 0.0, "con_weighted": 0.0, "total": 0.0, "psnr": 0.0}
    num_updated = steps_run = 0
    for i in pbar:
        compute_stream.wait_stream(transfer_stream)
        gt, lq, label, weak, strong, bank, bank_paths = batch

        with torch.no_grad():
            z = dino(dino_resize(torch.cat([lq, weak], dim=0))).pooler_output
            z_paired, z_unpaired = z[: lq.shape[0]], z[lq.shape[0] :]

        with autocast("cuda", dtype=ctx["amp_dtype"]):
            # Supervised loss on paired data.
            v_paired, log_likelihood = descriptor(z_paired)
            restored = student(lq, v_paired)
            loss_sup = (
                ctx["loss_struct"](restored, gt)
                + cfg.train.perceptual_weight * ctx["loss_per"](restored, gt)
                + cfg.train.ce_weight * criterion_ce(log_likelihood, label)
            )

            # Consistency loss on unpaired data against the reliable bank.
            v_unpaired, _ = descriptor(z_unpaired)
            student_out = student(strong, v_unpaired)
            with torch.no_grad():
                teacher_out = teacher(weak, v_unpaired)
                targets, updated = update_reliable_bank(
                    ctx["musiq"],
                    teacher_out,
                    student_out.detach(),
                    bank,
                    bank_paths,
                    dino,
                    descriptor.module,
                    dino_resize,
                    ctx["clear_idx"],
                    ctx["amp_dtype"],
                )
                num_updated += updated
            with autocast("cuda", enabled=False):
                loss_con = criterion_l1(student_out.float(), targets.float())
            loss_con_weighted = omega_u * loss_con
            loss_total = loss_sup + loss_con_weighted

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
        else:
            loss_total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(descriptor.parameters(), max_norm=1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        with torch.no_grad():
            update_teacher(teacher, student, alpha=cfg.train.ema_alpha, global_step=global_step)
            psnr = batch_psnr(gt, restored).mean()

        values = {
            "sup": loss_sup.detach().item(),
            "con": loss_con.detach().item(),
            "con_weighted": loss_con_weighted.detach().item(),
            "total": loss_total.detach().item(),
            "psnr": psnr.detach().item(),
        }
        for key, value in values.items():
            sums[key] += value
        if rank == 0 and (i + 1) % 10 == 0:
            n = steps_run + 1
            pbar.set_description(
                f"Epoch {epoch}/{cfg.train.epochs} | Sup: {sums['sup']/n:.4f}, Con: {sums['con']/n:.4f}, "
                f"ConW: {sums['con_weighted']/n:.4f}, PSNR: {sums['psnr']/n:.2f}, updated: {num_updated}"
            )
        if rank == 0:
            for key, value in values.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            writer.add_scalar("train/pseudo_update_cnt", num_updated, global_step)
            writer.add_scalar("train/rampup_weight", omega_u, global_step)

        if i + 1 < num_steps:
            paired_batch, paired_iter = next_batch(paired_iter, paired_loader, cycle_paired, "paired loader")
            unpaired_batch, unpaired_iter = next_batch(
                unpaired_iter, unpaired_loader, cycle_unpaired, "unpaired loader"
            )
            batch = to_device(paired_batch, unpaired_batch, device, transfer_stream)
        global_step += 1

        if rank == 0 and global_step % cfg.train.save_freq_steps == 0:
            state = ctx["state"](epoch, global_step)
            torch.save(state, os.path.join(ctx["checkpoint_dir"], f"stage{cfg.incremental.stage}_step_{global_step}.pth"))
            torch.save(state, os.path.join(ctx["checkpoint_dir"], f"stage{cfg.incremental.stage}_latest.pth"))
            print(f"Saved checkpoint: stage{cfg.incremental.stage}_step_{global_step}.pth")
        if rank == 0 and global_step % cfg.train.bank_log_freq_steps == 0:
            log_bank_samples(writer, cfg, ctx["classes"], global_step)
            writer.flush()
        if global_step % cfg.train.real_eval_freq_steps == 0:
            evaluate_real_iqa(
                teacher,
                dino,
                descriptor.module,
                ctx["real_eval_step"],
                device,
                writer,
                rank,
                ctx["world_size"],
                ctx["amp_dtype"],
                global_step,
                f"Step {global_step}",
                "RealEvalStep",
            )
            dist.barrier()
        steps_run += 1

    if rank == 0:
        n = max(1, steps_run)
        print(
            f"\n[Epoch {epoch}] Total: {sums['total']/n:.4f}, Sup: {sums['sup']/n:.4f}, Con: {sums['con']/n:.4f}, "
            f"ConW: {sums['con_weighted']/n:.4f}, PSNR: {sums['psnr']/n:.2f} dB, bank updates: {num_updated}"
        )
    return global_step, sums["total"] / max(1, steps_run), num_updated


def train(cfg, resume=None):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    enable_tf32()
    seed_everything(cfg.train.seed + rank)
    checkpoint_dir = os.path.join(cfg.train.save_dir, cfg.exp_name)
    use_bf16, amp_dtype = amp_dtype_from_cfg(cfg)
    writer = None
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.train.reliable_bank_dir, exist_ok=True)
        log_dir = os.path.join(cfg.train.log_dir, cfg.exp_name)
        if not resume:
            log_dir = f"{log_dir}_{datetime.now().strftime('%H_%M_%S')}"
        writer = SummaryWriter(log_dir=log_dir)
        print(f"World size: {world_size}, AMP dtype: {'BF16' if use_bf16 else 'FP16'}, TensorBoard: {log_dir}")

    classes = load_degradation_classes(cfg)
    paired_loader, unpaired_loader, paired_sampler, unpaired_sampler = build_loaders(cfg, classes, rank, world_size)
    real_eval_full = build_real_eval_datasets(cfg.train.real_eval_root, cfg.train.real_eval_sets)
    real_eval_step = build_real_eval_datasets(cfg.train.real_eval_step_root, cfg.train.real_eval_sets)
    steps_per_epoch = len(unpaired_loader)
    if rank == 0:
        print(f"Steps per epoch: {steps_per_epoch} (paired {len(paired_loader)}, unpaired {len(unpaired_loader)})")

    # Rank 0 builds first so that the BOFT extension is compiled once.
    if rank == 0:
        dino, descriptor, student, teacher = build_models(cfg, device)
    dist.barrier()
    if rank != 0:
        dino, descriptor, student, teacher = build_models(cfg, device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(descriptor.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    scaler = None if use_bf16 else GradScaler("cuda")
    musiq = pyiqa.create_metric("musiq", as_loss=True).cuda()
    vgg = vgg16(pretrained=True).features[:16].to(device)
    freeze(vgg)
    loss_struct = StructureLoss().to(device)
    loss_per = VGGPerceptualLoss(vgg).to(device)

    start_epoch, global_step, skip_steps = 1, 0, 0
    if resume:
        saved_epoch, global_step = ckpt.resume_semi_supervised(
            resume, student, teacher, descriptor, optimizer, scheduler, device
        )
        steps_in_epoch = global_step - (saved_epoch - 1) * steps_per_epoch
        if 0 < steps_in_epoch < steps_per_epoch:  # step checkpoint inside an epoch
            start_epoch, skip_steps = saved_epoch, steps_in_epoch
        else:
            start_epoch = saved_epoch + 1
        if rank == 0:
            print(f"Resume at epoch {start_epoch}, skipping {skip_steps} steps")
    else:
        previous = getattr(cfg.train, "previous_stage_checkpoint", None)
        if previous:
            ckpt.warm_start_from_previous(previous, student, teacher, descriptor)
        else:
            ckpt.warm_start_from_supervised(cfg.train.stage0_checkpoint, student, teacher, descriptor)

    student = DDP(student, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    descriptor = DDP(descriptor, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    def state(epoch, step):
        return ckpt.semi_supervised_state(
            epoch, step, student.module, teacher, descriptor.module, optimizer, scheduler, stage=cfg.incremental.stage
        )

    ctx = dict(
        cfg=cfg,
        rank=rank,
        world_size=world_size,
        device=device,
        amp_dtype=amp_dtype,
        writer=writer,
        classes=classes,
        clear_idx=classes.orig2classifier[0],
        student=student,
        teacher=teacher,
        dino=dino,
        descriptor=descriptor,
        optimizer=optimizer,
        scaler=scaler,
        musiq=musiq,
        loss_struct=loss_struct,
        loss_per=loss_per,
        paired_loader=paired_loader,
        unpaired_loader=unpaired_loader,
        real_eval_step=real_eval_step,
        checkpoint_dir=checkpoint_dir,
        state=state,
    )

    if not resume:  # IQA of the initial teacher
        evaluate_real_iqa(
            teacher,
            dino,
            descriptor.module,
            real_eval_full,
            device,
            writer,
            rank,
            world_size,
            amp_dtype,
            0,
            "Init Step 0",
            "RealEvalFull",
        )
        dist.barrier()

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        paired_sampler.set_epoch(epoch)
        unpaired_sampler.set_epoch(epoch)
        if rank == 0:
            print(
                f"\n{'=' * 60}\nEpoch {epoch}/{cfg.train.epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}\n{'=' * 60}"
            )
        global_step, train_loss, num_updated = train_one_epoch(ctx, epoch, global_step, skip_steps)
        skip_steps = 0
        scheduler.step()

        evaluate_real_iqa(
            teacher,
            dino,
            descriptor.module,
            real_eval_full,
            device,
            writer,
            rank,
            world_size,
            amp_dtype,
            global_step,
            f"Epoch {epoch} / Step {global_step}",
            "RealEvalFull",
        )
        dist.barrier()

        if rank == 0:
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Train/train_loss", train_loss, epoch)
            writer.add_scalar("Train/pseudo_update_cnt_epoch", num_updated, epoch)
            state_now = state(epoch, global_step)
            for name in (f"stage{cfg.incremental.stage}_epoch_{epoch}.pth", f"stage{cfg.incremental.stage}_latest.pth"):
                torch.save(state_now, os.path.join(checkpoint_dir, name))
            final_step_path = os.path.join(checkpoint_dir, f"stage{cfg.incremental.stage}_step_{global_step}.pth")
            if epoch == cfg.train.epochs and not os.path.exists(final_step_path):
                torch.save(state_now, final_step_path)
            print(f"Saved checkpoint: stage{cfg.incremental.stage}_epoch_{epoch}.pth")
        dist.barrier()

    if rank == 0:
        print("\nTraining completed!")
        writer.close()
    dist.destroy_process_group()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--resume", default=None, help="Semi-supervised checkpoint to resume from")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.exp_name = args.exp_name
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
