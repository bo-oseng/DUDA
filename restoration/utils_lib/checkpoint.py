"""Checkpoint formats of the supervised and semi-supervised stages.

Supervised:      model_state_dict, cgcd_model_state_dict, optimizer/scheduler states.
Semi-supervised: student_state_dict, teacher_state_dict, cgcd_state_dict, optimizer/scheduler states.
``cgcd_*`` entries hold the descriptor.
"""

import torch


def save_supervised(path, model, descriptor, optimizer, scheduler, epoch):
    torch.save(
        {
            "epoch": epoch,
            "global_step": 0,
            "model_state_dict": model.state_dict(),
            "cgcd_model_state_dict": descriptor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": None,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def resume_supervised(path, model, descriptor, optimizer, scheduler):
    """Restore a supervised run and return the next epoch."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    descriptor.load_state_dict(checkpoint["cgcd_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"Resumed from {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"] + 1


def warm_start_descriptor(descriptor, state):
    """Load the learnable descriptor weights (E, projection, W_res) and keep the current-stage
    cluster statistics and frozen PCA/standardization of ``descriptor``."""
    current = descriptor.state_dict()
    keep = {name for name, _ in descriptor.named_buffers()}
    keep.update(name for name, param in descriptor.named_parameters() if not param.requires_grad)
    merged = {key: value.clone() for key, value in current.items()}
    loaded = preserved = 0
    for key, value in state.items():
        if key in keep:
            preserved += 1
        elif tuple(current[key].shape) == tuple(value.shape):
            merged[key] = value.to(device=current[key].device, dtype=current[key].dtype)
            loaded += 1
        else:
            raise ValueError(
                f"Descriptor shape mismatch for {key}: {tuple(value.shape)} vs {tuple(current[key].shape)}"
            )
    descriptor.load_state_dict(merged, strict=True)
    print(f"Warm-started the descriptor (loaded={loaded}, preserved={preserved})")


def warm_start_from_supervised(path, student, teacher, descriptor):
    """Initialize the student and teacher with the supervised model (Sec. 3.4)."""
    checkpoint = torch.load(path, map_location=next(teacher.parameters()).device)
    student.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    warm_start_descriptor(descriptor, checkpoint["cgcd_model_state_dict"])
    print(f"Initialized student and teacher from {path} (epoch {checkpoint.get('epoch', 0)})")


def warm_start_from_previous(path, student, teacher, descriptor):
    """Initialize a new continual stage from the previous student, teacher and descriptor."""
    checkpoint = torch.load(path, map_location=next(teacher.parameters()).device)
    student.load_state_dict(checkpoint["student_state_dict"], strict=True)
    teacher.load_state_dict(checkpoint["teacher_state_dict"], strict=True)
    warm_start_descriptor(descriptor, checkpoint["cgcd_state_dict"])
    print(f"Initialized next stage from {path}")


def semi_supervised_state(epoch, global_step, student, teacher, descriptor, optimizer, scheduler, stage=1):
    return {
        "epoch": epoch,
        "stage": stage,
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "cgcd_state_dict": descriptor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "best_psnr": 0.0,
        "is_best": False,
    }


def resume_semi_supervised(path, student, teacher, descriptor, optimizer, scheduler, device):
    """Restore a semi-supervised run and return (saved epoch, global step)."""
    checkpoint = torch.load(path, map_location=device)
    student.load_state_dict(checkpoint["student_state_dict"])
    teacher.load_state_dict(checkpoint["teacher_state_dict"])
    descriptor.load_state_dict(checkpoint["cgcd_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"Resumed from {path} (epoch {checkpoint['epoch']}, global_step {checkpoint['global_step']})")
    return checkpoint["epoch"], checkpoint["global_step"]


def load_for_inference(path, model, descriptor, use_student=False):
    """Load a supervised or semi-supervised checkpoint for evaluation (teacher by default)."""
    checkpoint = torch.load(path, map_location="cpu")
    if "teacher_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["student_state_dict" if use_student else "teacher_state_dict"], strict=True)
        descriptor.load_state_dict(checkpoint["cgcd_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        descriptor.load_state_dict(checkpoint["cgcd_model_state_dict"], strict=True)
    return int(checkpoint.get("epoch", 0))
