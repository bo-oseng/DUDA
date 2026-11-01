"""Mean-teacher components: EMA teacher update and the consistency-weight ramp-up."""

import numpy as np
import torch


@torch.no_grad()
def update_teacher(teacher, student, alpha=0.996, global_step=0):
    """theta_T <- alpha * theta_T + (1 - alpha) * theta_S (Eq. 3), with a warm-up on alpha."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    teacher = teacher.module if hasattr(teacher, "module") else teacher
    student = student.module if hasattr(student, "module") else student
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.mul_(alpha).add_(s_param, alpha=1 - alpha)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def consistency_weight(epoch, consistency=0.2, rampup_length=100.0):
    """Epoch-dependent weight omega_u of the consistency loss (Eq. 9)."""
    return consistency * sigmoid_rampup(epoch, rampup_length)
