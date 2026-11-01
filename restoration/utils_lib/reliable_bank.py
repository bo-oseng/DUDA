"""Reliable bank of pseudo-targets for unpaired data (Sec. 3.1 and 3.3, Alg. 1 in the appendix).

The bank is a directory of images, one per unpaired patch, initialized with the degraded inputs.
An entry is replaced by the teacher prediction when

    1. MUSIQ(teacher) > max(MUSIQ(student), MUSIQ(bank)), or otherwise
    2. S_DA(teacher)  > max(S_DA(student),  S_DA(bank)).

An entry takes part in the consistency loss only after it has been updated at least once,
which is recorded by a ``<entry>.updated`` marker file. Until then the student prediction
(clamped to [0, 1]) is used as its target.
"""

import os

import torch
from torchvision.utils import save_image

from utils_lib.discovery_adaptive_score import discovery_adaptive_score

UPDATED_MARKER_SUFFIX = ".updated"


def marker_path(bank_path):
    return f"{bank_path}{UPDATED_MARKER_SUFFIX}"


def save_image_tensor(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor.clamp(0, 1), path)


def write_entry(image, bank_path):
    save_image_tensor(image, bank_path)
    with open(marker_path(bank_path), "w", encoding="ascii") as f:
        f.write("updated\n")


def updated_mask(bank_paths, device):
    states = [os.path.exists(p) and os.path.exists(marker_path(p)) for p in bank_paths]
    return torch.tensor(states, device=device, dtype=torch.bool)


@torch.no_grad()
def musiq_scores(metric, images):
    scores = torch.zeros(images.shape[0], device=images.device)
    for idx in range(images.shape[0]):
        scores[idx] = metric(images[idx : idx + 1])
    return scores


@torch.no_grad()
def _accept(targets, teacher, bank_paths, accept_mask):
    accepted = torch.nonzero(accept_mask, as_tuple=False).view(-1).tolist()
    for idx in accepted:
        targets[idx] = teacher[idx]
        write_entry(teacher[idx], bank_paths[idx])
    return len(accepted)


@torch.no_grad()
def update_reliable_bank(
    musiq_metric, teacher, student, bank, bank_paths, dino_model, descriptor, dino_transform, clear_idx, amp_dtype
):
    """Update the bank entries of one batch and return (pseudo-targets, number of updated entries).

    ``teacher``/``student`` are predictions for the weak/strong views, ``bank`` the current entries.
    """
    score_bank_musiq = musiq_scores(musiq_metric, bank)
    teacher, student, bank = teacher.clamp(0.0, 1.0), student.clamp(0.0, 1.0), bank.clamp(0.0, 1.0)

    active = updated_mask(bank_paths, teacher.device)
    targets = bank.clone()
    if (~active).any():
        targets[~active] = student.detach()[~active]

    # 1. MUSIQ. Entries that were never updated are compared with the student only.
    score_teacher = musiq_scores(musiq_metric, teacher)
    score_student = musiq_scores(musiq_metric, student)
    reference = torch.where(active, score_bank_musiq.detach().float().view(-1), score_student)
    accept_musiq = score_teacher > torch.maximum(score_student, reference)
    num_updated = _accept(targets, teacher, bank_paths, accept_musiq)

    # 2. Discovery-adaptive score for the candidates rejected by MUSIQ.
    fallback = ~accept_musiq
    if bool(fallback.any().item()):

        def s_da(images):
            return discovery_adaptive_score(images, dino_model, descriptor, dino_transform, clear_idx, amp_dtype)

        score_teacher, score_student, score_bank = s_da(teacher), s_da(student), s_da(bank)
        reference = torch.where(active, score_bank, score_student)
        accept_sda = fallback & (score_teacher > torch.maximum(score_student, reference))
        num_updated += _accept(targets, teacher, bank_paths, accept_sda)

    return targets, num_updated
