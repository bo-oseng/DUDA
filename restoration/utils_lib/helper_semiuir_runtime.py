import os
from datetime import datetime

import utils_lib.helper_semiuir_updated_marker_gate as semiuir_updated_marker_gate


def list_tensorboard_resume_candidates(log_root, exp_name):
    if not log_root or not os.path.isdir(log_root):
        return []

    prefix = f"{exp_name}_"
    candidates = []
    for entry in os.listdir(log_root):
        if not (entry == exp_name or entry.startswith(prefix)):
            continue
        full_path = os.path.join(log_root, entry)
        if os.path.isdir(full_path):
            candidates.append(full_path)

    def _candidate_sort_key(path):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = float("-inf")
        return (mtime, path)

    candidates.sort(key=_candidate_sort_key, reverse=True)
    return candidates


def resolve_tensorboard_log_dir(
    log_root,
    exp_name,
    resume_path=None,
    explicit_log_dir=None,
    reuse_latest_resume_dir=False,
    return_mode=False,
):
    if explicit_log_dir:
        if return_mode:
            return explicit_log_dir, "explicit"
        return explicit_log_dir

    base_dir = os.path.join(log_root, exp_name)
    if resume_path and reuse_latest_resume_dir:
        candidates = list_tensorboard_resume_candidates(log_root, exp_name)
        if candidates:
            if return_mode:
                return candidates[0], "resume_auto"
            return candidates[0]
        if return_mode:
            return base_dir, "resume_default"
        return base_dir

    if resume_path:
        if return_mode:
            return base_dir, "resume_default"
        return base_dir

    if not os.path.exists(base_dir):
        if return_mode:
            return base_dir, "new"
        return base_dir

    timestamp = datetime.now().strftime("%H_%M_%S")
    candidate = f"{base_dir}_{timestamp}"
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{base_dir}_{timestamp}_{suffix}"
        suffix += 1

    if return_mode:
        return candidate, "timestamped"
    return candidate


def is_supported_image_name(name):
    lower = name.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))


def resolve_step_anchor(cfg, labeled_steps, unlabeled_steps, rank=0, log_prefix="[WRES-SemiUIR]"):
    requested = str(getattr(cfg.train, "step_anchor", "unlabeled")).lower()
    if requested not in {"unlabeled", "labeled", "max", "min"}:
        raise ValueError(f"Invalid train.step_anchor={requested}. Choose one of: unlabeled, labeled, max, min.")

    auto_prefer_labeled = bool(getattr(cfg.train, "semiuir_prefer_labeled_anchor", True))
    effective = requested
    if auto_prefer_labeled and requested == "unlabeled" and unlabeled_steps < labeled_steps:
        effective = "labeled"
        if rank == 0:
            print(
                f"{log_prefix} Auto-switch step_anchor: unlabeled -> labeled "
                "to cycle the smaller image-based unlabeled set within each epoch."
            )
    return effective


def compute_steps_per_epoch(cfg, labeled_loader, unlabeled_loader, rank=0, log_prefix="[WRES-SemiUIR]"):
    labeled_steps = len(labeled_loader)
    unlabeled_steps = len(unlabeled_loader)
    step_anchor = resolve_step_anchor(cfg, labeled_steps, unlabeled_steps, rank=rank, log_prefix=log_prefix)
    if step_anchor == "unlabeled":
        steps_per_epoch = unlabeled_steps
    elif step_anchor == "labeled":
        steps_per_epoch = labeled_steps
    elif step_anchor == "max":
        steps_per_epoch = max(labeled_steps, unlabeled_steps)
    elif step_anchor == "min":
        steps_per_epoch = min(labeled_steps, unlabeled_steps)
    else:
        raise ValueError(f"Invalid train.step_anchor={step_anchor}. Choose one of: unlabeled, labeled, max, min.")
    return max(1, int(steps_per_epoch)), step_anchor, labeled_steps, unlabeled_steps


def resolve_pseudo_bank_ops(
    cfg,
    local_ops,
    rank=0,
    log_prefix="[WRES-SemiUIR]",
    updated_marker_message_suffix="",
    legacy_message_suffix="",
    updated_marker_ops=None,
):
    use_updated_marker_gate = bool(getattr(cfg.train, "use_updated_marker_gate", False))
    if updated_marker_ops is None:
        updated_marker_ops = {
            "initialize_pseudo_labels": semiuir_updated_marker_gate.initialize_pseudo_labels_semiuir,
            "copy_pseudo_labels": semiuir_updated_marker_gate.copy_pseudo_labels_semiuir,
            "initialize_zero_pseudo_labels": semiuir_updated_marker_gate.initialize_zero_pseudo_labels_semiuir,
            "get_reliable": semiuir_updated_marker_gate.get_reliable_semiuir,
        }

    if use_updated_marker_gate:
        if rank == 0:
            print(
                f"{log_prefix} Pseudo bank gate: updated-marker mode "
                f"(.updated required before unlabeled pseudo participates){updated_marker_message_suffix}."
            )
        return updated_marker_ops

    if rank == 0:
        print(
            f"{log_prefix} Pseudo bank gate: legacy file-exists / zero-placeholder mode"
            f"{legacy_message_suffix}."
        )
    return local_ops


def resolve_cgcd_anchor_params(cfg, cfg_getter):
    return {
        "clear_weight": float(cfg_getter(cfg, "cgcd_anchor_clear_weight", 1.0)),
        "neg_weight": float(cfg_getter(cfg, "cgcd_anchor_neg_weight", 1.0)),
        "softmin_tau": float(cfg_getter(cfg, "cgcd_anchor_softmin_tau", 0.0)),
    }


def build_real_eval_full_datasets(cfg, create_real_eval_datasets_fn, rank=0, log_prefix="[WRES-SemiUIR]"):
    full_root = str(getattr(cfg.train, "real_eval_root", "") or "").strip()
    full_sets = list(getattr(cfg.train, "real_eval_sets", []) or [])
    datasets = create_real_eval_datasets_fn(full_root, full_sets)

    additional_root = str(getattr(cfg.train, "real_eval_additional_root", "") or "").strip()
    additional_sets = list(getattr(cfg.train, "real_eval_additional_sets", []) or [])
    if additional_root and additional_sets:
        datasets.extend(create_real_eval_datasets_fn(additional_root, additional_sets))
        if rank == 0:
            print(
                f"{log_prefix} Full real eval additional source: root={additional_root}, "
                f"sets={additional_sets}"
            )

    return datasets


def build_real_eval_step_datasets(cfg, create_real_eval_datasets_fn, rank=0, log_prefix="[WRES-SemiUIR]"):
    use_real_eval = bool(getattr(cfg.train, "use_real_eval", False))
    if not use_real_eval:
        return []

    full_root = str(getattr(cfg.train, "real_eval_root", "") or "").strip()
    full_sets = list(getattr(cfg.train, "real_eval_sets", []))
    step_root = str(getattr(cfg.train, "real_eval_step_root", full_root) or "").strip()
    step_sets = list(getattr(cfg.train, "real_eval_step_sets", full_sets) or full_sets)

    if not step_root or len(step_sets) == 0:
        return []

    datasets = create_real_eval_datasets_fn(step_root, step_sets)
    if rank == 0:
        root_kind = "small-step" if os.path.abspath(step_root) != os.path.abspath(full_root) else "shared"
        print(
            f"{log_prefix} Step real eval source ({root_kind}): root={step_root}, "
            f"sets={step_sets}, count={len(datasets)}"
        )
    return datasets
