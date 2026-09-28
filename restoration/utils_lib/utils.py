import imageio
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import random
import torch


def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def saveImage(filename, image):
    imageTMP = np.clip(image * 255.0, 0, 255).astype("uint8")
    imageio.imwrite(filename, imageTMP)


def save_rgb(img, filename):

    img = np.clip(img, 0.0, 1.0)
    if np.max(img) <= 1:
        img = img * 255

    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def load_img(
    filename,
    norm=True,
):
    img = np.array(Image.open(filename).convert("RGB"))
    if norm:
        img = img / 255.0
        img = img.astype(np.float32)
    return img


def plot_all(images, figsize=(20, 10), axis="off", names=None):
    nplots = len(images)
    fig, axs = plt.subplots(1, nplots, figsize=figsize, dpi=80, constrained_layout=True)
    for i in range(nplots):
        axs[i].imshow(images[i])
        if names:
            axs[i].set_title(names[i])
        axs[i].axis(axis)
    plt.show()


def modcrop(img_in, scale=2):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r, :]
    else:
        raise ValueError("Wrong img ndim: [{:d}].".format(img.ndim))
    return img


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


########## MODEL


def count_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def save_checkpoint(
    model, cgcd_model, optimizer, scheduler, epoch, save_path, is_best=False, global_step=0, scaler=None
):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "cgcd_model_state_dict": cgcd_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "checkpoint_best.pth")
        torch.save(checkpoint, best_path)
        print(f"Best checkpoint saved to {best_path}")


def load_checkpoint(checkpoint_path, model, cgcd_model=None, optimizer=None, scheduler=None):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if cgcd_model and "cgcd_model_state_dict" in checkpoint:
        cgcd_model.load_state_dict(checkpoint["cgcd_model_state_dict"])
        print("Loaded cgcd_model state_dict")

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint.get("global_step", 0)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Resumed from epoch {checkpoint['epoch']}, global_step {global_step}")

    return start_epoch, global_step


def load_checkpoint_incremental(checkpoint_path, model, cgcd_model=None):
    """
    Load checkpoint for incremental training.
    Loads teacher_state_dict from previous stage to initialize both student and teacher.
    Extracts base weights from LoRA-wrapped model (strips 'base_model.model.' prefix and '.base_layer').

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into (student_base or teacher_base before LoRA injection)
        cgcd_model: CGCD model (optional)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load teacher weights from previous stage
    if "teacher_state_dict" in checkpoint:
        teacher_state = checkpoint["teacher_state_dict"]

        # Extract base weights from LoRA-wrapped model
        # LoRA wraps weights as: base_model.model.xxx.base_layer.weight
        # We need to extract to: xxx.weight
        base_state = {}
        for key, value in teacher_state.items():
            # Remove 'base_model.model.' prefix
            if key.startswith("base_model.model."):
                new_key = key.replace("base_model.model.", "")

                # Remove '.base_layer' from LoRA wrapped layers
                new_key = new_key.replace(".base_layer.", ".")

                # Skip LoRA-specific weights (lora_A, lora_B)
                if ".lora_A." not in new_key and ".lora_B." not in new_key:
                    base_state[new_key] = value

        # Load extracted base weights
        model.load_state_dict(base_state)
        print(f"✓ Loaded teacher base weights from previous stage ({len(base_state)} parameters)")
    else:
        raise KeyError(f"'teacher_state_dict' not found in checkpoint. Available keys: {list(checkpoint.keys())}")

    # Load CGCD weights
    if cgcd_model is not None:
        if "cgcd_state_dict" in checkpoint:
            load_cgcd_state_dict(
                cgcd_model,
                checkpoint["cgcd_state_dict"],
                state_name="cgcd_state_dict",
                preserve_current_stage_state=True,
            )
        else:
            print("Warning: cgcd_state_dict not found in checkpoint")

    epoch = checkpoint.get("epoch", 0)
    stage = checkpoint.get("stage", 0)
    print(f"✓ Loaded from stage {stage}, epoch {epoch}")

    return epoch, stage


def _resolve_state_module(module):
    return module.module if hasattr(module, "module") else module


def _cgcd_preserve_current_stage_keys(cgcd_model):
    cgcd_inner = _resolve_state_module(cgcd_model)
    preserve_keys = {name for name, _ in cgcd_inner.named_buffers()}
    preserve_keys.update(name for name, param in cgcd_inner.named_parameters() if not param.requires_grad)
    return preserve_keys


def _copy_tensor_prefix(dst_tensor, src_tensor):
    if not (torch.is_tensor(dst_tensor) and torch.is_tensor(src_tensor)):
        return None, 0
    if dst_tensor.ndim == 0 or src_tensor.ndim == 0 or dst_tensor.ndim != src_tensor.ndim:
        return None, 0
    if tuple(dst_tensor.shape[1:]) != tuple(src_tensor.shape[1:]):
        return None, 0

    overlap = min(int(dst_tensor.shape[0]), int(src_tensor.shape[0]))
    if overlap <= 0:
        return None, 0

    merged = dst_tensor.clone()
    merged[:overlap].copy_(src_tensor[:overlap].to(device=dst_tensor.device, dtype=dst_tensor.dtype))
    return merged, overlap


def load_cgcd_state_dict(cgcd_model, cgcd_state, state_name="cgcd_state_dict", preserve_current_stage_state=False):
    if cgcd_model is None or cgcd_state is None:
        return False

    cgcd_inner = _resolve_state_module(cgcd_model)

    if not preserve_current_stage_state:
        try:
            cgcd_inner.load_state_dict(cgcd_state, strict=True)
            print(f"✓ Loaded {state_name}")
        except RuntimeError as e:
            print(f"[WARNING] Strict CGCD load failed, retry with strict=False: {e}")
            incompat = cgcd_inner.load_state_dict(cgcd_state, strict=False)
            print(
                f"✓ Loaded {state_name} with strict=False "
                f"(missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)})"
            )
            if len(incompat.missing_keys) > 0:
                print(f"[DEBUG] CGCD missing keys sample: {incompat.missing_keys[:10]}")
            if len(incompat.unexpected_keys) > 0:
                print(f"[DEBUG] CGCD unexpected keys sample: {incompat.unexpected_keys[:10]}")
        return True

    current_state = cgcd_inner.state_dict()
    merged_state = {key: value.clone() for key, value in current_state.items()}
    preserve_keys = _cgcd_preserve_current_stage_keys(cgcd_inner)

    loaded_keys = 0
    preserved_keys_count = 0
    partial_keys = []
    skipped_shape_keys = []
    unexpected_keys = []

    for key, value in cgcd_state.items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue

        current_value = current_state[key]
        if key in preserve_keys:
            preserved_keys_count += 1
            continue

        if tuple(current_value.shape) == tuple(value.shape):
            merged_state[key] = value.to(device=current_value.device, dtype=current_value.dtype)
            loaded_keys += 1
            continue

        partial_value, overlap = _copy_tensor_prefix(current_value, value)
        if partial_value is not None:
            merged_state[key] = partial_value
            partial_keys.append((key, overlap, int(current_value.shape[0]), int(value.shape[0])))
            continue

        skipped_shape_keys.append((key, tuple(value.shape), tuple(current_value.shape)))

    cgcd_inner.load_state_dict(merged_state, strict=True)
    print(
        f"✓ Warm-started {state_name} while preserving current-stage CGCD stats "
        f"(loaded={loaded_keys}, preserved={preserved_keys_count}, "
        f"partial={len(partial_keys)}, unexpected={len(unexpected_keys)}, "
        f"shape_skipped={len(skipped_shape_keys)})"
    )
    if len(partial_keys) > 0:
        print(f"[DEBUG] CGCD partial warm-start keys sample: {partial_keys[:10]}")
    if len(unexpected_keys) > 0:
        print(f"[DEBUG] CGCD unexpected checkpoint keys sample: {unexpected_keys[:10]}")
    if len(skipped_shape_keys) > 0:
        print(f"[DEBUG] CGCD shape-skipped keys sample: {skipped_shape_keys[:10]}")
    return True


# def load_teacher_with_lora(checkpoint_path, lora_model, cgcd_model=None, preserve_cgcd_stage_state=True):
#     """
#     Load checkpoint for incremental training - loads full teacher model (base + LoRA).

#     This function should be called AFTER LoRA injection.
#     It loads the complete teacher state (base weights + LoRA weights) from the previous stage.

#     Handles both:
#     - Stage 0 checkpoints: saved with "model_state_dict" (no LoRA)
#     - Stage N checkpoints: saved with "teacher_state_dict" (with LoRA)

#     Args:
#         checkpoint_path: Path to checkpoint file
#         lora_model: Model with LoRA already injected (call this AFTER inject_lora_to_instructir)
#         cgcd_model: CGCD model (optional)

#     Returns:
#         epoch, stage from checkpoint
#     """
#     print(f"Loading checkpoint from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location="cpu")

#     print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")

#     # Load model weights - handle both Stage 0 and Stage N checkpoints
#     if "teacher_state_dict" in checkpoint:
#         # Stage N checkpoint (with LoRA)
#         teacher_state = checkpoint["teacher_state_dict"]
#         print(f"[DEBUG] Loading Stage N checkpoint with teacher_state_dict")
#         print(f"[DEBUG] Teacher state keys (first 5): {list(teacher_state.keys())[:5]}")
#         print(f"[DEBUG] LoRA model keys (first 5): {list(lora_model.state_dict().keys())[:5]}")

#         missing_keys, unexpected_keys = lora_model.load_state_dict(teacher_state, strict=False)
#         print(f"✓ Loaded teacher weights (base + LoRA) from Stage N checkpoint")
#         if missing_keys:
#             print(f"[WARNING] Missing keys: {missing_keys[:5]}... (showing first 5)")
#         if unexpected_keys:
#             print(f"[WARNING] Unexpected keys: {unexpected_keys[:5]}... (showing first 5)")

#     elif "model_state_dict" in checkpoint:
#         # Stage 0 checkpoint (no LoRA) - need to convert keys for LoRA-wrapped model
#         model_state = checkpoint["model_state_dict"]
#         print(f"[DEBUG] Loading Stage 0 checkpoint with model_state_dict")
#         print(f"[DEBUG] Model state keys (first 5): {list(model_state.keys())[:5]}")
#         print(f"[DEBUG] Model state total keys: {len(model_state.keys())}")

#         # Get LoRA model's current state to identify which layers have LoRA
#         lora_state = lora_model.state_dict()
#         print(f"[DEBUG] LoRA model type: {type(lora_model)}")
#         print(f"[DEBUG] LoRA model keys (first 5): {list(lora_state.keys())[:5]}")

#         # Convert Stage 0 keys to LoRA format
#         # Stage 0: "layer.weight" -> LoRA: "base_model.model.layer.base_layer.weight" (for LoRA layers)
#         # Stage 0: "layer.weight" -> LoRA: "base_model.model.layer.weight" (for non-LoRA layers)
#         converted_state = {}
#         conversion_log = []

#         for key, value in model_state.items():
#             # Check if this layer has LoRA in the target model
#             lora_key_with_base = f"base_model.model.{key.replace('.weight', '.base_layer.weight').replace('.bias', '.base_layer.bias')}"
#             lora_key_without_base = f"base_model.model.{key}"

#             if lora_key_with_base in lora_state:
#                 # This layer has LoRA, use .base_layer format
#                 new_key = lora_key_with_base
#                 converted_state[new_key] = value
#                 conversion_log.append(f"{key} -> {new_key}")
#             elif lora_key_without_base in lora_state:
#                 # This layer doesn't have LoRA, use direct format
#                 new_key = lora_key_without_base
#                 converted_state[new_key] = value
#                 conversion_log.append(f"{key} -> {new_key}")
#             else:
#                 # Keep original format as fallback
#                 converted_state[f"base_model.model.{key}"] = value
#                 conversion_log.append(f"{key} -> base_model.model.{key} (fallback)")

#         print(f"[DEBUG] Converted {len(converted_state)} keys from Stage 0 format to LoRA format")
#         print(f"[DEBUG] Sample conversions (first 3):")
#         for log in conversion_log[:3]:
#             print(f"  {log}")

#         # Load converted state dict
#         missing_keys, unexpected_keys = lora_model.load_state_dict(converted_state, strict=False)
#         print(f"✓ Loaded base model weights from Stage 0 checkpoint (converted to LoRA format)")

#         if missing_keys:
#             print(f"[WARNING] Missing keys ({len(missing_keys)} total)")
#             if len(missing_keys) <= 10:
#                 for key in missing_keys:
#                     print(f"  - {key}")
#             else:
#                 print(f"  First 5: {missing_keys[:5]}")
#         if unexpected_keys:
#             print(f"[WARNING] Unexpected keys ({len(unexpected_keys)} total)")
#             if len(unexpected_keys) <= 10:
#                 for key in unexpected_keys:
#                     print(f"  - {key}")
#             else:
#                 print(f"  First 5: {unexpected_keys[:5]}")
#     else:
#         raise KeyError(f"'teacher_state_dict' or 'model_state_dict' not found in checkpoint. Available keys: {list(checkpoint.keys())}")

#     # Load CGCD weights
#     if cgcd_model is not None:
#         if "cgcd_state_dict" in checkpoint:
#             print(f"[DEBUG] Loading cgcd_state_dict")
#             cgcd_model.load_state_dict(checkpoint["cgcd_state_dict"])
#             print("✓ Loaded cgcd_state_dict")
#         elif "cgcd_model_state_dict" in checkpoint:
#             print(f"[DEBUG] Loading cgcd_model_state_dict")
#             cgcd_model.load_state_dict(checkpoint["cgcd_model_state_dict"])
#             print("✓ Loaded cgcd_model_state_dict (Stage 0 format)")
#         else:
#             print("Warning: cgcd_state_dict not found in checkpoint")
#             print(f"[DEBUG] Available checkpoint keys: {list(checkpoint.keys())}")

#     epoch = checkpoint.get("epoch", 0)
#     stage = checkpoint.get("stage", 0)
#     print(f"✓ Loaded from stage {stage}, epoch {epoch}")

#     return epoch, stage


def load_teacher_with_lora(checkpoint_path, lora_model, cgcd_model=None, preserve_cgcd_stage_state=True):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "teacher_state_dict" in checkpoint:
        model_state = checkpoint["teacher_state_dict"]
    elif "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        raise KeyError("Checkpoint에 가중치 딕셔너리가 없습니다.")

    lora_state = lora_model.state_dict()
    converted_state = {}

    for key, value in model_state.items():
        # 1. 원래 이름 그대로 있는지 확인 (가장 일반적인 경우)
        if key in lora_state:
            converted_state[key] = value

        # 2. .weight -> .base_layer.weight 변환이 필요한지 확인 (LoRA 레이어인 경우)
        else:
            lora_key = key.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
            if lora_key in lora_state:
                converted_state[lora_key] = value
            else:
                # 3. 혹시나 base_model.model. 접두어가 실제로 필요한 경우를 대비
                alt_key = f"base_model.model.{key}"
                if alt_key in lora_state:
                    converted_state[alt_key] = value
                else:
                    # 매칭되는 걸 못 찾으면 일단 원래 키로 넣어둠 (strict=False에서 걸러짐)
                    converted_state[key] = value

    # 가중치 로드
    missing_keys, unexpected_keys = lora_model.load_state_dict(converted_state, strict=False)
    print(f"✓ Loaded weights. Matched {len(converted_state)} keys.")

    if unexpected_keys:
        print(f"[DEBUG] Unexpected keys (first 3): {unexpected_keys[:3]}")

    # CGCD 로딩 로직 (stage별 Gaussian 통계는 현재 모델 값을 유지)
    if cgcd_model is not None:
        cgcd_state = checkpoint.get("cgcd_state_dict", checkpoint.get("cgcd_model_state_dict"))
        if cgcd_state:
            load_cgcd_state_dict(
                cgcd_model,
                cgcd_state,
                state_name="CGCD state dict",
                preserve_current_stage_state=preserve_cgcd_stage_state,
            )

    return checkpoint.get("epoch", 0), checkpoint.get("stage", 0)


def load_teacher_wo_lora(
    checkpoint_path,
    student_model,
    teacher_model,
    cgcd_model=None,
    device=None,
    preserve_cgcd_stage_state=True,
):
    """
    Load checkpoint for incremental training WITHOUT LoRA (direct fine-tuning).

    This function handles loading from both Stage 0 and Stage N checkpoints.

    Handles both:
    - Stage 0 checkpoints: saved with "model_state_dict"
    - Stage N checkpoints: saved with "student_state_dict" and "teacher_state_dict"

    Args:
        checkpoint_path: Path to checkpoint file
        student_model: Student model (direct OneRestore, no LoRA wrapper), can be None for scratch mode
        teacher_model: Teacher model (direct OneRestore, no LoRA wrapper)
        cgcd_model: CGCD model (optional)
        device: Target device for model parameters (default: infer from teacher_model or student_model)
        preserve_cgcd_stage_state: Keep the current model's stage-specific CGCD stats and
            frozen preprocess state while warm-starting learnable CGCD weights from checkpoint.

    Returns:
        epoch, stage from checkpoint
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}")

    # Infer device from existing model parameters if not specified
    if device is None:
        if teacher_model is not None:
            device = next(teacher_model.parameters()).device
        elif student_model is not None:
            device = next(student_model.parameters()).device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")

    # Load model weights - handle both Stage 0 and Stage N checkpoints
    if "student_state_dict" in checkpoint and "teacher_state_dict" in checkpoint:
        # Stage N checkpoint (with student/teacher separation)
        student_state = checkpoint["student_state_dict"]
        teacher_state = checkpoint["teacher_state_dict"]

        print(f"=> Detected Stage N checkpoint format (student/teacher_state_dict)")
        print(f"[DEBUG] Student state keys (first 5): {list(student_state.keys())[:5]}")
        print(f"[DEBUG] Teacher state keys (first 5): {list(teacher_state.keys())[:5]}")

        # Load student weights (skip if student_model is None - scratch mode)
        if student_model is not None:
            missing_keys_s, unexpected_keys_s = student_model.load_state_dict(student_state, strict=True)
            print(f"✓ Loaded student weights from Stage N checkpoint")
        else:
            print(f"⊘ Skipped student weights loading (scratch mode)")

        # Load teacher weights
        missing_keys_t, unexpected_keys_t = teacher_model.load_state_dict(teacher_state, strict=True)
        print(f"✓ Loaded teacher weights from Stage N checkpoint")

    elif "model_state_dict" in checkpoint:
        # Stage 0 checkpoint (single model, no student/teacher separation)
        model_state = checkpoint["model_state_dict"]

        print(f"=> Detected Stage 0 checkpoint format (model_state_dict)")
        print(f"[DEBUG] Model state keys (first 5): {list(model_state.keys())[:5]}")
        print(f"[DEBUG] Model state total keys: {len(model_state.keys())}")

        # Load student weights (skip if student_model is None - scratch mode)
        if student_model is not None:
            missing_keys_s, unexpected_keys_s = student_model.load_state_dict(model_state, strict=True)
            print(f"✓ Loaded student weights from Stage 0 checkpoint")

            if missing_keys_s:
                print(f"[WARNING] Student missing keys ({len(missing_keys_s)} total)")
                if len(missing_keys_s) <= 10:
                    for key in missing_keys_s:
                        print(f"  - {key}")
                else:
                    print(f"  First 5: {missing_keys_s[:5]}")

            if unexpected_keys_s:
                print(f"[WARNING] Student unexpected keys ({len(unexpected_keys_s)} total)")
                if len(unexpected_keys_s) <= 10:
                    for key in unexpected_keys_s:
                        print(f"  - {key}")
                else:
                    print(f"  First 5: {unexpected_keys_s[:5]}")
        else:
            print(f"⊘ Skipped student weights loading (scratch mode)")

        missing_keys_t, unexpected_keys_t = teacher_model.load_state_dict(model_state, strict=True)
        print(f"✓ Loaded teacher weights from Stage 0 checkpoint (initialized with same weights)")

    else:
        raise KeyError(
            f"Neither 'student_state_dict'/'teacher_state_dict' nor 'model_state_dict' found in checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    # Load CGCD weights.
    # New perturb variants may add runtime-only buffers after older checkpoints were saved.
    # Try strict load first, then fall back to strict=False for backward compatibility.
    if cgcd_model is not None:
        cgcd_state = None
        cgcd_state_name = None
        if "cgcd_state_dict" in checkpoint:
            cgcd_state = checkpoint["cgcd_state_dict"]
            cgcd_state_name = "cgcd_state_dict"
        elif "cgcd_model_state_dict" in checkpoint:
            cgcd_state = checkpoint["cgcd_model_state_dict"]
            cgcd_state_name = "cgcd_model_state_dict (Stage 0 format)"

        if cgcd_state is not None:
            print(f"[DEBUG] Loading {cgcd_state_name}")
            load_cgcd_state_dict(
                cgcd_model,
                cgcd_state,
                state_name=cgcd_state_name,
                preserve_current_stage_state=preserve_cgcd_stage_state,
            )
        else:
            print("[WARNING] No CGCD state dict found in checkpoint")
            print(f"[DEBUG] Available checkpoint keys: {list(checkpoint.keys())}")

    epoch = checkpoint.get("epoch", 0)
    stage = checkpoint.get("stage", 0)

    print(f"\n✓ Successfully loaded checkpoint from Stage {stage}, Epoch {epoch}")
    print(f"{'='*60}\n")

    return epoch, stage


def save_checkpoint_nafnet(model, optimizer, scheduler, epoch, save_path, is_best=False, global_step=0, scaler=None):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

    if is_best:
        best_path = save_path.replace(".pth", "_best.pth")
        torch.save(checkpoint, best_path)
        print(f"Best checkpoint saved to {best_path}")


def load_checkpoint_nafnet(checkpoint_path, model, optimizer=None, scheduler=None):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint.get("global_step", 0)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Resumed from epoch {checkpoint['epoch']}, global_step {global_step}")

    return start_epoch, global_step
