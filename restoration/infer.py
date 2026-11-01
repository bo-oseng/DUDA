"""Restore one image with the released DUDA checkpoints."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run import ROOT, render_config, settings

from utils_lib.checkpoint import load_for_inference
from utils_lib.common import amp_dtype_from_cfg, dict2namespace, enable_tf32, seed_everything
from utils_lib.datasets import RealImageDataset
from utils_lib.descriptor import DiscoveryInstanceDescriptor
from utils_lib.dino import load_degradation_encoder
from utils_lib.onerestore import OneRestore
from utils_lib.real_eval import dino_resize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output PNG; a JSON run record is saved alongside it")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=1, help="0: supervised initialization; 1: adapted model")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input image does not exist: {args.input}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Expose one CUDA GPU, for example CUDA_VISIBLE_DEVICES=2")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    if (ROOT / "assets/dino_base").is_dir():
        os.environ.setdefault("DUDA_DINO_BASE", str(ROOT / "assets/dino_base"))
    stage = "supervised" if args.stage == 0 else "semi_supervised"
    values = settings(stage)
    cfg = dict2namespace(render_config(stage, values))
    checkpoint = args.checkpoint or Path(values["STAGE0_CHECKPOINT" if args.stage == 0 else "CHECKPOINT"])
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    seed_everything(cfg.train.seed)
    enable_tf32()
    _, amp_dtype = amp_dtype_from_cfg(cfg)
    encoder = load_degradation_encoder(
        cfg.discovery.dino_checkpoint, cfg.discovery.dino_num_classes, cfg.discovery.dino_model, device
    )
    from peft.tuners.boft import layer as boft_layer

    if boft_layer._FBD_CUDA is None:
        raise RuntimeError("BOFT CUDA extension failed to load; the one-factor fallback is incompatible with these weights")
    descriptor = DiscoveryInstanceDescriptor(
        cluster_dir=cfg.discovery.cluster_dir,
        output_dim=cfg.discovery.descriptor_dim,
        stage=cfg.incremental.stage,
        pca_path=cfg.discovery.pca_path,
        scaler_path=getattr(cfg.discovery, "scaler_path", None),
    ).to(device).eval()
    model = OneRestore(channel=cfg.model.width).to(device).eval()
    epoch = load_for_inference(checkpoint, model, descriptor)
    model.set_runtime_resize_fallback(True)
    image, _ = RealImageDataset([str(args.input)])[0]
    image = image.unsqueeze(0).to(device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        feature = encoder(dino_resize(image)).pooler_output
        embedding, _ = descriptor(feature)
        restored = model(image, embedding).float().clamp(0, 1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if restored.shape != image.shape or not bool(torch.isfinite(restored).all()):
        raise RuntimeError("Restoration output has an invalid shape or nonfinite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(restored[0], args.output)
    record = {
        "input": str(args.input), "output": str(args.output), "checkpoint": str(checkpoint),
        "epoch": epoch, "stage": cfg.incremental.stage, "model": "teacher" if args.stage == 1 else "supervised",
        "shape": list(restored.shape), "feature_shape": list(feature.shape),
        "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "forward_seconds": elapsed, "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "peft")},
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
