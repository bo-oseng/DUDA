"""Configuration, distributed setup, and runtime helpers shared by the entry points."""

import argparse
import datetime
import os
import random
import re

import numpy as np
import torch
import torch.distributed as dist
import yaml


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def load_config(path):
    with open(path, "r") as f:
        return dict2namespace(yaml.safe_load(f))


def configure_torch_extensions_dir():
    """Give each rank its own build directory for the BOFT CUDA extension."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    base_ext_dir = os.environ.get("BASE_EXT_DIR")
    if base_ext_dir:
        os.environ["TORCH_EXTENSIONS_DIR"] = f"{base_ext_dir}_{local_rank}"
        return
    current_ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if current_ext_dir and re.search(r"_\d+$", current_ext_dir):
        os.environ["TORCH_EXTENSIONS_DIR"] = f"{current_ext_dir.rsplit('_', 1)[0]}_{local_rank}"


def setup_ddp(timeout=datetime.timedelta(minutes=30)):
    dist.init_process_group(backend="nccl", timeout=timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def amp_dtype_from_cfg(cfg):
    use_bf16 = bool(getattr(cfg.train, "use_bf16", False))
    return use_bf16, (torch.bfloat16 if use_bf16 else torch.float16)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def batch_psnr(y_true, y_pred, maxval=1.0):
    mse = torch.mean((y_true - y_pred) ** 2, dim=(1, 2, 3))
    return (20 * torch.log10(maxval / torch.sqrt(mse))).unsqueeze(1)
