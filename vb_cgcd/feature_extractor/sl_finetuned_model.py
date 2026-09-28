import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
import torchvision
from torch.utils.data import DataLoader

from PIL import ImageFilter, ImageOps, Image
from torchvision import transforms

from peft import LoraConfig, get_peft_model, BOFTConfig
from transformers import AutoModel
from peft.peft_model import PeftModel
from peft.config import PeftConfig

from tqdm import tqdm
import copy

from torchvision import datasets, transforms
from torchvision import models as torchvision_models


# Fall back to absolute import when run directly
import vision_transformer as vits
from vision_transformer import DINOHead
import utils

import argparse
import os

from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

torch.manual_seed(42)  # Set random seed for reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.backbone = backbone

        self.head = head

    def forward(self, x):
        # Run the head forward on the concatenated features.
        pooler_output = self.backbone(x).pooler_output

        cls_output = self.head(pooler_output)

        return cls_output, pooler_output


def load_model(num_classes, model_name):
    LOCAL_BASE = "./local_models"
    name_map = {
        # dinov1
        "dino_vitb16": "dino-vitb16",
        "facebook/dino-vitb16": "dino-vitb16",
        # dinov2
        "dinov2_vits14": "dinov2-vits14",
        "dinov2_vitb14": "dinov2-vitb14",
        "dinov2_vitl14": "dinov2-vitl14",
        "dinov2_vitg14": "dinov2-vitg14",
        "facebook/dinov2-base": "dinov2-vitb14",
        # dinov3
        "dinov3_vits16": "dinov3-vits16",
        "dinov3_vitb16": "dinov3-vitb16",
        "dinov3_vitl16": "dinov3-vitl16",
    }

    internal_name = name_map.get(model_name, model_name.replace("/", "-"))
    local_path = os.environ.get("DUDA_DINO_BASE", os.path.join(LOCAL_BASE, internal_name))

    print(f"🔍 Local model found at: {local_path}")
    load_target = local_path
    local_only = True

    model = AutoModel.from_pretrained(load_target, local_files_only=local_only, trust_remote_code=True)

    peft_config = BOFTConfig(
        boft_block_size=4,
        boft_n_butterfly_factor=2,
        target_modules=["output.dense", "mlp.fc1", "mlp.fc2"],
        boft_dropout=0.1,
        bias="boft_only",
    )

    backbone = get_peft_model(model, peft_config)
    embed_dim = backbone.config.hidden_size

    lora_model = MultiCropWrapper(backbone, torch.nn.Linear(embed_dim, num_classes))
    lora_model.train()
    backbone.print_trainable_parameters()

    return lora_model


def load_finetuned_model_from_checkpoint(
    checkpoint_dir, num_classes=12, model_name="dino_vitb16", device="cuda", seed=42
):
    torch.manual_seed(seed)
    lora_model = load_model(num_classes, model_name)

    adapter_path = os.path.join(checkpoint_dir, "adapter")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(adapter_weights_path):
        adapter_weights = load_file(adapter_weights_path)
    else:
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
        adapter_weights = torch.load(adapter_weights_path, map_location=device)

    # Separate trainable parameters and buffers
    trainable_keys = {k: v for k, v in adapter_weights.items() if "boft_P" not in k}
    buffer_keys = {k: v for k, v in adapter_weights.items() if "boft_P" in k}

    # Load trainable parameters
    set_peft_model_state_dict(lora_model.backbone, trainable_keys)

    # Load buffers (boft_P) - these are not loaded by set_peft_model_state_dict!
    if buffer_keys:
        for key, value in buffer_keys.items():
            # Navigate to the buffer and copy it
            parts = key.split(".")
            module = lora_model.backbone
            for part in parts[:-1]:
                module = getattr(module, part)
            buffer_name = parts[-1]
            if hasattr(module, buffer_name):
                getattr(module, buffer_name).copy_(value.to(device))
        print(f"✓ Loaded {len(buffer_keys)} buffers (boft_P)")

    print(f"✓ Adapter weights loaded from {adapter_path}")

    # Load base_layer.bias parameters
    base_bias_path = os.path.join(checkpoint_dir, "base_bias.pt")
    if os.path.exists(base_bias_path):
        base_bias_dict = torch.load(base_bias_path, map_location=device)
        # Load into model
        model_state_dict = lora_model.backbone.state_dict()
        for key, value in base_bias_dict.items():
            if key in model_state_dict:
                model_state_dict[key].copy_(value.to(device))
        print(f"✓ Loaded {len(base_bias_dict)} base layer bias parameters")

    # Load pooler weights
    pooler_path = os.path.join(checkpoint_dir, "pooler.pt")
    if os.path.exists(pooler_path):
        pooler_state = torch.load(pooler_path, map_location=device)
        target_model = (
            lora_model.backbone.base_model.model if hasattr(lora_model.backbone, "base_model") else lora_model.backbone
        )
        if hasattr(target_model, "pooler"):
            target_model.pooler.load_state_dict(pooler_state)
            print(f"✓ Pooler weights loaded from {pooler_path}")

    # Set to eval mode and move to device
    lora_model.backbone.eval()
    lora_model.backbone = lora_model.backbone.to(device)

    return lora_model.backbone


def finetune_dino(
    train_set,
    num_classes,
    epochs=10,
    batch_size=128,
    model_name="facebook/dino-vitb16",
    save_dir=None,
    seed=42,
    test_set=None,
):
    torch.manual_seed(seed)
    interpolation = 3
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = 224
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )

    batch_size = batch_size

    lora_model = load_model(num_classes, model_name)

    # Optimizer
    optimizer = AdamW(lora_model.parameters(), lr=1e-3)

    lora_model.to(device)

    print(train_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2) if test_set is not None else None

    # Training loop
    for epoch in range(epochs):  # Adjust epochs as needed
        lora_model.train()
        epoch_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for bidx, batch in progress_bar:

            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            output, pooler_output = lora_model(images)

            cls_loss = F.cross_entropy(output, labels)

            loss = cls_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Accuracy
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar with current loss and accuracy
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss/num_batches:.4f}",
                    "acc": f"{correct/total:.4f}",
                }
            )

        train_acc = correct / total
        test_acc_str = ""
        if test_loader is not None:
            lora_model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["images"].to(device)
                    labels = batch["labels"].to(device)
                    output, _ = lora_model(images)
                    preds = output.argmax(dim=1)
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)
            if test_total > 0:
                test_acc = test_correct / test_total
                test_acc_str = f" | Test Acc: {test_acc:.4f}"

        print(
            f"Epoch {epoch+1}/{epochs} completed - Avg Loss: {epoch_loss/num_batches:.4f} | Train Acc: {train_acc:.4f}{test_acc_str}"
        )

    lora_model.eval()
    # Save model checkpoint if save_dir is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Save PEFT adapter (BOFT weights)
        adapter_path = os.path.join(save_dir, "adapter")
        lora_model.backbone.save_pretrained(adapter_path)
        print(f"\n✓ PEFT adapter saved to: {adapter_path}")

        # CRITICAL: Save base_layer.bias parameters (trainable but not saved by save_pretrained)
        # BOFT with bias="boft_only" makes base_layer.bias trainable, but save_pretrained doesn't save them
        base_bias_dict = {}
        for name, param in lora_model.backbone.named_parameters():
            if "base_layer.bias" in name and param.requires_grad:
                base_bias_dict[name] = param.data.cpu()

        if base_bias_dict:
            base_bias_path = os.path.join(save_dir, "base_bias.pt")
            torch.save(base_bias_dict, base_bias_path)
            print(f"✓ Base layer biases saved to: {base_bias_path} ({len(base_bias_dict)} parameters)")

        pooler_path = os.path.join(save_dir, "pooler.pt")
        if hasattr(lora_model.backbone.base_model.model, "pooler"):
            torch.save(lora_model.backbone.base_model.model.pooler.state_dict(), pooler_path)
            print(f"✓ Pooler weights saved to: {pooler_path}")

        # Save classification head
        head_path = os.path.join(save_dir, "head.pt")
        torch.save(
            {
                "state_dict": lora_model.head.state_dict(),
                "num_classes": num_classes,
                "embed_dim": lora_model.head.in_features,
            },
            head_path,
        )
        print(f"✓ Classification head saved to: {head_path}")

        # Save training config
        config_path = os.path.join(save_dir, "training_config.txt")
        with open(config_path, "w") as f:
            f.write(f"model_name: {model_name}\n")
            f.write(f"num_classes: {num_classes}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"epochs: {epochs}\n")
            f.write(f"learning_rate: 1e-3\n")
            f.write(f"optimizer: AdamW\n")
        print(f"✓ Training config saved to: {config_path}\n")

    return lora_model.backbone
