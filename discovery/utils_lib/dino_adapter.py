"""DINOv3 feature extractor with a BOFT adapter fine-tuned to classify the known degradations."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import BOFTConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from utils_lib.datasets import collate

DINO_MODELS = {"dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m"}


class AdapterClassifier(nn.Module):
    """DINO backbone with a linear head on the pooled feature z."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        z = self.backbone(x).pooler_output
        return self.head(z), z


def build_adapter_classifier(num_classes, model_name="dinov3_vitl16"):
    """DINO base model (``$DUDA_DINO_BASE`` or the Hugging Face Hub) with a fresh BOFT adapter."""
    base = os.environ.get("DUDA_DINO_BASE") or DINO_MODELS[model_name]
    print(f"[DINO] Loading base model: {base}")
    model = AutoModel.from_pretrained(base, local_files_only=os.path.isdir(base), trust_remote_code=True)
    peft_config = BOFTConfig(
        boft_block_size=4,
        boft_n_butterfly_factor=2,
        target_modules=["attention.o_proj", "mlp.up_proj", "mlp.down_proj"],
        boft_dropout=0.1,
        bias="boft_only",
    )
    backbone = get_peft_model(model, peft_config)
    classifier = AdapterClassifier(backbone, nn.Linear(backbone.config.hidden_size, num_classes))
    classifier.train()
    backbone.print_trainable_parameters()
    return classifier


def finetune_adapter(train_set, num_classes, epochs, batch_size, model_name, save_dir, seed=42, test_set=None):
    """Fine-tune the adapter with cross-entropy on the known classes (one process per GPU)."""
    torch.manual_seed(seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    classifier = build_adapter_classifier(num_classes, model_name).to(device)
    classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = AdamW(classifier.parameters(), lr=1e-3)
    sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True, collate_fn=collate
    )
    if test_set is not None and is_main:
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate
        )

    for epoch in range(epochs):
        classifier.train()
        sampler.set_epoch(epoch)
        loss_sum, num_batches, correct, total = 0.0, 0, 0, 0
        batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") if is_main else train_loader
        for batch in batches:
            images, labels = batch["images"].to(device), batch["labels"].to(device)
            logits, _ = classifier(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            num_batches += 1
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            if is_main:
                batches.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        message = f"Epoch {epoch + 1}/{epochs} - loss {loss_sum / num_batches:.4f}, train acc {correct / total:.4f}"
        if test_set is not None and is_main:
            model = classifier.module
            model.eval()
            test_correct = test_total = 0
            with torch.no_grad():
                for batch in test_loader:
                    logits, _ = model(batch["images"].to(device))
                    test_correct += (logits.argmax(dim=1) == batch["labels"].to(device)).sum().item()
                    test_total += batch["labels"].size(0)
            message += f", test acc {test_correct / test_total:.4f}"
        if is_main:
            print(message)

    model = classifier.module
    model.eval()
    if is_main:
        save_adapter(model, save_dir, num_classes, model_name, batch_size, epochs)
    dist.barrier()
    return model.backbone


def save_adapter(model, save_dir, num_classes, model_name, batch_size, epochs):
    os.makedirs(save_dir, exist_ok=True)
    model.backbone.save_pretrained(os.path.join(save_dir, "adapter"))
    # With bias="boft_only" the base-layer biases are trained but not saved by save_pretrained.
    biases = {
        n: p.data.cpu() for n, p in model.backbone.named_parameters() if "base_layer.bias" in n and p.requires_grad
    }
    torch.save(biases, os.path.join(save_dir, "base_bias.pt"))
    torch.save(
        {"state_dict": model.head.state_dict(), "num_classes": num_classes, "embed_dim": model.head.in_features},
        os.path.join(save_dir, "head.pt"),
    )
    with open(os.path.join(save_dir, "training_config.txt"), "w") as f:
        f.write(f"model_name: {model_name}\nnum_classes: {num_classes}\nbatch_size: {batch_size}\nepochs: {epochs}\n")
        f.write(f"learning_rate: 1e-3\noptimizer: AdamW\ndistributed: True\nworld_size: {dist.get_world_size()}\n")
    print(f"Saved adapter, base-layer biases, and head to {save_dir}")


def load_adapter_backbone(checkpoint_dir, num_classes, model_name="dinov3_vitl16", device="cuda", seed=42):
    """Frozen DINO backbone with the fine-tuned adapter. ``z = backbone(x).pooler_output``."""
    torch.manual_seed(seed)
    classifier = build_adapter_classifier(num_classes, model_name)
    backbone = classifier.backbone
    weights = load_file(os.path.join(checkpoint_dir, "adapter", "adapter_model.safetensors"))
    # boft_P buffers are not restored by set_peft_model_state_dict.
    set_peft_model_state_dict(backbone, {k: v for k, v in weights.items() if "boft_P" not in k})
    for key, value in weights.items():
        if "boft_P" in key:
            module = backbone
            for part in key.split(".")[:-1]:
                module = getattr(module, part)
            if hasattr(module, key.split(".")[-1]):
                getattr(module, key.split(".")[-1]).copy_(value.to(device))
    state = backbone.state_dict()
    for key, value in torch.load(os.path.join(checkpoint_dir, "base_bias.pt"), map_location=device).items():
        if key in state:
            state[key].copy_(value.to(device))
    print(f"[DINO] Loaded adapter from {checkpoint_dir}")
    backbone.eval()
    return backbone.to(device)
