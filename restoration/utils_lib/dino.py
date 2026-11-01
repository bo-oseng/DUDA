"""DINOv3 backbone with the BOFT adapter fine-tuned for known-degradation classification."""

import os

import torch
from peft import BOFTConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModel

DINO_MODELS = {"dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m"}


def load_degradation_encoder(checkpoint_dir, num_classes, model_name="dinov3_vitl16", device="cuda", seed=42):
    """Frozen feature extractor z = DINO(x) used by VB-CGCD and the descriptor.

    The base model is read from ``$DUDA_DINO_BASE`` when set, otherwise from the Hugging Face Hub.
    """
    # The global seed is reset here, as in the original training code, so that every
    # module created afterwards is initialized from the same random state.
    torch.manual_seed(seed)

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
    # The classification head is not used here; it is created only to keep the random state
    # identical to the adapter training code.
    torch.nn.Linear(backbone.config.hidden_size, num_classes)
    backbone.train()
    backbone.print_trainable_parameters()

    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    adapter_weights = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    # boft_P buffers are not restored by set_peft_model_state_dict.
    set_peft_model_state_dict(backbone, {k: v for k, v in adapter_weights.items() if "boft_P" not in k})
    buffers = {k: v for k, v in adapter_weights.items() if "boft_P" in k}
    for key, value in buffers.items():
        module = backbone
        for part in key.split(".")[:-1]:
            module = getattr(module, part)
        if hasattr(module, key.split(".")[-1]):
            getattr(module, key.split(".")[-1]).copy_(value.to(device))
    print(f"[DINO] Loaded {len(buffers)} boft_P buffers and adapter weights from {adapter_dir}")

    base_bias = torch.load(os.path.join(checkpoint_dir, "base_bias.pt"), map_location=device)
    state = backbone.state_dict()
    for key, value in base_bias.items():
        if key in state:
            state[key].copy_(value.to(device))
    print(f"[DINO] Loaded {len(base_bias)} base-layer biases")

    backbone.eval()
    backbone = backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone
