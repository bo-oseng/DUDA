# Official PyTorch Implementation of
# DUDA: Discovering Unseen Degradations to Adapt Open-World Image Restoration

> Accepted at NeurIPS 2026.

## TODO

- [ ] Add the paper link and citation information.
- [ ] Verify the setup and run instructions from a fresh checkout in the target environment.

## Setup

The reference training environment uses Python 3.11, PyTorch 2.4.0, torchvision 0.19.0, transformers 4.56.0, PEFT 0.12.0, and pyiqa 0.1.14.1. Training uses three CUDA GPUs; BOFT requires a CUDA toolkit and C++ compiler.

```bash
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Q-Align uses a separate environment because its model code is incompatible with the newer Transformers version used for DINOv3. The tested environment is Python 3.10 with PyTorch 2.1.2, torchvision 0.16.2, Transformers 4.37.2, and pyiqa 0.1.14.1. Install `requirements-qalign.txt` there and set `QALIGN_PYTHON` to that environment's Python executable.

## Data and pretrained assets

Datasets and checkpoints are not included. Link local copies into `assets/`:

```bash
python scripts/link_assets.py \
  --data /path/to/wres_datasets \
  --features /path/to/feature_root \
  --dino-base /path/to/dinov3-vitl16 \
  --dino-adapter /path/to/dinov3-vitl16-sl/checkpoint \
  --cgcd /path/to/cgcd/saved_models \
  --stage0 /path/to/stage0_epoch500.pth \
  --adapted /path/to/semi_supervised_epoch10.pth
```

The data root must contain `syn_train_dataset`, `real_train_dataset`, `real_train_patches`, `real_eval_dataset`, and `real_eval_small_dataset`. Keep the existing train/test split manifests. Unlabeled restoration training uses precomputed 224-pixel patches. The feature directory must contain `wres_with_clear/dinov3-vitl16-sl/`; the CGCD directory must contain the PCA, scaler, class mapping, and stage Gaussian parameter files.

## Quick start

Run commands from the repository root. The launcher writes generated files to `outputs/`. It accepts `NPROC_PER_NODE` (default `3`), `OUTPUT_ROOT`, `DATA_ROOT`, and `EXP_NAME`; select GPUs with `CUDA_VISIBLE_DEVICES`.

```bash
# Build DINO features and fit VB-CGCD
python scripts/run.py features
python scripts/run.py cgcd

# Train supervised initialization, then semi-supervised restoration
python scripts/run.py supervised
python scripts/run.py semi_supervised

# Evaluate the epoch-10 teacher and compute Q-Align on saved outputs
python scripts/run.py evaluate
export QALIGN_PYTHON=/path/to/qalign-env/bin/python
NPROC_PER_NODE=1 python scripts/run.py qalign
```

The supervised stage trains for 500 epochs. Semi-supervised adaptation trains for 10 epochs with the default augmentation. To continue an interrupted run, set `RESUME_CHECKPOINT` to its checkpoint; adaptation resumes also require the matching pseudo-label bank. Use `python scripts/run.py <stage> --dry-run` to inspect a command and its resolved configuration.

## Repository layout

- `vb_cgcd/` — feature learning, incremental Gaussian fitting, clustering, and data loading.
- `restoration/` — OneRestore, CGCD conditioning, training, and evaluation.
- `configs/` — supervised and semi-supervised configurations.
- `scripts/` — launch and asset-link helpers.
- `reference/` — small reference metrics in JSON format.

See `THIRD_PARTY.md` for upstream components and their applicable terms.
