# Continual streams (Table 3)

This guide covers the two continual streams reported in Table 3. Each stream starts from the 6 base degradations and adds new ones over several stages:

| Stream | Script suffix | New degradations per stage | Stages |
|---|---|---|---|
| 6+1+2 | `6p1p2` | +1, then +2 | 1–2 |
| 6+1+1+1 | `6p1p1p1` | +1, +1, then +1 | 1–3 |

## 1. Download the weights

All continual weights are in a single archive, `continual/duda-continual-weights.tar.gz`, at [ses7720/DUDA](https://huggingface.co/ses7720/DUDA). Extract it in the DUDA repository root:

```bash
hf download ses7720/DUDA --include 'continual/duda-continual-weights.tar.gz' --local-dir /tmp/duda-hf
cd /path/to/DUDA
tar -xzf /tmp/duda-hf/continual/duda-continual-weights.tar.gz --strip-components=1
```

This creates, for each stream:

```text
assets/continual/{stream}/
├── cgcd/          # VB-CGCD statistics: class_mappings.json, PCA, scaler, class statistics
└── checkpoints/   # restoration checkpoints, one per stage
```

The base DINOv3 and BOFT adapter are the same as in the main release.

> [!IMPORTANT]
> Always use a checkpoint with the `cgcd/` directory of the **same** stream. Do not combine a 6+1+2 checkpoint with the 6+1+1+1 statistics, or the reverse.

## 2. Evaluate

Set `STAGE` and run the test script for the stream:

```bash
STAGE=2 bash bash/test_6p1p2.sh      # 6+1+2, stage 2
STAGE=3 bash bash/test_6p1p1p1.sh    # 6+1+1+1, stage 3
```

By default, evaluation runs on GPU 0 with the released checkpoint for that stage.

| Variable | Purpose |
|---|---|
| `CHECKPOINT=/path/to/checkpoint.pth` | Evaluate your own checkpoint instead of the released one |
| `RESULTS_ROOT=/path/to/results` | Write results to a separate directory for each checkpoint |

## 3. Train

Training requires the WRES datasets, including the real training patches.

Run the stages in order. Each stage starts from the previous one:

```bash
STAGE=1 bash bash/train_6p1p2.sh
STAGE=2 bash bash/train_6p1p2.sh
```

| Stage | Starts from |
|---|---|
| 1 | The main supervised checkpoint |
| 2 or 3 | `outputs/checkpoints/{stream}_stage{previous}/stage{previous}_latest.pth` |

To start from a different previous-stage checkpoint, set `PREVIOUS_STAGE_CHECKPOINT`. For example, to skip stage 1 and start stage 2 from the released stage 1 checkpoint:

```bash
STAGE=2 PREVIOUS_STAGE_CHECKPOINT="$PWD/assets/continual/6p1p2/checkpoints/stage1_latest.pth" bash bash/train_6p1p2.sh
```

Defaults: GPUs 0,1,2 and batch size 56 per GPU. Each stage gets its own experiment directory and reliable bank.

Each stage saves these checkpoints, with the stage number also recorded in the checkpoint metadata:

- `stage{stage}_latest.pth`
- `stage{stage}_epoch_{epoch}.pth`
- `stage{stage}_step_{step}.pth`

The main semi-supervised run counts as stage 1.

## 4. Dry run

Add `--dry-run` to any train or test script to print the generated command and configuration without running anything.
