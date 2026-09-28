# Reproduction stages

| Stage | Command | Configuration | Output |
|---|---|---|---|
| Feature learning | `python scripts/run.py features` | DINOv3 ViT-L/16, BOFT, 10 epochs, seed 42 | `outputs/features/wres_with_clear/` |
| VB-CGCD | `python scripts/run.py cgcd` | 6+3 schedule, 384-dimensional PCA | `outputs/cgcd/wres_with_clear/cgcd_6p3/` |
| Supervised restoration | `python scripts/run.py supervised` | `configs/supervised.yml`, 500 epochs | `outputs/checkpoints/supervised/` |
| Semi-supervised restoration | `python scripts/run.py semi_supervised` | `configs/semi_supervised.yml`, 10 epochs, default augmentation | `outputs/checkpoints/semi_supervised/` |
| Teacher evaluation | `python scripts/run.py evaluate` | Semi-supervised config, epoch-10 teacher | `outputs/results/table1/` |
| Saved-image Q-Align | `python scripts/run.py qalign` | Teacher outputs | `outputs/results/table1/` |

Fresh supervised training starts without a resume checkpoint. Adaptation initializes from the supplied epoch-500 supervised model. Set `RESUME_CHECKPOINT` only when continuing an interrupted run. Resolved runtime configurations are written under `outputs/configs/`.
