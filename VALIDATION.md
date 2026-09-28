# Validation

The extracted release was checked in the existing CUDA environment on an NVIDIA A100 GPU.

- All 56 Python files passed syntax checks.
- All seven original entrypoints passed import and `--help` checks: feature learning, real-feature export, VB-CGCD, supervised restoration, semi-supervised restoration, restoration evaluation and saved-image Q-Align.
- Dataset/model links resolve from both the host and container project locations.
- The initial checkpoint reports epoch 500. The adapted checkpoint reports epoch 10 and global step 14470.
- Both restoration checkpoints loaded with strict state-dictionary matching for OneRestore and its CGCD module.
- A real image crop passed through the DINOv3 adapter, CGCD module and epoch-10 teacher. Features had shape `[1, 1024]`; the restored image had shape `[1, 3, 224, 224]`.
- Using the same features and checkpoint, the extracted and original OneRestore/CGCD implementations produced identical output: maximum absolute difference `0.0` under FP16 autocast.
- MUSIQ, CLIP-IQA and LIQE each produced a finite score for a single real-image crop in the main environment (PyTorch 2.4.0+cu121, pyiqa 0.1.14.1).
- Q-Align produced a finite score for the same crop in the separate compatible environment (Python 3.10, PyTorch 2.1.2+cu121, Transformers 4.37.2, pyiqa 0.1.14.1).
- The reference Overall metrics equal the unweighted mean of the three per-dataset metrics.

These checks validate extraction, loading and a small inference/evaluation path. They do not constitute a new 500-epoch/10-epoch training run or a complete evaluation of all 7,971 images. The stored reference metrics were copied numerically from the supplied evaluation records.
