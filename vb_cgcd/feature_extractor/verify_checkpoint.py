#!/usr/bin/env python3
"""
Verify that checkpoint loading produces identical features to trained model
"""

import torch
import numpy as np


def verify_checkpoint_loading(dino_trained, dino_checkpoint, test_dataset, collate_fn, device="cuda", batch_size=64):
    """
    Compare features from trained model (in memory) vs model loaded from checkpoint.

    Args:
        dino_trained: Model after training (in memory)
        dino_checkpoint: Model loaded from checkpoint
        test_dataset: Dataset to extract features from
        collate_fn: Collate function for dataloader
        device: Device to run on
        batch_size: Batch size for verification

    Returns:
        bool: True if features match, False otherwise
    """
    print(f"\n{'='*80}")
    print("CHECKPOINT VERIFICATION")
    print(f"{'='*80}")

    # Create a small test loader
    test_batch_size = min(batch_size, len(test_dataset))
    test_subset = torch.utils.data.Subset(test_dataset, range(test_batch_size))
    test_verify_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=test_batch_size, collate_fn=collate_fn, shuffle=False
    )

    print(f"\nComparing features from trained model vs checkpoint model...")
    print(f"Using {test_batch_size} samples for verification")

    for batch in test_verify_loader:
        images = batch["images"].to(device)

        with torch.no_grad():
            # Features from trained model (in memory)
            feat_trained = dino_trained(images).pooler_output.cpu().numpy()

            # Features from checkpoint model
            feat_checkpoint = dino_checkpoint(images).pooler_output.cpu().numpy()

        # Compare
        max_diff = np.abs(feat_trained - feat_checkpoint).max()
        mean_diff = np.abs(feat_trained - feat_checkpoint).mean()

        print(f"\nFeature Statistics:")
        print(f"  Trained model:")
        print(f"    mean: {feat_trained.mean():.6f}, std: {feat_trained.std():.6f}")
        print(f"    min: {feat_trained.min():.6f}, max: {feat_trained.max():.6f}")
        print(f"  Checkpoint model:")
        print(f"    mean: {feat_checkpoint.mean():.6f}, std: {feat_checkpoint.std():.6f}")
        print(f"    min: {feat_checkpoint.min():.6f}, max: {feat_checkpoint.max():.6f}")

        print(f"\nDifference:")
        print(f"  Max difference: {max_diff:.10f}")
        print(f"  Mean difference: {mean_diff:.10f}")

        if max_diff < 1e-5:
            print(f"\n  ✅ SUCCESS! Features match perfectly (max diff < 1e-5)")
            print(f"     Checkpoint loading works correctly!")
            success = True
        elif max_diff < 1e-3:
            print(f"\n  ⚠️  WARNING! Small difference detected (1e-5 < max diff < 1e-3)")
            print(f"     This might be due to numerical precision")
            success = False
        else:
            print(f"\n  ❌ FAILURE! Significant difference detected (max diff >= 1e-3)")
            print(f"     Checkpoint loading has issues!")
            print(f"\n  Possible causes:")
            print(f"    1. Buffers (boft_P) not loaded correctly")
            print(f"    2. Pooler weights not loaded correctly")
            print(f"    3. Random initialization mismatch")
            print(f"    4. Model architecture mismatch")
            success = False

        break  # Only need one batch

    print(f"{'='*80}\n")

    return success


if __name__ == "__main__":
    print("This is a utility module. Import and use verify_checkpoint_loading() function.")
