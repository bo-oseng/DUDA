"""
Utility functions for incremental learning class management.
"""

def load_class_order(class_order_path):
    """Load class order from file."""
    with open(class_order_path, 'r') as f:
        order = [int(x) for x in f.read().strip().split(',')]
    return order


def get_old_new_classes(config):
    """
    Determine OLD and NEW classes based on config.

    Args:
        config: Config dict with 'deg_map', 'incremental', and 'cgcd' sections

    Returns:
        old_classes: List of OLD class names (labeled data)
        new_classes: List of NEW class names (unlabeled data)
        old_indices: List of OLD class indices in deg_map
        new_indices: List of NEW class indices in deg_map
    """
    # Load class order
    class_order = load_class_order(config['cgcd']['class_order'])

    # Get incremental learning parameters
    base_class_num = config['incremental']['base_class_num']
    inc_class_num = config['incremental']['inc_class_num']

    # Create reverse mapping: index -> class_name
    deg_map = config['deg_map']
    idx_to_class = {v: k for k, v in deg_map.items()}

    # Split class_order into OLD and NEW
    old_order_indices = class_order[:base_class_num]  # First base_class_num classes
    new_order_indices = class_order[base_class_num:base_class_num + inc_class_num]  # Next inc_class_num classes

    # Convert to class names
    old_classes = [idx_to_class[idx] for idx in old_order_indices]
    new_classes = [idx_to_class[idx] for idx in new_order_indices]

    return old_classes, new_classes, old_order_indices, new_order_indices


def print_incremental_info(config):
    """Print incremental learning configuration."""
    old_classes, new_classes, old_indices, new_indices = get_old_new_classes(config)

    stage = config['incremental']['stage']
    base_class_num = config['incremental']['base_class_num']
    inc_class_num = config['incremental']['inc_class_num']

    print(f"\n{'='*60}")
    print(f"Incremental Learning Stage {stage} Configuration")
    print(f"{'='*60}")
    print(f"\nOLD Classes (Labeled, {base_class_num} classes):")
    for i, (cls, idx) in enumerate(zip(old_classes, old_indices)):
        print(f"  [{i}] {cls:15s} (deg_map index: {idx})")

    print(f"\nNEW Classes (Unlabeled, {inc_class_num} classes):")
    for i, (cls, idx) in enumerate(zip(new_classes, new_indices)):
        print(f"  [{i}] {cls:15s} (deg_map index: {idx})")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Test with example config
    import yaml

    config_path = "configs/train_wcgcd_stage1_lora_debug.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print_incremental_info(config)

    # Show usage example
    old_classes, new_classes, old_indices, new_indices = get_old_new_classes(config)
    print("\nUsage Example:")
    print(f"OLD classes for labeled training: {old_classes}")
    print(f"NEW classes for unlabeled training: {new_classes}")
