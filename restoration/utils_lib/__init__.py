"""
Utility library for incremental learning and dataset management.
"""

from .utils import (
    seed_everything,
    saveImage,
    save_rgb,
    load_img,
    plot_all,
    modcrop,
    dict2namespace,
    count_params,
    save_checkpoint,
    load_checkpoint,
)

from .utils_incremental import (
    load_class_order,
    get_old_new_classes,
    print_incremental_info,
)

from .utils_dataset import (
    generate_train_val_data,
    filter_old_new_data,
    get_train_val_data_for_stage,
)

__all__ = [
    # General utilities
    'seed_everything',
    'saveImage',
    'save_rgb',
    'load_img',
    'plot_all',
    'modcrop',
    'dict2namespace',
    'count_params',
    'save_checkpoint',
    'load_checkpoint',
    # Incremental learning utilities
    'load_class_order',
    'get_old_new_classes',
    'print_incremental_info',
    # Dataset utilities
    'generate_train_val_data',
    'filter_old_new_data',
    'get_train_val_data_for_stage',
]
