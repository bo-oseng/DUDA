"""Check release syntax, asset links and reproduction configurations."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

from run import ROOT, render_config, settings


def main():
    errors = []
    files = list(ROOT.rglob('*.py'))
    for path in files:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(str(exc))
    required = ['data/syn_train_dataset', 'data/real_train_patches', 'data/real_eval_dataset',
                'data/real_eval_small_dataset', 'data/real_train_dataset',
                'features/wres_with_clear/dinov3-vitl16-sl/class_order.txt',
                'dino_base/config.json', 'dino_adapter/adapter/adapter_config.json',
                'cgcd/pca_model.pkl', 'cgcd/scaler_model.pkl', 'cgcd/class_mappings.json',
                'cgcd/stage0class_means.npy', 'cgcd/stage0class_covariances.npy',
                'cgcd/stage1class_means.npy', 'cgcd/stage1class_covariances.npy',
                'checkpoints/stage0_epoch500.pth', 'checkpoints/semi_supervised_epoch10.pth']
    for name in required:
        if not (ROOT / 'assets' / name).exists():
            errors.append('Missing asset: ' + name)
    for stage in ('supervised', 'semi_supervised'):
        config = render_config(stage, settings(stage))
        for key in ('pca_path','scaler_path','class_mappings','dino_checkpoint','class_order'):
            if not Path(config['cgcd'][key]).exists():
                errors.append(f'{stage}: missing {key}')
    semi = render_config('semi_supervised', settings('semi_supervised'))
    assert semi['train']['total_epoch'] == 10
    assert semi['train']['semiuir_strong_aug_variant'] == 'default'
    metrics = json.loads((ROOT / 'reference/table1_metrics.json').read_text())
    for metric, value in metrics['average'].items():
        mean = sum(d[metric] for d in metrics['per_set'].values()) / 3
        if abs(mean - value) > 1e-9:
            errors.append(f'Incorrect reference aggregate: {metric}')
    missing_modules = [m for m in ('torch','torchvision','pyiqa','peft','transformers','jax','numpyro','continuum')
                       if importlib.util.find_spec(m) is None]
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()
               and 'outputs' not in p.relative_to(ROOT).parts and 'assets' not in p.relative_to(ROOT).parts)
    print(f'Parsed {len(files)} Python files. Code and documentation: {size / 1024**2:.2f} MiB.')
    print('Missing Python packages:', ', '.join(missing_modules) or 'none')
    for error in errors:
        print('ERROR:', error)
    if errors or missing_modules:
        sys.exit(1)
    print('Project checks passed.')


if __name__ == '__main__':
    main()
