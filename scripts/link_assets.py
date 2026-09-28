"""Connect local datasets and weights without copying large assets."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Directory containing the five WRES dataset folders')
    parser.add_argument('--features', type=Path, help='Directory containing wres_with_clear/dinov3-vitl16-sl')
    parser.add_argument('--dino-base', type=Path)
    parser.add_argument('--dino-adapter', type=Path)
    parser.add_argument('--cgcd', type=Path, help='Directory with PCA, scaler, mapping and stage Gaussian parameters')
    parser.add_argument('--stage0', type=Path)
    parser.add_argument('--adapted', type=Path)
    args = parser.parse_args()
    mapping = {'data': 'data', 'features': 'features', 'dino_base': 'dino_base',
               'dino_adapter': 'dino_adapter', 'cgcd': 'cgcd',
               'stage0': 'checkpoints/stage0_epoch500.pth',
               'adapted': 'checkpoints/semi_supervised_epoch10.pth'}
    pending = []
    for name, destination in mapping.items():
        source = getattr(args, name)
        if source is None:
            continue
        source = source.expanduser().resolve(strict=True)
        target = ROOT / 'assets' / destination
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            parser.error(f'{target} already exists; refusing to replace it')
        pending.append((source, target))
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())
        print(f'Linked {target.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
