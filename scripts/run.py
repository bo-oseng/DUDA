"""Launch the published DUDA training and evaluation stages."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
from string import Template
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def settings(stage):
    def path(name, default):
        return str(Path(os.environ.get(name, str(default))).expanduser().absolute())
    output = Path(path('OUTPUT_ROOT', ROOT / 'outputs'))
    feature_root = Path(path('FEATURE_ROOT', ROOT / 'assets/features'))
    return {
        'OUTPUT_ROOT': str(output),
        'DATA_ROOT': path('DATA_ROOT', ROOT / 'assets/data'),
        'FEATURE_ROOT': str(feature_root),
        'FEATURE_DIR': path('FEATURE_DIR', feature_root / 'wres_with_clear/dinov3-vitl16-sl'),
        'DINO_ADAPTER_DIR': path('DINO_ADAPTER_DIR', ROOT / 'assets/dino_adapter'),
        'CGCD_DIR': path('CGCD_DIR', ROOT / 'assets/cgcd'),
        'STAGE0_CHECKPOINT': path('STAGE0_CHECKPOINT', ROOT / 'assets/checkpoints/stage0_epoch500.pth'),
        'CHECKPOINT': path('CHECKPOINT', ROOT / 'assets/checkpoints/semi_supervised_epoch10.pth'),
        'RESULTS_ROOT': path('RESULTS_ROOT', output / 'results/table1'),
        'EXP_NAME': os.environ.get('EXP_NAME', 'supervised' if stage == 'supervised' else 'semi_supervised'),
    }


def render_config(stage, values):
    name = 'supervised' if stage == 'supervised' else 'semi_supervised'
    config = yaml.safe_load((ROOT / 'configs' / (name + '.yml')).read_text())
    def expand(value):
        if isinstance(value, str):
            return Template(value).substitute(values)
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value
    config = expand(config)
    if stage == 'supervised' and os.environ.get('RESUME_CHECKPOINT'):
        config['train']['resume'] = str(Path(os.environ['RESUME_CHECKPOINT']).absolute())
    return config


def prepare_feature_view(data_root, destination):
    """Keep generated split manifests local while sharing image storage."""
    for split in ('syn_train_dataset', 'real_train_dataset'):
        source = data_root / split
        for class_dir in sorted(source.iterdir()):
            if not class_dir.is_dir():
                continue
            target = destination / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for item in class_dir.iterdir():
                dest = target / item.name
                if dest.exists() or dest.is_symlink():
                    continue
                if item.is_file() and item.suffix in ('.txt', '.json', '.jsonl'):
                    shutil.copyfile(item, dest)
                else:
                    dest.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'cgcd', 'supervised', 'semi_supervised', 'evaluate', 'qalign'])
    parser.add_argument('--dry-run', action='store_true', help='Print commands and configuration without writing or launching')
    args, extra = parser.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    v = settings(args.stage)
    if Path(v['EXP_NAME']).name != v['EXP_NAME'] or v['EXP_NAME'] in ('.', '..'):
        parser.error('EXP_NAME must be a single directory name')
    output = Path(v['OUTPUT_ROOT'])
    config_path = output / 'configs' / (v['EXP_NAME'] + '.yml')
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env['PYTHONUNBUFFERED'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    env.setdefault('TMPDIR', str(output / 'tmp'))
    env.setdefault('BASE_EXT_DIR', str(output / 'torch_extensions/rank'))
    # Both backbones resolve the same configured local base model.
    env.setdefault('DUDA_DINO_BASE', str(ROOT / 'assets/dino_base'))
    nproc = env.get('NPROC_PER_NODE', '3')
    python = env.get('QALIGN_PYTHON', sys.executable) if args.stage == 'qalign' else sys.executable
    distributed = [python, '-m', 'torch.distributed.run', '--nproc_per_node', nproc]
    if env.get('MASTER_PORT'):
        distributed += ['--master_port', env['MASTER_PORT']]
    else:
        distributed += ['--standalone']
    commands = []
    cwd = ROOT / 'restoration'
    cfg = None
    if args.stage == 'features':
        cwd = ROOT / 'vb_cgcd'
        generated = output / 'features'
        view = output / 'feature_data'
        commands = [distributed + ['feature_extractor/dino_wrevlm_my.py', '--model', 'dinov3_vitl16',
            '--exp_name', 'wres_with_clear', '--batch_size', '128', '--seed', '42', '--epochs', '10',
            '--train_root', str(view / 'syn_train_dataset'), '--output_dir', str(generated),
            '--max_samples_per_class', '900', '--test_split', '0.1'],
            [sys.executable, 'feature_extractor/append_wres_real_features.py', '--exp_name', 'wres_with_clear',
             '--output_dir', str(generated), '--pretrained_model_name', 'dinov3-vitl16-sl',
             '--real_root', str(view / 'real_train_dataset'), '--real_classes', 'RainReal', 'SnowReal',
             'UnannotatedHazyImages', '--real_start_label', '6', '--test_split', '0.1']]
    elif args.stage == 'cgcd':
        cwd = ROOT / 'vb_cgcd'
        commands = [[sys.executable, 'main_my_wres_schedule.py', '--base', '6', '--num_classes', '9',
            '--increment_schedule', '6,3', '--exp_name', 'wres_with_clear',
            '--pretrained_model_name', 'dinov3-vitl16-sl', '--dataset', 'wresvlm',
            '--data_dir', v['FEATURE_ROOT'], '--classifier_alg', 'mngmm_wresvlm',
            '--trail_name', 'cgcd_6p3', '--load_mode', 't1', '--use_correct_scaling_factor',
            '--real_data_dir', str(output / 'feature_data/real_train_dataset'), '--real_test_split', '0.1',
            '--output_dir', str(output / 'cgcd')]]
    elif args.stage in ('supervised', 'semi_supervised', 'evaluate'):
        cfg = render_config(args.stage, v)
        if args.stage == 'supervised':
            commands = [distributed + ['train_wres_dataset_adain.py', '--config', str(config_path), '--exp_name', v['EXP_NAME']]]
        elif args.stage == 'semi_supervised':
            command = distributed + ['train_wres_dataset_continual_semiuir_unlabeled_patches.py',
                '--config', str(config_path), '--exp_name', v['EXP_NAME'], '--stage', '1',
                '--base_class_num', '6', '--inc_class_num', '3', '--cgcd_score_mode', 'anchor',
                '--pseudo_update_mode', 'musiq_then_cgcd', '--skip_pseudo_init']
            if env.get('RESUME_CHECKPOINT'):
                command += ['--resume', str(Path(env['RESUME_CHECKPOINT']).absolute())]
            commands = [command]
        else:
            commands = [distributed + ['eval_onerestore_real.py', '--config', str(config_path),
                '--checkpoint', v['CHECKPOINT'], '--eval_model', 'teacher', '--stage', '1',
                '--output', v['RESULTS_ROOT'], '--save_image', '--labeled_eval',
                '--real_eval_root', v['DATA_ROOT'] + '/real_eval_dataset',
                '--real_eval_sets', 'RealRain_2320,RTTS,Snow100K_R', '--max_eval_side', '0']]
    else:
        commands = [distributed + ['eval_onerestore_qaling.py', '--results_root', v['RESULTS_ROOT'],
            '--model_tag', 'teacher', '--sets', 'RealRain_2320,RTTS,Snow100K_R', '--suffix', 'saved_qalign']]
    commands[-1] += extra
    print('Working directory:', cwd)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        if cfg:
            print(yaml.safe_dump(cfg, sort_keys=False))
        return
    for directory in (output, Path(env['TMPDIR']), output / 'torch_extensions', output / 'logs'):
        directory.mkdir(parents=True, exist_ok=True)
    if args.stage in ('features', 'cgcd'):
        prepare_feature_view(Path(v['DATA_ROOT']), output / 'feature_data')
    if cfg:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if args.stage == 'semi_supervised' and not env.get('RESUME_CHECKPOINT'):
        bank = Path(cfg['train']['pseudo_patches_dir'])
        if bank.exists() or bank.is_symlink():
            raise FileExistsError(f'Pseudo bank already exists: {bank}. Choose a new EXP_NAME or resume the run.')
        print('Initializing writable pseudo bank:', bank, flush=True)
        shutil.copytree(cfg['train']['lq_patches_dir'], bank)
    for command in commands:
        subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == '__main__':
    main()
