"""Launch the DUDA reproduction stages."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
from string import Template
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["features", "cgcd", "supervised", "semi_supervised", "evaluate", "qalign"]


def settings(stage):
    def path(name, default):
        return str(Path(os.environ.get(name, str(default))).expanduser().absolute())

    output = Path(path("OUTPUT_ROOT", ROOT / "outputs"))
    feature_root = Path(path("FEATURE_ROOT", ROOT / "assets/features"))
    stream = os.environ.get("CONTINUAL_STREAM")
    stage_num = int(os.environ.get("CONTINUAL_STAGE", "1"))
    if stream and (stream not in ("6p1p2", "6p1p1p1") or stage_num not in range(1, 3 if stream == "6p1p2" else 4)):
        raise ValueError("CONTINUAL_STREAM/STAGE must be 6p1p2:1-2 or 6p1p1p1:1-3")
    stream_assets = ROOT / "assets/continual" / stream if stream else None
    default_exp = f"{stream}_stage{stage_num}" if stream else ("supervised" if stage == "supervised" else "semi_supervised")
    return {
        "OUTPUT_ROOT": str(output),
        "DATA_ROOT": path("DATA_ROOT", ROOT / "datasets/wres_datasets"),
        "FEATURE_ROOT": str(feature_root),
        "FEATURE_DIR": path("FEATURE_DIR", feature_root / "wres_with_clear/dinov3-vitl16-sl"),
        "DINO_ADAPTER_DIR": path("DINO_ADAPTER_DIR", ROOT / "assets/dino_adapter"),
        "CGCD_DIR": path("CGCD_DIR", stream_assets / "cgcd" if stream else ROOT / "assets/cgcd"),
        "STAGE0_CHECKPOINT": path("STAGE0_CHECKPOINT", ROOT / "assets/checkpoints/stage0_latest.pth"),
        "CHECKPOINT": path("CHECKPOINT", stream_assets / "checkpoints" / f"stage{stage_num}_latest.pth" if stream else ROOT / "assets/checkpoints/semi_supervised_epoch10.pth"),
        "RESULTS_ROOT": path("RESULTS_ROOT", output / "results" / default_exp if stream else output / "results/table1"),
        "EXP_NAME": os.environ.get("EXP_NAME", default_exp),
        "CONTINUAL_STREAM": stream,
        "CONTINUAL_STAGE": stage_num,
    }


def render_config(stage, values):
    name = "supervised" if stage == "supervised" else "semi_supervised"
    config = yaml.safe_load((ROOT / "configs" / f"{name}.yml").read_text())

    def expand(value):
        if isinstance(value, str):
            return Template(value).substitute(values)
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value

    config = expand(config)
    if values["CONTINUAL_STREAM"]:
        if stage == "supervised":
            raise ValueError("Continual streams start from the existing supervised checkpoint")
        stream, stage_num = values["CONTINUAL_STREAM"], values["CONTINUAL_STAGE"]
        config["incremental"].update(stage=stage_num, base_class_num=6 + stage_num - 1,
                                     inc_class_num=2 if stream == "6p1p2" and stage_num == 2 else 1)
        if stage_num > 1:
            default_prev = Path(values["OUTPUT_ROOT"]) / "checkpoints" / f"{stream}_stage{stage_num - 1}" / f"stage{stage_num - 1}_latest.pth"
            config["train"]["previous_stage_checkpoint"] = str(Path(os.environ.get("PREVIOUS_STAGE_CHECKPOINT", str(default_prev))).absolute())
    if stage == "supervised" and os.environ.get("RESUME_CHECKPOINT"):
        config["train"]["resume"] = str(Path(os.environ["RESUME_CHECKPOINT"]).absolute())
    return config


def prepare_feature_view(data_root, destination):
    """Link the image folders and copy the split manifests, so that new manifests stay under outputs/."""
    for split in ("syn_train_dataset", "real_train_dataset"):
        for class_dir in sorted((data_root / split).iterdir()):
            if not class_dir.is_dir():
                continue
            target = destination / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for item in class_dir.iterdir():
                dest = target / item.name
                if dest.exists() or dest.is_symlink():
                    continue
                if item.is_file() and item.suffix == ".txt":
                    shutil.copyfile(item, dest)
                else:
                    dest.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--dry-run", action="store_true", help="Print the commands and configuration without running")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    v = settings(args.stage)
    if Path(v["EXP_NAME"]).name != v["EXP_NAME"] or v["EXP_NAME"] in (".", ".."):
        parser.error("EXP_NAME must be a single directory name")
    output = Path(v["OUTPUT_ROOT"])
    config_path = output / "configs" / f"{v['EXP_NAME']}.yml"

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TMPDIR", str(output / "tmp"))
    env.setdefault("BASE_EXT_DIR", str(output / "torch_extensions/rank"))  # BOFT CUDA extension, one per rank
    if (ROOT / "assets/dino_base").is_dir():
        env.setdefault("DUDA_DINO_BASE", str(ROOT / "assets/dino_base"))
    python = env.get("QALIGN_PYTHON", sys.executable) if args.stage == "qalign" else sys.executable
    torchrun = [python, "-m", "torch.distributed.run", "--nproc_per_node", env.get("NPROC_PER_NODE", "3")]
    torchrun += ["--master_port", env["MASTER_PORT"]] if env.get("MASTER_PORT") else ["--standalone"]

    cfg = None
    cwd = ROOT / "restoration"
    if args.stage == "features":
        cwd = ROOT / "discovery"
        view = output / "feature_data"
        commands = [
            torchrun
            + [
                "train_adapter.py",
                "--train_root",
                str(view / "syn_train_dataset"),
                "--output_dir",
                str(output / "features"),
            ],
            [
                sys.executable,
                "extract_real_features.py",
                "--feature_root",
                str(output / "features"),
                "--real_root",
                str(view / "real_train_dataset"),
            ],
        ]
    elif args.stage == "cgcd":
        cwd = ROOT / "discovery"
        commands = [
            [sys.executable, "run_vbcgcd.py", "--feature_root", v["FEATURE_ROOT"], "--output_dir", str(output / "cgcd")]
        ]
    elif args.stage == "supervised":
        cfg = render_config(args.stage, v)
        commands = [torchrun + ["train_supervised.py", "--config", str(config_path), "--exp_name", v["EXP_NAME"]]]
    elif args.stage == "semi_supervised":
        cfg = render_config(args.stage, v)
        command = torchrun + ["train_semi_supervised.py", "--config", str(config_path), "--exp_name", v["EXP_NAME"]]
        if env.get("RESUME_CHECKPOINT"):
            command += ["--resume", str(Path(env["RESUME_CHECKPOINT"]).absolute())]
        commands = [command]
    elif args.stage == "evaluate":
        cfg = render_config(args.stage, v)
        commands = [
            torchrun
            + [
                "evaluate.py",
                "--config",
                str(config_path),
                "--checkpoint",
                v["CHECKPOINT"],
                "--output",
                v["RESULTS_ROOT"],
            ]
        ]
    else:
        commands = [torchrun + ["evaluate_qalign.py", "--results_root", v["RESULTS_ROOT"]]]
    commands[-1] += extra

    print("Working directory:", cwd)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        if cfg:
            print(yaml.safe_dump(cfg, sort_keys=False))
        return

    for directory in (output, Path(env["TMPDIR"]), output / "torch_extensions"):
        directory.mkdir(parents=True, exist_ok=True)
    if args.stage == "features":
        prepare_feature_view(Path(v["DATA_ROOT"]), output / "feature_data")
    if cfg:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if args.stage == "semi_supervised" and not env.get("RESUME_CHECKPOINT"):
        bank = Path(cfg["train"]["reliable_bank_dir"])
        if bank.exists() or bank.is_symlink():
            raise FileExistsError(f"Reliable bank already exists: {bank}. Choose a new EXP_NAME or resume the run.")
        print("Initializing the reliable bank with the unpaired inputs:", bank, flush=True)
        shutil.copytree(cfg["train"]["unpaired_patches_dir"], bank)
    for command in commands:
        subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
