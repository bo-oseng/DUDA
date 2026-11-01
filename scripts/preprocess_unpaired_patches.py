#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PatchTask:
    source_path: str
    relative_path: str
    source_index: int
    output_root: str
    patch_size: int
    stride: int


def list_images(input_root):
    input_root = Path(input_root)
    return sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_exclude_manifest(input_root, manifest_path):
    if manifest_path is None:
        return set()

    input_root = Path(input_root)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ValueError(f"Exclude manifest does not exist: {manifest_path}")

    excluded_paths = set()
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue

        relative_path = Path(value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe path in exclude manifest at line {line_number}: {value}")

        source_path = input_root / relative_path
        if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Excluded image from manifest line {line_number} was not found under input_root: {value}"
            )
        excluded_paths.add(relative_path.as_posix())

    return excluded_paths


def patch_positions(length, patch_size, stride):
    """Return grid positions, adding a final overlapping position for border coverage."""
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > patch_size:
        raise ValueError(
            f"stride must be less than or equal to patch_size for full coverage, got {stride} > {patch_size}"
        )
    if length < patch_size:
        return []

    positions = list(range(0, length - patch_size + 1, stride))
    end_position = length - patch_size
    if positions[-1] != end_position:
        positions.append(end_position)
    return positions


def _patchify_image(task):
    source_path = Path(task.source_path)
    relative_path = Path(task.relative_path)
    output_dir = Path(task.output_root) / relative_path.parent
    written_paths = []

    try:
        with Image.open(source_path) as opened_image:
            image = opened_image if opened_image.mode == "RGB" else opened_image.convert("RGB")
            width, height = image.size
            xs = patch_positions(width, task.patch_size, task.stride)
            ys = patch_positions(height, task.patch_size, task.stride)

            if not xs or not ys:
                return {
                    "status": "skipped_small",
                    "source_path": str(source_path),
                    "patch_count": 0,
                    "error": None,
                }

            output_dir.mkdir(parents=True, exist_ok=True)
            patch_count = 0
            prefix = f"{source_path.stem}__id{task.source_index:06d}"
            for y in ys:
                for x in xs:
                    patch = image.crop((x, y, x + task.patch_size, y + task.patch_size))
                    patch_name = f"{prefix}__y{y:05d}_x{x:05d}.png"
                    output_path = output_dir / patch_name
                    written_paths.append(output_path)
                    patch.save(output_path)
                    patch_count += 1

        return {
            "status": "processed",
            "source_path": str(source_path),
            "patch_count": patch_count,
            "error": None,
        }
    except Exception as exc:
        cleanup_errors = []
        for output_path in written_paths:
            try:
                output_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{output_path}: {cleanup_exc}")

        output_root = Path(task.output_root)
        empty_dir = output_dir
        while empty_dir != output_root:
            try:
                empty_dir.rmdir()
            except OSError:
                break
            empty_dir = empty_dir.parent

        error = str(exc)
        if cleanup_errors:
            error = f"{error}; cleanup failed: {'; '.join(cleanup_errors)}"
        return {
            "status": "failed",
            "source_path": str(source_path),
            "patch_count": 0,
            "error": error,
        }


def _validate_paths(input_root, output_root):
    if not input_root.is_dir():
        raise ValueError(f"Input folder does not exist: {input_root}")

    resolved_input = input_root.resolve()
    resolved_output = output_root.resolve()
    if resolved_output == resolved_input or resolved_input in resolved_output.parents:
        raise ValueError("output_root must be outside input_root")

    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            f"Output folder already exists and is not empty: {output_root}. " "Choose an empty output folder."
        )


def patchify_directory(
    input_root,
    output_root,
    patch_size=224,
    stride=None,
    num_workers=1,
    show_progress=True,
    exclude_manifest=None,
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    patch_size = int(patch_size)
    stride = patch_size if stride is None else int(stride)
    num_workers = int(num_workers)

    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > patch_size:
        raise ValueError(
            f"stride must be less than or equal to patch_size for full coverage, got {stride} > {patch_size}"
        )
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    _validate_paths(input_root, output_root)
    excluded_paths = read_exclude_manifest(input_root, exclude_manifest)
    source_images = [
        path
        for path in list_images(input_root)
        if path.relative_to(input_root).as_posix() not in excluded_paths
    ]
    if not source_images:
        raise ValueError(f"No images remaining after exclusions under: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        PatchTask(
            source_path=str(source_path),
            relative_path=source_path.relative_to(input_root).as_posix(),
            source_index=source_index,
            output_root=str(output_root),
            patch_size=patch_size,
            stride=stride,
        )
        for source_index, source_path in enumerate(source_images)
    ]

    summary = {
        "source_images": len(source_images),
        "excluded_images": len(excluded_paths),
        "processed_images": 0,
        "skipped_small_images": 0,
        "failed_images": 0,
        "generated_patches": 0,
        "failures": [],
    }

    def collect_results(results):
        iterator = tqdm(results, total=len(tasks), desc="Extracting unpaired patches", disable=not show_progress)
        for result in iterator:
            if result["status"] == "processed":
                summary["processed_images"] += 1
                summary["generated_patches"] += result["patch_count"]
            elif result["status"] == "skipped_small":
                summary["skipped_small_images"] += 1
            else:
                summary["failed_images"] += 1
                summary["failures"].append({"source_path": result["source_path"], "error": result["error"]})
    if num_workers == 1:
        collect_results(map(_patchify_image, tasks))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            collect_results(executor.map(_patchify_image, tasks, chunksize=1))

    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract every fixed-size grid patch from unpaired images without resizing, "
            "filtering, or a per-image patch limit."
        )
    )
    parser.add_argument("--input_root", type=str, required=True, help="Folder containing source images")
    parser.add_argument("--output_root", type=str, required=True, help="Empty folder for output patches")
    parser.add_argument("--patch_size", type=int, default=224, help="Square patch size (default: 224)")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Grid stride (default: patch_size). The last border position is always included.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of image-processing worker processes (default: 1)",
    )
    parser.add_argument(
        "--exclude_manifest",
        type=str,
        default=None,
        help="Optional text file of input-root-relative image paths to exclude",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    stride = args.patch_size if args.stride is None else args.stride

    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {stride}")
    print(f"Workers: {args.num_workers}")
    print(f"Exclude manifest: {args.exclude_manifest or '<none>'}")

    summary = patchify_directory(
        input_root=args.input_root,
        output_root=args.output_root,
        patch_size=args.patch_size,
        stride=stride,
        num_workers=args.num_workers,
        exclude_manifest=args.exclude_manifest,
    )

    print("\nPatch extraction complete")
    print(f"Source images: {summary['source_images']}")
    print(f"Excluded images: {summary['excluded_images']}")
    print(f"Processed images: {summary['processed_images']}")
    print(f"Skipped small images: {summary['skipped_small_images']}")
    print(f"Failed images: {summary['failed_images']}")
    print(f"Generated patches: {summary['generated_patches']}")
    if summary["failures"]:
        print("Failures:")
        for failure in summary["failures"][:10]:
            print(f"  - {failure['source_path']}: {failure['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
