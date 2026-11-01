import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from unittest.mock import patch

from PIL import Image, ImageChops

from preprocess_unpaired_patches import main, patch_positions, patchify_directory


class PreprocessUnpairedPatchesTest(unittest.TestCase):
    def test_patch_positions_include_the_last_border(self):
        self.assertEqual(patch_positions(500, patch_size=224, stride=224), [0, 224, 276])

    def test_patch_positions_reject_invalid_sizes_and_gaps(self):
        with self.assertRaisesRegex(ValueError, "patch_size must be positive"):
            patch_positions(500, patch_size=0, stride=1)
        with self.assertRaisesRegex(ValueError, "stride must be positive"):
            patch_positions(500, patch_size=224, stride=0)
        with self.assertRaisesRegex(ValueError, "full coverage"):
            patch_positions(500, patch_size=224, stride=225)

    def test_patchify_directory_saves_every_full_coverage_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            nested_input = input_root / "nested"
            nested_input.mkdir(parents=True)
            source_path = nested_input / "sample.png"
            source_image = Image.new("RGB", (500, 300))
            source_image.putdata(
                [(x % 256, y % 256, (x + y) % 256) for y in range(300) for x in range(500)]
            )
            source_image.save(source_path)

            summary = patchify_directory(
                input_root=input_root,
                output_root=output_root,
                patch_size=224,
                stride=224,
                num_workers=2,
                show_progress=False,
            )

            patch_paths = sorted((output_root / "nested").glob("*.png"))
            expected_positions = [(x, y) for y in [0, 76] for x in [0, 224, 276]]
            expected_names = {
                f"sample__id000000__y{y:05d}_x{x:05d}.png" for x, y in expected_positions
            }
            self.assertEqual(summary["source_images"], 1)
            self.assertEqual(summary["processed_images"], 1)
            self.assertEqual(summary["skipped_small_images"], 0)
            self.assertEqual(summary["failed_images"], 0)
            self.assertEqual(summary["generated_patches"], 6)
            self.assertEqual(len(patch_paths), 6)
            self.assertEqual({path.name for path in patch_paths}, expected_names)

            with Image.open(source_path) as saved_source:
                for x, y in expected_positions:
                    patch_path = output_root / "nested" / (
                        f"sample__id000000__y{y:05d}_x{x:05d}.png"
                    )
                    with Image.open(patch_path) as saved_patch:
                        expected_patch = saved_source.crop((x, y, x + 224, y + 224))
                        self.assertIsNone(ImageChops.difference(saved_patch, expected_patch).getbbox())

    def test_patchify_directory_skips_images_smaller_than_the_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            input_root.mkdir()
            Image.new("RGB", (200, 224), color=(10, 20, 30)).save(input_root / "small.png")

            summary = patchify_directory(
                input_root=input_root,
                output_root=output_root,
                patch_size=224,
                stride=224,
                num_workers=1,
                show_progress=False,
            )

            self.assertEqual(summary["processed_images"], 0)
            self.assertEqual(summary["skipped_small_images"], 1)
            self.assertEqual(summary["generated_patches"], 0)
            self.assertEqual(list(output_root.rglob("*.png")), [])

    def test_patchify_directory_cleans_partial_files_after_save_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            nested_input = input_root / "nested"
            nested_input.mkdir(parents=True)
            Image.new("RGB", (448, 224)).save(nested_input / "source.png")

            original_save = Image.Image.save
            save_calls = 0

            def fail_second_save(image, output_path, *args, **kwargs):
                nonlocal save_calls
                save_calls += 1
                if save_calls == 2:
                    raise OSError("injected save failure")
                return original_save(image, output_path, *args, **kwargs)

            with patch.object(Image.Image, "save", new=fail_second_save):
                summary = patchify_directory(
                    input_root=input_root,
                    output_root=output_root,
                    patch_size=224,
                    stride=224,
                    num_workers=1,
                    show_progress=False,
                )

            self.assertEqual(summary["failed_images"], 1)
            self.assertEqual(summary["generated_patches"], 0)
            self.assertEqual(list(output_root.rglob("*.png")), [])
            self.assertEqual(list(output_root.iterdir()), [])

    def test_patchify_directory_rejects_a_nonempty_output_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            input_root.mkdir()
            output_root.mkdir()
            Image.new("RGB", (224, 224)).save(input_root / "source.png")
            (output_root / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not empty"):
                patchify_directory(input_root=input_root, output_root=output_root)

    def test_patchify_directory_excludes_manifest_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            input_root.mkdir()
            Image.new("RGB", (224, 224), color=(10, 20, 30)).save(input_root / "train.png")
            Image.new("RGB", (224, 224), color=(40, 50, 60)).save(input_root / "held_out.png")
            manifest_path = input_root / "test_data_info.txt"
            manifest_path.write_text("# held-out test image\nheld_out.png\n", encoding="utf-8")

            summary = patchify_directory(
                input_root=input_root,
                output_root=output_root,
                patch_size=224,
                stride=224,
                num_workers=1,
                show_progress=False,
                exclude_manifest=manifest_path,
            )

            patch_paths = list(output_root.glob("*.png"))
            self.assertEqual(summary["source_images"], 1)
            self.assertEqual(summary["excluded_images"], 1)
            self.assertEqual(summary["generated_patches"], 1)
            self.assertEqual(len(patch_paths), 1)
            self.assertTrue(patch_paths[0].name.startswith("train__id"))

    def test_patchify_directory_rejects_missing_manifest_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            input_root.mkdir()
            Image.new("RGB", (224, 224)).save(input_root / "train.png")
            manifest_path = input_root / "test_data_info.txt"
            manifest_path.write_text("missing.png\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not found under input_root"):
                patchify_directory(
                    input_root=input_root,
                    output_root=output_root,
                    exclude_manifest=manifest_path,
                )

    def test_cli_exits_nonzero_when_an_image_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            input_root.mkdir()
            (input_root / "corrupt.png").write_bytes(b"not an image")

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--input_root",
                        str(input_root),
                        "--output_root",
                        str(output_root),
                        "--patch_size",
                        "224",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
