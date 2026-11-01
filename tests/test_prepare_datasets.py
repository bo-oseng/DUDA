import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_datasets as prepare


class PrepareDatasetsTest(unittest.TestCase):
    def test_layout_and_class_local_patch_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare.create_layout(root)
            for name in prepare.PAIRED_CLASSES:
                self.assertTrue((root / "syn_train_dataset" / name / "gt").is_dir())
            for name in prepare.REAL_CLASSES:
                Image.new("RGB", (448, 224)).save(root / "real_train_dataset" / name / "sample.jpg")
            summaries = prepare.prepare_patches(root, num_workers=1)
            for name in prepare.REAL_CLASSES:
                paths = sorted((root / "real_train_patches" / name).glob("*.png"))
                self.assertEqual(summaries[name]["generated_patches"], 2)
                self.assertEqual(paths[0].name, "sample__id000000__y00000_x00000.png")
                with Image.open(paths[0]) as image:
                    self.assertEqual(image.size, (224, 224))

    def test_preflight_preserves_existing_patches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare.create_layout(root)
            for name in prepare.REAL_CLASSES:
                Image.new("RGB", (224, 224)).save(root / "real_train_dataset" / name / "sample.png")
            existing = root / "real_train_patches" / prepare.REAL_CLASSES[-1] / "keep.png"
            existing.write_bytes(b"original patch")
            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare.prepare_patches(root, num_workers=1)
            self.assertEqual(existing.read_bytes(), b"original patch")
            self.assertEqual(list((root / "real_train_patches" / prepare.REAL_CLASSES[0]).iterdir()), [])

    def test_monitoring_subset_uses_relative_links_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = root / "code/datasets/manifests/real_eval_small"
            manifests.mkdir(parents=True)
            for name in prepare.EVAL_SETS:
                (manifests / f"{name}.txt").write_text("keep.png\n")
                source = root / "data/real_eval_dataset" / name
                source.mkdir(parents=True)
                Image.new("RGB", (224, 224)).save(source / "keep.png")
                Image.new("RGB", (224, 224)).save(source / "other.png")
            with patch.object(prepare, "ROOT", root / "code"):
                prepare.prepare_eval_subset(root / "data")
                prepare.prepare_eval_subset(root / "data")
            for name in prepare.EVAL_SETS:
                target = root / "data/real_eval_small_dataset" / name / "keep.png"
                self.assertTrue(target.is_symlink())
                self.assertFalse(target.readlink().is_absolute())
                self.assertEqual(target.resolve(), root / "data/real_eval_dataset" / name / "keep.png")
                self.assertEqual(len(list(target.parent.iterdir())), 1)

    def test_monitoring_missing_source_fails_before_linking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = root / "code/datasets/manifests/real_eval_small"
            manifests.mkdir(parents=True)
            for name in prepare.EVAL_SETS:
                (manifests / f"{name}.txt").write_text("missing.png\n")
            with patch.object(prepare, "ROOT", root / "code"):
                with self.assertRaisesRegex(ValueError, "missing from full"):
                    prepare.prepare_eval_subset(root / "data")
            self.assertFalse((root / "data/real_eval_small_dataset").exists())


if __name__ == "__main__":
    unittest.main()
