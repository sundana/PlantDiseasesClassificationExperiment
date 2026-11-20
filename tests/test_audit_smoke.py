"""CPU-only smoke tests for the Path B audit utilities.

Run from the repository root:
    python -m unittest discover -s tests -v

The model/training code needs torch and is covered by the runbook, not here.
These tests cover the pieces where a silent bug would invalidate a table:
metrics, VOC parsing, cropping and the border probe geometry.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from src.audit import (
    BorderMask,
    classification_metrics,
    crop_to_boxes,
    index_voc_annotations,
    lookup_annotation,
    mcnemar_exact,
    paired_bootstrap_difference,
    parse_voc_boxes,
    tost_equivalence,
)


class TestMetrics(unittest.TestCase):
    def test_hand_example(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "B", "B", "C"]
        result = classification_metrics(y_true, y_pred, ["A", "B"])
        self.assertAlmostEqual(result["accuracy"], 50.0)
        self.assertAlmostEqual(result["precision"], 75.0)
        self.assertAlmostEqual(result["recall"], 50.0)
        self.assertAlmostEqual(result["f1"], (2 / 3 + 0.5) / 2 * 100, places=6)
        self.assertAlmostEqual(result["unmapped_rate"], 25.0)
        self.assertAlmostEqual(result["background_rate"], 0.0)
        self.assertEqual(result["per_class"]["A"]["support"], 2)

    def test_matches_repository_compute_metrics(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            import compute_metrics
        except ImportError:
            self.skipTest("compute_metrics.py is not present in this checkout")

        y_true = ["A", "A", "B", "B", "C"]
        y_pred = ["A", "B", "B", "C", "C"]
        ours = classification_metrics(y_true, y_pred, ["A", "B", "C"])
        theirs = compute_metrics.macro_metrics(y_true, y_pred, ["A", "B", "C"])
        self.assertAlmostEqual(ours["accuracy"], theirs["accuracy"], places=9)
        self.assertAlmostEqual(ours["precision"], theirs["precision"], places=9)
        self.assertAlmostEqual(ours["recall"], theirs["recall"], places=9)
        self.assertAlmostEqual(ours["f1"], theirs["f1"], places=9)

    def test_rejects_unknown_true_label(self):
        with self.assertRaises(ValueError):
            classification_metrics(["Z"], ["A"], ["A"])


class TestMcNemar(unittest.TestCase):
    def test_known_value(self):
        # 10 discordant pairs in one direction, 2 in the other.
        y_true = ["A"] * 12
        pred_a = ["B"] * 10 + ["A"] * 2   # A wrong 10 times
        pred_b = ["A"] * 10 + ["B"] * 2   # B wrong 2 times
        result = mcnemar_exact(y_true, pred_a, pred_b)
        self.assertEqual(result["n01"], 10)
        self.assertEqual(result["n10"], 2)
        self.assertAlmostEqual(result["p_value"], 158 / 4096, places=6)

    def test_identical_predictions(self):
        result = mcnemar_exact(["A", "B"], ["A", "B"], ["A", "B"])
        self.assertEqual(result["p_value"], 1.0)


class TestBootstrap(unittest.TestCase):
    def test_zero_difference_for_identical_models(self):
        y_true = ["A", "B", "A", "B"] * 10
        pred = list(y_true)
        result = paired_bootstrap_difference(y_true, pred, pred, n_bootstrap=200, seed=1)
        self.assertEqual(result["mean_diff_pp"], 0.0)
        self.assertLessEqual(result["ci_low_pp"], 0.0)
        self.assertGreaterEqual(result["ci_high_pp"], 0.0)

    def test_tost_equivalent_for_identical_models(self):
        y_true = ["A", "B"] * 50
        pred = list(y_true)
        result = tost_equivalence(y_true, pred, pred, margin_pp=1.0, n_bootstrap=200)
        self.assertTrue(result["equivalent"])


class TestBorderMask(unittest.TestCase):
    def test_geometry(self):
        image = Image.new("RGB", (100, 80), (10, 20, 30))
        masked = BorderMask(fraction=0.2, fill=(1, 2, 3))(image)
        self.assertEqual(masked.size, (100, 80))
        self.assertEqual(masked.getpixel((0, 0)), (10, 20, 30))       # corner kept
        self.assertEqual(masked.getpixel((50, 40)), (1, 2, 3))        # centre filled
        self.assertEqual(masked.getpixel((19, 40)), (10, 20, 30))     # inside frame

    def test_zero_fraction_is_identity(self):
        image = Image.new("RGB", (10, 10), (5, 5, 5))
        self.assertIs(BorderMask(fraction=0.0)(image), image)


class TestAnnotations(unittest.TestCase):
    def _write_xml(self, path, boxes):
        objects = "".join(
            f"<object><name>leaf</name><bndbox>"
            f"<xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax>"
            f"</bndbox></object>"
            for x0, y0, x1, y1 in boxes
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"<annotation>{objects}</annotation>")

    def test_parse_and_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "leaf_1.xml")
            self._write_xml(xml, [(10, 10, 40, 40), (50, 20, 90, 70)])
            boxes = parse_voc_boxes(xml)
            self.assertEqual(len(boxes), 2)
            image = Image.new("RGB", (100, 80), (0, 0, 0))
            crop, used = crop_to_boxes(image, boxes, margin=0.0)
            self.assertEqual(crop.size, (80, 60))
            self.assertEqual(used, (10, 10, 90, 70))

    def test_index_and_coll_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "Tomato leaf_12.xml")
            self._write_xml(xml, [(1, 1, 5, 5)])
            index = index_voc_annotations([tmp])
            self.assertIn("Tomato leaf_12", index)
            self.assertEqual(lookup_annotation("Tomato leaf_12_coll", index), xml)

    def test_invalid_boxes_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = os.path.join(tmp, "x.xml")
            self._write_xml(xml, [(10, 10, 10, 20), (0, 0, 20, 20)])
            self.assertEqual(len(parse_voc_boxes(xml)), 2)
            crop, used = crop_to_boxes(
                Image.new("RGB", (40, 40)), parse_voc_boxes(xml), margin=0.0
            )
            self.assertEqual(used, (0, 0, 20, 20))


class TestPlantDocDataset(unittest.TestCase):
    def test_crop_dataset_end_to_end(self):
        from src.audit import PlantDocCropDataset, collect_plantdoc_samples

        with tempfile.TemporaryDirectory() as tmp:
            plantdoc = os.path.join(tmp, "plantdoc")
            class_dir = os.path.join(plantdoc, "Apple leaf")
            os.makedirs(class_dir)
            Image.new("RGB", (200, 100), (0, 100, 0)).save(
                os.path.join(class_dir, "img1.jpg")
            )
            ann_dir = os.path.join(tmp, "annotations")
            os.makedirs(ann_dir)
            xml = os.path.join(ann_dir, "img1.xml")
            with open(xml, "w", encoding="utf-8") as handle:
                handle.write(
                    "<annotation><object><bndbox>"
                    "<xmin>20</xmin><ymin>10</ymin><xmax>100</xmax><ymax>80</ymax>"
                    "</bndbox></object></annotation>"
                )
            index = index_voc_annotations([ann_dir])
            samples = collect_plantdoc_samples(
                plantdoc, annotation_index=index, verbose=False
            )
            self.assertEqual(len(samples), 1)
            dataset = PlantDocCropDataset(samples, transform=None, margin=0.0)
            full, crop, label, sample_index, has_annotation, n_boxes = dataset[0]
            self.assertEqual(full.size, (200, 100))
            self.assertEqual(crop.size, (80, 70))
            self.assertEqual(sample_index, 0)
            self.assertEqual(has_annotation, 1)
            self.assertEqual(n_boxes, 1)


    def test_split_layout_resolves_duplicate_stems(self):
        from src.audit import collect_plantdoc_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "plantdoc")
            for split in ("train", "test"):
                class_dir = os.path.join(root, split, "Apple leaf")
                os.makedirs(class_dir)
                Image.new("RGB", (100, 60), (0, 100, 0)).save(
                    os.path.join(class_dir, "same.jpg")
                )
            ann_root = os.path.join(tmp, "annotations")
            for split, x1 in (("TRAIN", 30), ("TEST", 90)):
                split_dir = os.path.join(ann_root, split)
                os.makedirs(split_dir)
                with open(os.path.join(split_dir, "same.xml"), "w", encoding="utf-8") as handle:
                    handle.write(
                        "<annotation><object><bndbox>"
                        f"<xmin>0</xmin><ymin>0</ymin><xmax>{x1}</xmax><ymax>50</ymax>"
                        "</bndbox></object></annotation>"
                    )
            index = index_voc_annotations([ann_root])
            samples = collect_plantdoc_samples(
                root, annotation_index=index, verbose=False
            )
            self.assertEqual(len(samples), 2)
            by_split = {sample["split"]: sample for sample in samples}
            self.assertEqual(parse_voc_boxes(by_split["train"]["annotation"])[0][2], 30)
            self.assertEqual(parse_voc_boxes(by_split["test"]["annotation"])[0][2], 90)


if __name__ == "__main__":
    unittest.main()
