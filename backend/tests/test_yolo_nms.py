import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.batch import BatchRunner
from app.dataset import Dataset
from app.nms import nms_by_class
from app.roi import apply_table_rules
from app.yolo import boxes_to_yolo, classes_match, classes_text, yolo_to_boxes


class YoloTests(unittest.TestCase):
    def test_round_trip(self):
        boxes = [{"class_id": 0, "x1": 10, "y1": 20, "x2": 30, "y2": 40}]
        text = boxes_to_yolo(boxes, 100, 50)
        self.assertEqual(text, "0 0.200000 0.600000 0.200000 0.400000\n")
        restored = yolo_to_boxes(text, 100, 50)
        self.assertEqual(restored[0]["class_id"], 0)
        self.assertAlmostEqual(restored[0]["x1"], 10)
        self.assertAlmostEqual(restored[0]["y2"], 40)

    def test_classes_file(self):
        self.assertTrue(classes_match(classes_text()))
        self.assertFalse(classes_match("bag\nbox\nirregular\n"))

    def test_rejects_rotated_line(self):
        with self.assertRaises(ValueError):
            yolo_to_boxes("0 0.1 0.1 0.2 0.2 0.3\n", 100, 100)


class NmsTests(unittest.TestCase):
    def test_nested_robot_and_item_both_stay(self):
        boxes = [
            {"class_id": 1, "x1": 0, "y1": 0, "x2": 100, "y2": 100, "score": 0.9},
            {"class_id": 0, "x1": 10, "y1": 10, "x2": 30, "y2": 40, "score": 0.8},
        ]
        kept = nms_by_class(boxes, 0.5)
        self.assertEqual(len(kept), 2)

    def test_same_class_overlap_keeps_higher_score(self):
        boxes = [
            {"class_id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.4},
            {"class_id": 0, "x1": 1, "y1": 1, "x2": 11, "y2": 11, "score": 0.9},
        ]
        kept = nms_by_class(boxes, 0.5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["score"], 0.9)


class TableRuleTests(unittest.TestCase):
    top = [[0.0, 0.10], [1.0, 0.10]]
    bottom = [[0.0, 0.90], [1.0, 0.90]]

    def test_keeps_item_between_the_table_edges(self):
        boxes = [{"class_id": 0, "x1": 40, "y1": 40, "x2": 70, "y2": 70, "score": 0.8}]
        kept = apply_table_rules(boxes, 100, 100, self.top, self.bottom, 0.001)
        self.assertEqual(len(kept), 1)

    def test_drops_boxes_outside_the_red_lines(self):
        boxes = [
            {"class_id": 0, "x1": 40, "y1": 0, "x2": 70, "y2": 8, "score": 0.8},
            {"class_id": 1, "x1": 10, "y1": 92, "x2": 40, "y2": 100, "score": 0.9},
        ]
        kept = apply_table_rules(boxes, 100, 100, self.top, self.bottom, 0.0)
        self.assertEqual(kept, [])

    def test_drops_tile_joint_sized_items_and_keeps_a_hand(self):
        boxes = [
            {"class_id": 0, "x1": 10, "y1": 40, "x2": 12, "y2": 42, "score": 0.8},
            {"class_id": 2, "x1": 10, "y1": 40, "x2": 12, "y2": 42, "score": 0.8},
        ]
        kept = apply_table_rules(boxes, 100, 100, self.top, self.bottom, 0.001)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["class_id"], 2)

    def test_item_box_on_the_robot_body_is_dropped(self):
        boxes = [
            {"class_id": 1, "x1": 0, "y1": 0, "x2": 100, "y2": 100, "score": 0.8},
            {"class_id": 0, "x1": 4, "y1": 4, "x2": 96, "y2": 96, "score": 0.6},
            {"class_id": 0, "x1": 40, "y1": 40, "x2": 55, "y2": 55, "score": 0.9},
        ]
        kept = apply_table_rules(boxes, 100, 100, self.top, self.bottom, 0.001)
        self.assertEqual([box["class_id"] for box in kept], [1, 0])
        self.assertEqual(kept[1]["score"], 0.9)

    def test_drops_a_short_robot_box_on_the_edge_light(self):
        boxes = [
            {"class_id": 1, "x1": 10, "y1": 30, "x2": 40, "y2": 60, "score": 0.9},
            {"class_id": 1, "x1": 70, "y1": 10, "x2": 95, "y2": 22, "score": 0.65},
        ]
        kept = apply_table_rules(boxes, 100, 100, self.top, self.bottom, 0.0, min_robot_side=0.18)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["score"], 0.9)


class DatasetTests(unittest.TestCase):
    def test_flat_folder_and_images_subdir(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (40, 20), "white").save(root / "a.jpg")
            opened = Dataset()
            opened.open(folder)
            opened.write_labels("a.jpg", [{"class_id": 1, "x1": 1, "y1": 1, "x2": 10, "y2": 10}])
            self.assertTrue((root / "labels" / "a.txt").read_text(encoding="utf-8").startswith("1 "))
            self.assertEqual((root / "classes.txt").read_text(encoding="utf-8").split(), ["item", "robot", "hand"])

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            Image.new("RGB", (80, 40), "black").save(root / "images" / "b.png")
            opened = Dataset()
            opened.open(str(root / "images"))
            opened.write_labels("b.png", [{"class_id": 2, "x1": 0, "y1": 0, "x2": 8, "y2": 8}])
            self.assertTrue((root / "labels" / "b.txt").exists())
            self.assertTrue((root / "classes.txt").exists())


class PreannotationBatchTests(unittest.TestCase):
    def _image(self, folder: Path, name: str):
        Image.new("RGB", (40, 20), "white").save(folder / name)

    def _box(self):
        return [{"class_id": 0, "x1": 2, "y1": 2, "x2": 12, "y2": 12, "score": 0.8}]

    def test_one_click_skips_existing_labels_and_cancel_deletes_new_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._image(root, "new.jpg")
            self._image(root, "kept.jpg")
            opened = Dataset()
            opened.open(folder)
            opened.write_labels("kept.jpg", [{"class_id": 1, "x1": 1, "y1": 1, "x2": 10, "y2": 10}])
            original = (root / "labels" / "kept.txt").read_text(encoding="utf-8")
            self.assertEqual(opened.write_preannotation("kept.jpg", self._box()), "skipped")
            self.assertEqual((root / "labels" / "kept.txt").read_text(encoding="utf-8"), original)
            self.assertEqual(opened.write_preannotation("new.jpg", self._box()), "written")
            self.assertTrue((root / "labels" / "new.txt").exists())
            rows = {row["name"]: row for row in opened.images()}
            self.assertTrue(rows["kept.jpg"]["has_robot"])
            self.assertFalse(rows["new.jpg"]["has_robot"])
            result = opened.revert_preannotations()
            self.assertEqual(result, {"reverted": 1, "kept": 0})
            self.assertFalse((root / "labels" / "new.txt").exists())
            self.assertEqual((root / "labels" / "kept.txt").read_text(encoding="utf-8"), original)
            self.assertFalse(opened.batch_undo_available())

    def test_cancel_keeps_a_label_edited_after_the_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._image(root, "a.jpg")
            opened = Dataset()
            opened.open(folder)
            opened.write_preannotation("a.jpg", self._box())
            label = root / "labels" / "a.txt"
            label.write_text("2 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
            result = opened.revert_preannotations()
            self.assertEqual(result["kept"], 1)
            self.assertEqual(result["reverted"], 0)
            self.assertTrue(label.read_text(encoding="utf-8").startswith("2 "))

    def test_cancel_restores_a_previous_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._image(root, "a.jpg")
            opened = Dataset()
            opened.open(folder)
            label = root / "labels" / "a.txt"
            previous = "2 0.250000 0.250000 0.200000 0.200000\n"
            written = "0 0.500000 0.500000 0.200000 0.200000\n"
            label.write_text(written, encoding="utf-8")
            (root / "labels" / ".batch-undo.json").write_text(
                json.dumps({"files": {"a.jpg": {"previous": previous, "written": written}}}),
                encoding="utf-8",
            )
            result = opened.revert_preannotations()
            self.assertEqual(result["reverted"], 1)
            self.assertEqual(label.read_text(encoding="utf-8"), previous)

    def test_runner_skips_hold_and_reverts_when_cancelled_midway(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._image(root, "a.jpg")
            self._image(root, "b.jpg")
            self._image(root, "c.jpg")
            self._image(root, "d.jpg")
            opened = Dataset()
            opened.open(folder)
            opened.write_labels("c.jpg", [{"class_id": 2, "x1": 1, "y1": 1, "x2": 8, "y2": 8}])
            calls = {"n": 0}

            def detect(image, threshold, nms_iou):
                calls["n"] += 1
                if calls["n"] == 2:
                    runner.cancel()
                return [{"class_id": 0, "x1": 1, "y1": 1, "x2": 8, "y2": 8, "score": 0.4}]

            runner = BatchRunner(opened, detect)
            status = runner.start(0.3, 0.5, hold="d.jpg")
            self.assertTrue(status["running"])
            self.assertEqual(status["total"], 2)
            self.assertEqual(status["skipped"], 2)
            runner.wait(3)
            done = runner.status()
            self.assertFalse(done["running"])
            self.assertTrue(done["cancel_requested"])
            self.assertEqual(done["reverted"], 1)
            self.assertFalse((root / "labels" / "a.txt").exists())
            self.assertFalse((root / "labels" / "b.txt").exists())
            self.assertFalse((root / "labels" / "d.txt").exists())
            self.assertTrue((root / "labels" / "c.txt").exists())
            self.assertFalse(done["undo_available"])


if __name__ == "__main__":
    unittest.main()
