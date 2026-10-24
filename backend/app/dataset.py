"""Open a local image folder and the YOLO files beside it."""

import json
from pathlib import Path

from PIL import Image

from .yolo import boxes_to_yolo, classes_match, classes_text, yolo_to_boxes

UNDO_NAME = ".batch-undo.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Dataset:
    def __init__(self):
        self.root: Path | None = None
        self.image_dir: Path | None = None
        self.label_dir: Path | None = None
        self.classes_path: Path | None = None

    def open(self, raw_path: str):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"目录不存在：{path}")
        if (path / "images").is_dir():
            root = path
            image_dir = path / "images"
        elif path.name == "images":
            root = path.parent
            image_dir = path
        else:
            root = path
            image_dir = path
        label_dir = root / "labels"
        classes_path = root / "classes.txt"
        label_dir.mkdir(parents=True, exist_ok=True)
        if classes_path.exists():
            text = classes_path.read_text(encoding="utf-8")
            if not classes_match(text):
                raise ValueError("classes.txt 必须依次是 item、robot、hand")
        else:
            classes_path.write_text(classes_text(), encoding="utf-8")
        self.root = root
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.classes_path = classes_path
        return self.info()

    def info(self):
        if self.root is None:
            return {"open": False}
        return {
            "open": True,
            "root": str(self.root),
            "image_dir": str(self.image_dir),
            "label_dir": str(self.label_dir),
        }

    def require(self):
        if self.image_dir is None or self.label_dir is None:
            raise RuntimeError("还没有打开图片目录")

    def images(self):
        self.require()
        rows = []
        for path in sorted(self.image_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = self.label_dir / f"{path.stem}.txt"
            labeled = label_path.exists()
            robot = _text_has_robot(label_path.read_text(encoding="utf-8")) if labeled else False
            rows.append({"name": path.name, "labeled": labeled, "has_robot": robot})
        return rows

    def image_names(self):
        self.require()
        names = []
        for path in sorted(self.image_dir.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                names.append(path.name)
        return names

    def has_label(self, name: str) -> bool:
        return self.label_path(name).exists()

    def image_path(self, name: str) -> Path:
        self.require()
        path = (self.image_dir / Path(name).name).resolve()
        if path.parent != self.image_dir.resolve() or not path.is_file():
            raise FileNotFoundError(name)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(name)
        return path

    def label_path(self, name: str) -> Path:
        self.require()
        image = self.image_path(name)
        return self.label_dir / f"{image.stem}.txt"

    def read_labels(self, name: str):
        image_path = self.image_path(name)
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = self.label_path(name)
        text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        boxes = yolo_to_boxes(text, width, height)
        return {"name": image_path.name, "width": width, "height": height, "boxes": boxes}

    def write_labels(self, name: str, boxes):
        current = self.read_labels(name)
        text = boxes_to_yolo(boxes, current["width"], current["height"])
        self.label_path(name).write_text(text, encoding="utf-8")
        saved = yolo_to_boxes(text, current["width"], current["height"])
        return {
            "name": current["name"],
            "width": current["width"],
            "height": current["height"],
            "boxes": saved,
        }

    def write_preannotation(self, name: str, boxes):
        """Write a model draft. Existing label files are left untouched."""
        label_path = self.label_path(name)
        if label_path.exists():
            return "skipped"
        current = self.read_labels(name)
        text = boxes_to_yolo(boxes, current["width"], current["height"])
        label_path.write_text(text, encoding="utf-8")
        journal = self._read_undo()
        journal[name] = {"previous": None, "written": text}
        self._write_undo(journal)
        return "written"

    def batch_undo_available(self) -> bool:
        if self.label_dir is None:
            return False
        return bool(self._read_undo())

    def revert_preannotations(self):
        """Put back files still equal to the last one-click write. Edited files stay."""
        journal = self._read_undo()
        reverted = 0
        kept = 0
        for name, entry in journal.items():
            try:
                label_path = self.label_path(name)
            except FileNotFoundError:
                continue
            if not label_path.exists():
                continue
            current = label_path.read_text(encoding="utf-8")
            if current != entry.get("written", ""):
                kept += 1
                continue
            previous = entry.get("previous")
            if previous is None:
                label_path.unlink()
            else:
                label_path.write_text(previous, encoding="utf-8")
            reverted += 1
        self._write_undo({})
        return {"reverted": reverted, "kept": kept}

    def _undo_path(self) -> Path:
        self.require()
        return self.label_dir / UNDO_NAME

    def _read_undo(self):
        path = self._undo_path()
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files", {})
        if not isinstance(files, dict):
            return {}
        return files

    def _write_undo(self, files):
        path = self._undo_path()
        if not files:
            if path.exists():
                path.unlink()
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"files": files}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)


def _text_has_robot(text: str) -> bool:
    for raw in text.splitlines():
        parts = raw.split()
        if parts and parts[0] == "1":
            return True
    return False
