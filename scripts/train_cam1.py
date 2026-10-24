"""Train a three-class YOLO on human-corrected cam1 labels.

Images and txt files stay side by side in data/labeled/cam1. This script
copies them into a YOLO layout under data/yolo/cam1. The later session
(filenames at or after 152331) is the validation set, so adjacent frames
from one stretch are not mixed into both sides.

Two label fixes are applied only in the training copy:
- a 4-pixel robot speck in cam1_20261021_141749
- the parked cart in cam1_20261021_141849, which was saved as an item
Boxes smaller than 8 pixels on both sides are dropped. Partial carts at
the frame edge are kept.
"""

from pathlib import Path
import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "labeled" / "cam1"
DATASET = ROOT / "data" / "yolo" / "cam1"
VAL_FROM = 152331
CLASS_NAMES = ["item", "robot", "hand"]
ROBOT_AS_ITEM = "0 0.212357 0.896306 0.155518 0.207389"


def stamp(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def clean_label(path: Path) -> str:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == ROBOT_AS_ITEM:
            line = "1" + line[1:]
        parts = line.split()
        if len(parts) != 5:
            continue
        width = float(parts[3]) * 960
        height = float(parts[4]) * 544
        if width < 8 and height < 8:
            continue
        lines.append(line)
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def prepare() -> None:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)
    counts = {"train": 0, "val": 0}
    for image in sorted(SOURCE.glob("cam1_*.jpg")):
        label = SOURCE / f"{image.stem}.txt"
        if not label.exists():
            continue
        split = "val" if stamp(image) >= VAL_FROM else "train"
        shutil.copy2(image, DATASET / "images" / split / image.name)
        text = clean_label(label)
        (DATASET / "labels" / split / f"{image.stem}.txt").write_text(text, encoding="utf-8")
        counts[split] += 1
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (DATASET / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {DATASET}",
                "train: images/train",
                "val: images/val",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"prepared train={counts['train']} val={counts['val']}")


def train() -> None:
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATASET / "data.yaml"),
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        patience=20,
        cache=True,
        project=str(ROOT / "data" / "runs"),
        name="cam1",
        exist_ok=True,
        degrees=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
    )


if __name__ == "__main__":
    prepare()
    train()
