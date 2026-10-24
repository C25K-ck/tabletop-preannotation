"""YOLO-HBB read and write. One line is: class cx cy w h, all normalized."""

from .taxonomy import CLASS_BY_ID, CLASS_NAMES


def classes_text():
    return "\n".join(CLASS_NAMES) + "\n"


def classes_match(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines == CLASS_NAMES


def clamp_box(box, width, height):
    x1 = min(box["x1"], box["x2"])
    y1 = min(box["y1"], box["y2"])
    x2 = max(box["x1"], box["x2"])
    y2 = max(box["y1"], box["y2"])
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    x2 = min(max(x2, 0.0), width)
    y2 = min(max(y2, 0.0), height)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    if box["class_id"] not in CLASS_BY_ID:
        raise ValueError(f"unknown class id: {box['class_id']}")
    kept = {
        "class_id": int(box["class_id"]),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }
    if box.get("score") is not None:
        kept["score"] = float(box["score"])
    return kept


def boxes_to_yolo(boxes, width, height) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    lines = []
    for box in boxes:
        kept = clamp_box(box, width, height)
        if kept is None:
            continue
        cx = ((kept["x1"] + kept["x2"]) / 2) / width
        cy = ((kept["y1"] + kept["y2"]) / 2) / height
        bw = (kept["x2"] - kept["x1"]) / width
        bh = (kept["y2"] - kept["y1"]) / height
        lines.append(
            f"{kept['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def yolo_to_boxes(text: str, width: int, height: int):
    boxes = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {line_no} is not YOLO-HBB")
        class_id = int(parts[0])
        cx, cy, bw, bh = (float(part) for part in parts[1:])
        box = clamp_box(
            {
                "class_id": class_id,
                "x1": (cx - bw / 2) * width,
                "y1": (cy - bh / 2) * height,
                "x2": (cx + bw / 2) * width,
                "y2": (cy + bh / 2) * height,
            },
            width,
            height,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def has_robot(boxes) -> bool:
    return any(box["class_id"] == 1 for box in boxes)
