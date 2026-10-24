"""Local configuration. Checkpoints stay outside the repository."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.json"


def load_config():
    data = {
        "sam3_root": "/home/ck/Downloads/sam3_preannotation_delivery_final",
        "checkpoint": "",
        "threshold": 0.3,
        "nms_iou": 0.5,
        "table_top": None,
        "table_bottom": None,
        "min_item_area": 0.0,
        "min_robot_side": 0.0,
    }
    if CONFIG_PATH.exists():
        data.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    checkpoint = str(data.get("checkpoint") or "").strip()
    if not checkpoint:
        candidate = Path(data["sam3_root"]) / "models" / "sam3.pt"
        if candidate.is_file():
            checkpoint = str(candidate)
    elif not Path(checkpoint).is_absolute():
        checkpoint = str((ROOT / checkpoint).resolve())
    data["checkpoint"] = checkpoint
    data["threshold"] = float(data["threshold"])
    data["nms_iou"] = float(data["nms_iou"])
    data["sam3_root"] = str(data["sam3_root"])
    data["table_top"] = _edge(data.get("table_top"))
    data["table_bottom"] = _edge(data.get("table_bottom"))
    data["min_item_area"] = float(data.get("min_item_area") or 0.0)
    data["min_robot_side"] = float(data.get("min_robot_side") or 0.0)
    return data


def _edge(value):
    """Two normalized points, far edge then near edge. Missing means no crop."""
    if not value:
        return None
    points = [[float(point[0]), float(point[1])] for point in value]
    if len(points) != 2:
        raise ValueError("台面边界要是两个点")
    return points
