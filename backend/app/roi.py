"""Keep draft boxes on the table, and drop tile-joint sized item boxes."""

from .nms import _iou


def _line_y(line, x: float) -> float:
    (x0, y0), (x1, y1) = line
    if x1 == x0:
        return float(y0)
    t = (x - x0) / (x1 - x0)
    return float(y0) + t * (float(y1) - float(y0))


def on_table(box, width: int, height: int, top, bottom) -> bool:
    """True when the box center sits between the far and near table edges."""
    cx = ((box["x1"] + box["x2"]) / 2) / width
    cy = ((box["y1"] + box["y2"]) / 2) / height
    cx = min(1.0, max(0.0, cx))
    return _line_y(top, cx) <= cy <= _line_y(bottom, cx)


# Burned-in camera name at the lower-left corner of these overhead frames.
_CAMERA_LABEL = (0.0, 0.92, 0.12, 1.0)


def _on_camera_label(box, width: int, height: int) -> bool:
    """True when an item box is mostly the CAMO1 overlay, not an object beside it."""
    if box["class_id"] != 0:
        return False
    rx1, ry1, rx2, ry2 = _CAMERA_LABEL
    ix1 = max(box["x1"], rx1 * width)
    iy1 = max(box["y1"], ry1 * height)
    ix2 = min(box["x2"], rx2 * width)
    iy2 = min(box["y2"], ry2 * height)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
    if area <= 0:
        return False
    return (ix2 - ix1) * (iy2 - iy1) / area >= 0.7


def apply_table_rules(
    boxes,
    width: int,
    height: int,
    top,
    bottom,
    min_item_area: float,
    min_robot_side: float = 0.0,
):
    image_area = width * height
    kept = []
    for box in boxes:
        if top is not None and bottom is not None and not on_table(box, width, height, top, bottom):
            continue
        if box["class_id"] == 0 and min_item_area > 0:
            area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]) / image_area
            if area < min_item_area:
                continue
        if box["class_id"] == 1 and min_robot_side > 0:
            shorter = min(box["x2"] - box["x1"], box["y2"] - box["y1"]) / height
            if shorter < min_robot_side:
                continue
        if _on_camera_label(box, width, height):
            continue
        kept.append(box)
    return _drop_items_that_are_the_robot(kept)


def _drop_items_that_are_the_robot(boxes, iou_threshold: float = 0.5):
    """An item box covering the robot body is the same object. A small item in the gripper stays."""
    robots = [box for box in boxes if box["class_id"] == 1]
    if not robots:
        return boxes
    kept = []
    for box in boxes:
        if box["class_id"] == 0 and any(_iou(box, robot) >= iou_threshold for robot in robots):
            continue
        kept.append(box)
    return kept
