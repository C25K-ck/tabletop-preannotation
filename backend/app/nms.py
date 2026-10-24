"""Same-class NMS. Boxes from different classes are never removed."""


def _iou(a, b) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_by_class(boxes, iou_threshold: float):
    kept = []
    class_ids = sorted({box["class_id"] for box in boxes})
    for class_id in class_ids:
        group = [box for box in boxes if box["class_id"] == class_id]
        order = sorted(range(len(group)), key=lambda i: group[i].get("score") or 0.0, reverse=True)
        selected = []
        while order:
            current = order.pop(0)
            selected.append(group[current])
            order = [
                index
                for index in order
                if _iou(group[current], group[index]) < iou_threshold
            ]
        kept.extend(selected)
    return kept
