"""YOLO draft detection. Same-class NMS only; nested robot and item boxes both stay."""

import threading
from pathlib import Path

from PIL import Image

from .nms import nms_by_class
from .roi import apply_table_rules
from .settings import load_config

_lock = threading.Lock()
_model = None
_loaded_path = ""


def status():
    config = load_config()
    reasons = []
    weights = config["yolo_weights"]
    if not weights or not Path(weights).is_file():
        reasons.append("未配置可用的 YOLO 权重")
    try:
        import torch
    except Exception as exc:
        reasons.append(f"无法导入 torch：{exc}")
        torch = None
    if torch is not None and not torch.cuda.is_available():
        reasons.append("当前环境没有可用的 CUDA")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "checkpoint": weights,
        "threshold": config["threshold"],
        "nms_iou": config["nms_iou"],
    }


def _load():
    global _model, _loaded_path
    current = status()
    if not current["ready"]:
        raise RuntimeError("；".join(current["reasons"]))
    weights = current["checkpoint"]
    if _model is not None and _loaded_path == weights:
        return _model
    from ultralytics import YOLO

    _model = YOLO(weights)
    _loaded_path = weights
    return _model


def detect_image(image: Image.Image, threshold: float, nms_iou: float):
    with _lock:
        model = _load()
        image = image.convert("RGB")
        results = model.predict(
            source=image,
            conf=threshold,
            iou=nms_iou,
            imgsz=640,
            device=0,
            verbose=False,
        )
        collected = []
        boxes = results[0].boxes
        if boxes is not None:
            for xyxy, class_id, score in zip(
                boxes.xyxy.tolist(),
                boxes.cls.tolist(),
                boxes.conf.tolist(),
            ):
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                collected.append(
                    {
                        "class_id": int(class_id),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": float(score),
                    }
                )
        config = load_config()
        collected = apply_table_rules(
            collected,
            image.width,
            image.height,
            config["table_top"],
            config["table_bottom"],
            config["min_item_area"],
            config["min_robot_side"],
        )
        return nms_by_class(collected, nms_iou)
