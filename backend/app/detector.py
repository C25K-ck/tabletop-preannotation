"""SAM3 text detection. Same-class NMS only; nested robot and item boxes both stay."""

import sys
import threading
from pathlib import Path

from PIL import Image

from .nms import nms_by_class
from .roi import apply_table_rules
from .settings import load_config
from .taxonomy import prompt_catalog

_lock = threading.Lock()
_runtime = None


def status():
    config = load_config()
    reasons = []
    checkpoint = config["checkpoint"]
    if not checkpoint or not Path(checkpoint).is_file():
        reasons.append("未配置可用的 sam3.pt")
    sam3_root = Path(config["sam3_root"])
    if not (sam3_root / "sam3").is_dir():
        reasons.append("找不到 SAM3 源码目录")
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
        "checkpoint": checkpoint,
        "threshold": config["threshold"],
        "nms_iou": config["nms_iou"],
    }


def _load():
    global _runtime
    if _runtime is not None:
        return _runtime
    current = status()
    if not current["ready"]:
        raise RuntimeError("；".join(current["reasons"]))
    config = load_config()
    root = config["sam3_root"]
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device="cpu",
        load_from_HF=False,
        enable_segmentation=False,
    )
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    checkpoint = checkpoint.get("model", checkpoint)
    detector_weights = {
        key[len("detector.") :]: value
        for key, value in checkpoint.items()
        if key.startswith("detector.")
    }
    missing, _extra = model.load_state_dict(detector_weights, strict=False)
    if missing:
        raise RuntimeError(f"检测权重缺少参数：{missing}")
    model = model.cuda().eval()
    processor = Sam3Processor(model, resolution=1008, device="cuda")
    prompts = prompt_catalog()
    with torch.inference_mode():
        texts = model.backbone.forward_text(
            [item["prompt"] for item in prompts],
            device="cuda",
        )
    _runtime = {
        "model": model,
        "processor": processor,
        "texts": texts,
        "prompts": prompts,
        "box_cxcywh_to_xyxy": box_cxcywh_to_xyxy,
        "torch": torch,
    }
    return _runtime


def detect_image(image: Image.Image, threshold: float, nms_iou: float):
    with _lock:
        runtime = _load()
        torch = runtime["torch"]
        model = runtime["model"]
        processor = runtime["processor"]
        prompts = runtime["prompts"]
        to_xyxy = runtime["box_cxcywh_to_xyxy"]
        image = image.convert("RGB")
        import copy

        import numpy as np

        n = len(prompts)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tensor = processor.transform(
                torch.from_numpy(np.array(image)).permute(2, 0, 1).cuda()
            ).unsqueeze(0)
            features = model.backbone.forward_image(tensor)
            features.update(runtime["texts"])
            stage = copy.copy(processor.find_stage)
            stage.img_ids = torch.zeros(n, dtype=torch.long, device="cuda")
            stage.text_ids = torch.arange(n, device="cuda")
            output = model.forward_grounding(
                backbone_out=features,
                find_input=stage,
                geometric_prompt=model._get_dummy_prompt(n),
                find_target=None,
            )
        scale = torch.tensor(
            [image.width, image.height, image.width, image.height],
            device="cuda",
        )
        collected = []
        for index, prompt in enumerate(prompts):
            probability = (
                output["pred_logits"][index].sigmoid()
                * output["presence_logit_dec"][index].sigmoid()
            ).flatten()
            keep = probability > threshold
            if int(keep.sum()) == 0:
                continue
            boxes = to_xyxy(output["pred_boxes"][index][keep]) * scale
            boxes[:, 0::2].clamp_(0, image.width)
            boxes[:, 1::2].clamp_(0, image.height)
            scores = probability[keep]
            for box, score in zip(boxes.tolist(), scores.tolist()):
                if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                    continue
                collected.append(
                    {
                        "class_id": prompt["class_id"],
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
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
