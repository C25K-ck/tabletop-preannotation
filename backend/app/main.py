"""Local review API. Images stay on this machine."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

from .batch import BatchRunner
from .dataset import Dataset
from .detector import detect_image, status
from .settings import load_config
from .taxonomy import CLASSES
from .yolo import has_robot

app = FastAPI(title="台面三类预标注")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset = Dataset()
batch = BatchRunner(dataset, detect_image)


class OpenBody(BaseModel):
    path: str


class BoxIn(BaseModel):
    class_id: int = Field(ge=0, le=2)
    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = None


class SaveBody(BaseModel):
    boxes: list[BoxIn]


class DetectBody(BaseModel):
    threshold: float | None = None
    nms_iou: float | None = None


class BatchBody(BaseModel):
    threshold: float | None = None
    nms_iou: float | None = None
    hold: str | None = None


def _fail(exc: Exception, code: int = 400):
    raise HTTPException(status_code=code, detail=str(exc)) from exc


@app.get("/api/meta")
def meta():
    config = load_config()
    return {
        "classes": [
            {"id": item["id"], "name": item["name"], "label": item["label"], "color": item["color"]}
            for item in CLASSES
        ],
        "threshold": config["threshold"],
        "nms_iou": config["nms_iou"],
        "table_top": config["table_top"],
        "table_bottom": config["table_bottom"],
        "detector": status(),
        "dataset": dataset.info(),
    }


@app.post("/api/dataset")
def open_dataset(body: OpenBody):
    if batch.status()["running"]:
        _fail(RuntimeError("一键预标注正在进行"), 409)
    try:
        return dataset.open(body.path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(exc)


@app.get("/api/images")
def list_images():
    try:
        return {"images": dataset.images()}
    except RuntimeError as exc:
        _fail(exc)


@app.get("/api/media/{name}")
def media(name: str):
    try:
        return FileResponse(dataset.image_path(name))
    except (FileNotFoundError, RuntimeError) as exc:
        _fail(exc, 404)


@app.get("/api/labels/{name}")
def read_labels(name: str):
    try:
        payload = dataset.read_labels(name)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _fail(exc, 404)
    payload["has_robot"] = has_robot(payload["boxes"])
    return payload


@app.put("/api/labels/{name}")
def write_labels(name: str, body: SaveBody):
    try:
        payload = dataset.write_labels(name, [box.model_dump() for box in body.boxes])
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _fail(exc)
    payload["has_robot"] = has_robot(payload["boxes"])
    return payload


@app.get("/api/batch")
def batch_status():
    return batch.status()


@app.post("/api/batch")
def start_batch(body: BatchBody | None = None):
    current = status()
    if not current["ready"]:
        _fail(RuntimeError("；".join(current["reasons"])), 503)
    config = load_config()
    body = body or BatchBody()
    threshold = config["threshold"] if body.threshold is None else body.threshold
    nms_iou = config["nms_iou"] if body.nms_iou is None else body.nms_iou
    try:
        return batch.start(threshold, nms_iou, body.hold)
    except RuntimeError as exc:
        _fail(exc, 409)


@app.post("/api/batch/cancel")
def cancel_batch():
    try:
        return batch.cancel()
    except RuntimeError as exc:
        _fail(exc)


@app.post("/api/detect/{name}")
def detect(name: str, body: DetectBody | None = None):
    if batch.status()["running"]:
        _fail(RuntimeError("一键预标注正在进行"), 409)
    current = status()
    if not current["ready"]:
        _fail(RuntimeError("；".join(current["reasons"])), 503)
    config = load_config()
    threshold = config["threshold"] if body is None or body.threshold is None else body.threshold
    nms_iou = config["nms_iou"] if body is None or body.nms_iou is None else body.nms_iou
    try:
        image_path = dataset.image_path(name)
        with Image.open(image_path) as image:
            image.load()
            boxes = detect_image(image, threshold, nms_iou)
            width, height = image.size
    except (FileNotFoundError, RuntimeError) as exc:
        code = 404 if isinstance(exc, FileNotFoundError) else 503
        _fail(exc, code)
    return {
        "name": image_path.name,
        "width": width,
        "height": height,
        "boxes": boxes,
        "has_robot": has_robot(boxes),
    }
