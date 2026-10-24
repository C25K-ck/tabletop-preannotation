"""One-click pre-annotation for every unlabeled image, with a revert journal."""

import threading

from PIL import Image


class BatchRunner:
    def __init__(self, dataset, detect):
        self.dataset = dataset
        self.detect = detect
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._cancel_requested = False
        self._total = 0
        self._done = 0
        self._skipped = 0
        self._written = 0
        self._errors = 0
        self._current = ""
        self._last_error: str | None = None
        self._reverted = 0
        self._kept = 0
        self._message = ""

    def status(self):
        with self._lock:
            return self._status()

    def start(self, threshold: float, nms_iou: float, hold: str | None = None):
        self.dataset.require()
        names = self.dataset.image_names()
        pending = []
        skipped = 0
        for name in names:
            if name == hold or self.dataset.has_label(name):
                skipped += 1
            else:
                pending.append(name)
        with self._lock:
            if self._running:
                raise RuntimeError("一键预标注正在进行")
            self._cancel_requested = False
            self._done = 0
            self._written = 0
            self._errors = 0
            self._current = ""
            self._last_error = None
            self._reverted = 0
            self._kept = 0
            self._skipped = skipped
            self._total = len(pending)
            if not pending:
                self._running = False
                self._message = "没有需要预标注的图片"
                return self._status()
            self._running = True
            self._message = ""
            self._thread = threading.Thread(
                target=self._run,
                args=(pending, threshold, nms_iou),
                daemon=True,
            )
            self._thread.start()
            return self._status()

    def cancel(self):
        with self._lock:
            if self._running:
                self._cancel_requested = True
                self._message = "正在取消预标注"
                return self._status()
        result = self.dataset.revert_preannotations()
        with self._lock:
            self._reverted = result["reverted"]
            self._kept = result["kept"]
            self._cancel_requested = True
            self._message = "已取消预标注"
            return self._status()

    def wait(self, timeout: float | None = None):
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, pending, threshold: float, nms_iou: float):
        try:
            for name in pending:
                with self._lock:
                    if self._cancel_requested:
                        break
                    self._current = name
                try:
                    image_path = self.dataset.image_path(name)
                    with Image.open(image_path) as image:
                        image.load()
                        boxes = self.detect(image, threshold, nms_iou)
                except Exception as exc:
                    with self._lock:
                        self._errors += 1
                        self._done += 1
                        self._last_error = f"{name}: {exc}"
                    continue
                with self._lock:
                    if self._cancel_requested:
                        break
                    outcome = self.dataset.write_preannotation(name, boxes)
                    self._done += 1
                    if outcome == "written":
                        self._written += 1
                    else:
                        self._skipped += 1
            with self._lock:
                cancelled = self._cancel_requested
            if cancelled:
                result = self.dataset.revert_preannotations()
                with self._lock:
                    self._reverted = result["reverted"]
                    self._kept = result["kept"]
                    self._message = "已取消预标注"
            else:
                with self._lock:
                    self._message = "一键预标注完成"
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._errors += 1
                self._message = "一键预标注中断"
        finally:
            with self._lock:
                self._running = False
                self._current = ""

    def _status(self):
        undo = False
        if self.dataset.root is not None:
            undo = self.dataset.batch_undo_available()
        return {
            "running": self._running,
            "cancel_requested": self._cancel_requested,
            "total": self._total,
            "done": self._done,
            "skipped": self._skipped,
            "written": self._written,
            "errors": self._errors,
            "current": self._current,
            "last_error": self._last_error,
            "undo_available": undo,
            "reverted": self._reverted,
            "kept": self._kept,
            "message": self._message,
        }
