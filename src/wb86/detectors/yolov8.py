from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .base import BaseDetector


class YOLOv8Detector(BaseDetector):
    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.3):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("ultralytics is required for YOLOv8Detector. Install with [detect] extra.") from e
        self._YOLO = YOLO
        self.model = self._YOLO(model)
        self.conf = conf

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        # Ultralytics expects RGB
        rgb = img_bgr[:, :, ::-1]
        res = self.model.predict(source=rgb, verbose=False, conf=self.conf)
        dets: List[Tuple[float, float, float, float, float]] = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = int(b.cls.item()) if b.cls is not None else -1
                if cls != 0:
                    continue  # keep person class only
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                score = float(b.conf.item()) if b.conf is not None else 1.0
                dets.append((x1, y1, x2, y2, score))
        return dets

