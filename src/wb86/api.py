from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .detectors import YOLOv8Detector, BaseDetector
from .wholebody import WholeBodyEstimator
from .refine import HandRefiner, FaceRefiner
from .ops import (
    apply_clahe_lab,
    denoise_bilateral,
    upscale,
    body25_from_wholebody133,
    compose_wb86,
    compute_face_roi,
    compute_hand_roi,
    draw_wb86,
    ArrayOneEuro,
    KEYPOINT_NAMES_86,
)


@dataclass
class WB86Config:
    detector: str = "yolov8n.pt"
    detector_conf: float = 0.3
    wholebody_model: str = "rtmpose-wholebody"
    hand_model: str = "rtmpose-hand"
    face_model: str = "rtmpose-face"
    apply_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise: bool = False
    upscale_factor: float = 2.0
    roi_pad_face: float = 1.5
    roi_size_hand: float = 120.0
    smooth: bool = True
    smooth_fps: float = 25.0
    smooth_min_cutoff: float = 1.0
    smooth_beta: float = 0.0
    smooth_d_cutoff: float = 1.0
    bbox_score_thresh: float = 0.3


def load_config(path: Optional[str]) -> WB86Config:
    if path is None:
        return WB86Config()
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    cfg = WB86Config(**{k: v for k, v in d.items() if k in WB86Config.__annotations__})
    return cfg


class WB86Pipeline:
    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_config(config_path)
        self.detector: BaseDetector = YOLOv8Detector(self.cfg.detector, conf=self.cfg.detector_conf)
        self.wholebody = WholeBodyEstimator(self.cfg.wholebody_model)
        self.hand_refiner = HandRefiner(self.cfg.hand_model)
        self.face_refiner = FaceRefiner(self.cfg.face_model)
        self.smoother = ArrayOneEuro(
            freq=self.cfg.smooth_fps,
            min_cutoff=self.cfg.smooth_min_cutoff,
            beta=self.cfg.smooth_beta,
            d_cutoff=self.cfg.smooth_d_cutoff,
        ) if self.cfg.smooth else None

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        out = img
        if self.cfg.apply_clahe:
            out = apply_clahe_lab(out, self.cfg.clahe_clip, self.cfg.clahe_grid)
        if self.cfg.denoise:
            out = denoise_bilateral(out)
        if self.cfg.upscale_factor and self.cfg.upscale_factor != 1.0:
            out = upscale(out, self.cfg.upscale_factor)
        return out

    def _postprocess(self, img: np.ndarray, kps86: np.ndarray) -> np.ndarray:
        if self.smoother is not None:
            xy = self.smoother(kps86[:, :2])
            kps86[:, :2] = xy
        return kps86

    def _run_single(self, img_bgr: np.ndarray) -> List[Dict]:
        h, w = img_bgr.shape[:2]
        proc = self._preprocess(img_bgr)

        dets = self.detector.detect(proc)
        dets = [d for d in dets if d[4] >= self.cfg.bbox_score_thresh]

        persons = self.wholebody.infer(proc, bboxes_xyxy=[d[:4] for d in dets] if dets else None)
        results = []
        for p in persons:
            k133 = p["keypoints133"]
            c133 = p.get("conf")
            body25, c25 = body25_from_wholebody133(k133, c133)

            # ROIs
            face_roi = compute_face_roi(k133, (w, h), pad=self.cfg.roi_pad_face)
            lhand_roi = compute_hand_roi(k133, True, (w, h), size=self.cfg.roi_size_hand)
            rhand_roi = compute_hand_roi(k133, False, (w, h), size=self.cfg.roi_size_hand)

            # Crop and refine
            def crop(xyxy):
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    return None
                return proc[y1:y2, x1:x2]

            face_img = crop(face_roi)
            lhand_img = crop(lhand_roi)
            rhand_img = crop(rhand_roi)

            face68 = None
            c_face = None
            if face_img is not None:
                face68, c_face = self.face_refiner.infer(face_img)
                # map back to full image coords
                fx1, fy1, _, _ = face_roi
                face68 = face68 + np.array([fx1, fy1])

            lhand21 = None
            c_lhand = None
            if lhand_img is not None:
                lhand21, c_lhand = self.hand_refiner.infer(lhand_img)
                lx1, ly1, _, _ = lhand_roi
                lhand21 = lhand21 + np.array([lx1, ly1])

            rhand21 = None
            c_rhand = None
            if rhand_img is not None:
                rhand21, c_rhand = self.hand_refiner.infer(rhand_img)
                rx1, ry1, _, _ = rhand_roi
                rhand21 = rhand21 + np.array([rx1, ry1])

            comp = compose_wb86(body25, c25, lhand21, c_lhand, rhand21, c_rhand, face68, c_face)
            kps86 = comp["keypoints"]
            kps86 = self._postprocess(proc, kps86)

            # derive person box from whole-body keypoints
            valid = ~np.isnan(k133[:, 0])
            if np.any(valid):
                mins = np.nanmin(k133[valid], axis=0)
                maxs = np.nanmax(k133[valid], axis=0)
                person_box = [float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])]
            else:
                person_box = [0.0, 0.0, float(w - 1), float(h - 1)]

            results.append({
                "keypoints": kps86.tolist(),
                "schema": KEYPOINT_NAMES_86,
                "boxes": [person_box],
                "rois": {
                    "face": [float(x) for x in face_roi.tolist()],
                    "lhand": [float(x) for x in lhand_roi.tolist()],
                    "rhand": [float(x) for x in rhand_roi.tolist()],
                },
                "meta": {
                    "config": self.cfg.__dict__,
                },
            })
        return results

    def infer_image(self, path: str, out_dir: Optional[str] = None, visualize: bool = False) -> List[Dict]:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        res = self._run_single(img)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            for i, r in enumerate(res):
                with open(os.path.join(out_dir, f"{base}_person{i}.json"), "w", encoding="utf-8") as f:
                    json.dump(r, f)
                if visualize:
                    vis = draw_wb86(img, np.array(r["keypoints"]))
                    cv2.imwrite(os.path.join(out_dir, f"{base}_person{i}.jpg"), vis)
        return res
