from __future__ import annotations

from typing import Tuple
import numpy as np


def _expand_box(xyxy: np.ndarray, scale: float, img_wh: Tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(float)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1), (y2 - y1)
    nw, nh = w * scale, h * scale
    nx1, ny1 = max(0.0, cx - nw / 2.0), max(0.0, cy - nh / 2.0)
    nx2, ny2 = min(img_wh[0] - 1.0, cx + nw / 2.0), min(img_wh[1] - 1.0, cy + nh / 2.0)
    return np.array([nx1, ny1, nx2, ny2], dtype=float)


def compute_face_roi(kps133: np.ndarray, img_wh: Tuple[int, int], pad: float = 1.5) -> np.ndarray:
    # Use eyes and ears to bound face if available, else nose fallback
    body = kps133[:17]
    pts = []
    for idx in [0, 1, 2, 3, 4]:  # nose, l_eye, r_eye, l_ear, r_ear
        p = body[idx]
        if not np.any(np.isnan(p)):
            pts.append(p)
    if len(pts) == 0:
        return np.array([0, 0, img_wh[0] - 1, img_wh[1] - 1], dtype=float)
    pts = np.array(pts)
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return _expand_box(np.array([x1, y1, x2, y2]), pad, img_wh)


def compute_hand_roi(kps133: np.ndarray, left: bool, img_wh: Tuple[int, int], size: float = 120.0) -> np.ndarray:
    # Center at wrist with square box of given size; fallback to elbow
    body = kps133[:17]
    wrist_idx = 9 if left else 10
    elbow_idx = 7 if left else 8
    p = body[wrist_idx]
    if np.any(np.isnan(p)):
        p = body[elbow_idx]
    if np.any(np.isnan(p)):
        return np.array([0, 0, 0, 0], dtype=float)
    cx, cy = p
    half = size / 2.0
    x1, y1 = max(0.0, cx - half), max(0.0, cy - half)
    x2, y2 = min(img_wh[0] - 1.0, cx + half), min(img_wh[1] - 1.0, cy + half)
    return np.array([x1, y1, x2, y2], dtype=float)

