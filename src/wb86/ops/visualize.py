from __future__ import annotations

from typing import List, Tuple
import numpy as np
import cv2


_BODY25_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # nose-neck-right arm
    (1, 5), (5, 6), (6, 7),          # left arm
    (1, 8), (8, 9), (9, 10), (10, 11),  # right leg
    (8, 12), (12, 13), (13, 14),        # left leg
    (0, 15), (0, 16), (15, 17), (16, 18) # eyes/ears
]

_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # index
    (0, 9), (9, 10), (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


def _draw_lines(img, pts, edges, color, thickness=2):
    for i, j in edges:
        pi, pj = pts[i], pts[j]
        if not (np.any(np.isnan(pi)) or np.any(np.isnan(pj))):
            cv2.line(img, tuple(pi.astype(int)), tuple(pj.astype(int)), color, thickness)


def _draw_points(img, pts, color, radius=2):
    for p in pts:
        if not np.any(np.isnan(p)):
            cv2.circle(img, tuple(p.astype(int)), radius, color, -1)


def draw_wb86(img_bgr: np.ndarray, kps86: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    body = kps86[0:25]
    lhand = kps86[25:46]
    rhand = kps86[46:67]
    face = kps86[67:86]

    _draw_lines(out, body, _BODY25_EDGES, (0, 255, 0), 2)
    _draw_lines(out, lhand, _HAND_EDGES, (255, 0, 0), 2)
    _draw_lines(out, rhand, _HAND_EDGES, (0, 0, 255), 2)
    _draw_points(out, face, (0, 255, 255), 2)
    return out

