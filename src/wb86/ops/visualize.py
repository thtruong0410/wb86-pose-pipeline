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


def _draw_points_and_labels(img, pts, offset, color, radius=3):
    for i, p in enumerate(pts):
        if not np.any(np.isnan(p)):
            xy = tuple(p.astype(int))
            cv2.circle(img, xy, radius, color, -1)
            cv2.putText(
                img,
                str(i + offset),
                (xy[0] + 4, xy[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                lineType=cv2.LINE_AA,
            )


def draw_wb86(img_bgr: np.ndarray, kps86: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    pts = kps86[:, :2] if kps86.shape[1] >= 2 else kps86
    conf = kps86[:, 2] if kps86.shape[1] >= 3 else np.ones((pts.shape[0],), dtype=float)

    body = pts[0:25]
    lhand = pts[25:46]
    rhand = pts[46:67]
    face = pts[67:86]

    # Draw skeleton lines
    _draw_lines(out, body, _BODY25_EDGES, (0, 255, 0), 2)
    _draw_lines(out, lhand, _HAND_EDGES, (255, 0, 0), 2)
    _draw_lines(out, rhand, _HAND_EDGES, (0, 0, 255), 2)

    # Draw points with indices per region
    _draw_points_and_labels(out, body, 0, (0, 255, 0), 3)
    _draw_points_and_labels(out, lhand, 25, (255, 0, 0), 3)
    _draw_points_and_labels(out, rhand, 46, (0, 0, 255), 3)
    _draw_points_and_labels(out, face, 67, (0, 255, 255), 3)

    # Top-left overlay: total detected keypoints
    valid_xy = (~np.isnan(pts[:, 0])) & (~np.isnan(pts[:, 1]))
    valid = valid_xy & (conf > 0)
    total = int(valid.sum())
    text = f"Keypoints: {total}/{pts.shape[0]}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pad = 6
    cv2.rectangle(out, (5, 5), (5 + tw + 2 * pad, 5 + th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, text, (5 + pad, 5 + th + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    return out
