from __future__ import annotations

import cv2
import numpy as np


def apply_clahe_lab(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise_bilateral(img_bgr: np.ndarray, d: int = 5, sigma_color: float = 50, sigma_space: float = 50) -> np.ndarray:
    return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def upscale(img_bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
    if scale == 1.0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)

