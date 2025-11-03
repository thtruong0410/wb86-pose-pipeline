from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ops import compose_wb86, draw_wb86, KEYPOINT_NAMES_86


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _to_xy_conf_mp_landmarks(landmarks, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(landmarks)
    xy = np.full((n, 2), np.nan, dtype=float)
    c = np.zeros((n,), dtype=float)
    for i, lm in enumerate(landmarks):
        x = float(lm.x) * img_w
        y = float(lm.y) * img_h
        # clamp to image bounds for sanity
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xy[i] = np.array([x, y], dtype=float)
        vis = getattr(lm, "visibility", None)
        if vis is None:
            # Some landmark types (hands/face) may not expose visibility; mark as present.
            vis = 1.0
        c[i] = float(vis)
    return xy, c


def body25_from_mediapipe33(pose_landmarks, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Map MediaPipe Pose 33 landmarks to OpenPose BODY_25 order.

    MediaPipe Pose indices (subset):
      0 Nose, 2 LEye, 5 REye, 7 LEar, 8 REar,
      11 LShoulder, 12 RShoulder, 13 LElbow, 14 RElbow, 15 LWrist, 16 RWrist,
      23 LHip, 24 RHip, 25 LKnee, 26 RKnee, 27 LAnkle, 28 RAnkle,
      29 LHeel, 30 RHeel, 31 LBigToe, 32 RBigToe.
    BODY_25 order (0..24):
      Nose, Neck, RShoulder, RElbow, RWrist,
      LShoulder, LElbow, LWrist, MidHip, RHip,
      RKnee, RAnkle, LHip, LKnee, LAnkle,
      REye, LEye, REar, LEar, LBigToe,
      LSmallToe, LHeel, RBigToe, RSmallToe, RHeel
    """
    xy33, c33 = _to_xy_conf_mp_landmarks(pose_landmarks, img_w, img_h)

    def P(i: int) -> np.ndarray:
        return xy33[i]

    def C(i: int) -> float:
        return c33[i]

    neck = _midpoint(P(11), P(12))
    c_neck = (C(11) + C(12)) / 2.0
    midhip = _midpoint(P(23), P(24))
    c_midhip = (C(23) + C(24)) / 2.0

    # Compose BODY_25 geometry
    k25 = np.stack([
        P(0),               # Nose
        neck,
        P(12), P(14), P(16),  # Right shoulder/elbow/wrist
        P(11), P(13), P(15),  # Left shoulder/elbow/wrist
        midhip,
        P(24), P(26), P(28),  # Right hip/knee/ankle
        P(23), P(25), P(27),  # Left hip/knee/ankle
        P(5), P(2),           # REye, LEye
        P(8), P(7),           # REar, LEar
        P(31),                # LBigToe
        P(31),                # LSmallToe (approximate as LBigToe)
        P(29),                # LHeel
        P(32),                # RBigToe
        P(32),                # RSmallToe (approximate as RBigToe)
        P(30),                # RHeel
    ], axis=0)

    c25 = np.array([
        C(0), c_neck,
        C(12), C(14), C(16),
        C(11), C(13), C(15),
        c_midhip,
        C(24), C(26), C(28),
        C(23), C(25), C(27),
        C(5), C(2),
        C(8), C(7),
        C(31), C(31), C(29),
        C(32), C(32), C(30),
    ], dtype=float)

    return k25, c25


def _prefer_realesrgan_upscale(img_bgr: np.ndarray, scale: int, device: str = "cuda") -> np.ndarray | None:
    # Soft import to avoid hard dependency
    try:
        from PIL import Image
        from realesrgan import RealESRGAN  # type: ignore
        import torch  # type: ignore
    except Exception:
        return None
    try:
        dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        model_scale = 4
        model = RealESRGAN(dev, model_scale)
        model.load_weights(RealESRGAN.weights_path("RealESRGAN_x4plus"))
        # Convert BGR->RGB PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(img_rgb)
        pil_out = model.predict(pil_in)
        if scale != model_scale:
            pil_out = pil_out.resize((img_bgr.shape[1] * scale, img_bgr.shape[0] * scale), Image.LANCZOS)
        out_rgb = np.array(pil_out)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def face68_from_mp468(face_landmarks, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate subset of iBUG-68 from MediaPipe Face Mesh (468).

    Only fills the indices used by face19_from_face68:
      - Eyebrows (17,19,21, 22,24,26) -> left empty (NaN)
      - Eye corners: L(36 outer, 39 inner) from MP [362, 263]; R(42 outer, 45 inner) from MP [33, 133]
      - Nose: tip(30), left(31), right(35), root(27), base center(33) -> left empty (NaN)
      - Mouth: left(48=MP61), right(54=MP291), top(51=MP13), bottom(57=MP14)
    """
    # Prepare arrays
    xy68 = np.full((68, 2), np.nan, dtype=float)
    c68 = np.zeros((68,), dtype=float)

    # Helper to read an MP index safely
    def mp_xy(idx: int) -> Tuple[float, float]:
        lm = face_landmarks[idx]
        return float(lm.x) * img_w, float(lm.y) * img_h

    # Mapping definitions
    mapping = {
        # Eyes (iBUG -> MediaPipe)
        36: 362,  # left eye outer
        39: 263,  # left eye inner
        42: 33,   # right eye outer
        45: 133,  # right eye inner
        # Mouth corners and centers
        48: 61,   # left mouth corner
        54: 291,  # right mouth corner
        51: 13,   # upper lip center
        57: 14,   # lower lip center
        # Optional: nose tip could be 1, but keep unset to avoid misplacement
    }

    for ibug_idx, mp_idx in mapping.items():
        try:
            x, y = mp_xy(mp_idx)
            xy68[ibug_idx] = np.array([x, y], dtype=float)
            c68[ibug_idx] = 1.0
        except Exception:
            pass

    return xy68, c68


@dataclass
class MP86Config:
    scale: int = 1  # 1,2,3,4
    sr_prefer: str = "realesrgan"  # or "bicubic"
    sr_device: str = "cuda"
    mp_model_complexity: int = 1
    mp_enable_segmentation: bool = False
    mp_refine_face_landmarks: bool = False


class MP86Pipeline:
    """MediaPipe-based 86-keypoint pipeline with optional SR upscaling.

    Steps:
      - Optional: upscale input with Real-ESRGAN (preferred) or bicubic to preserve aspect ratio.
      - Run MediaPipe Holistic to get pose(33) + hands(21/21) + face(468) landmarks.
      - Map pose33->BODY_25; compose to 86 with hands and (optionally) face subset.
      - Output JSON compatible with WB86 results and optional visualization.
    """

    def __init__(self, cfg: MP86Config | None = None):
        self.cfg = cfg or MP86Config()
        # Lazy init for mediapipe to allow import in environments without it
        self._mp_holistic = None
        self._mp_hands = None

    def _ensure_holistic(self):
        if self._mp_holistic is None:
            try:
                import mediapipe as mp  # type: ignore
            except Exception as e:  # pragma: no cover
                raise ImportError("mediapipe is required for MP86Pipeline.") from e
            self._mp = mp
            self._mp_holistic = mp.solutions.holistic.Holistic(
                static_image_mode=True,
                model_complexity=self.cfg.mp_model_complexity,
                enable_segmentation=self.cfg.mp_enable_segmentation,
                refine_face_landmarks=self.cfg.mp_refine_face_landmarks,
            )

    def _ensure_hands(self):
        if self._mp_hands is None:
            self._ensure_holistic()
            self._mp_hands = self._mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=self.cfg.mp_model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def _infer_hand_from_roi(self, img_bgr: np.ndarray, center_xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray] | None:
        self._ensure_hands()
        h, w = img_bgr.shape[:2]
        cx, cy = center_xy
        if not np.isfinite(cx) or not np.isfinite(cy):
            return None
        # ROI size heuristic
        size = int(0.18 * max(h, w))
        size = max(64, size)
        x1 = max(0, int(cx - size // 2))
        y1 = max(0, int(cy - size // 2))
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        rgb = roi[:, :, ::-1]
        res = self._mp_hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        h_pts = res.multi_hand_landmarks[0].landmark
        xy21 = np.zeros((21, 2), dtype=float)
        c21 = np.ones((21,), dtype=float)
        rw, rh = roi.shape[1], roi.shape[0]
        for i, lm in enumerate(h_pts):
            x = float(lm.x) * rw + x1
            y = float(lm.y) * rh + y1
            xy21[i] = [x, y]
        return xy21, c21

    def _upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        s = int(self.cfg.scale)
        if s <= 1:
            return img_bgr
        if self.cfg.sr_prefer == "realesrgan":
            out = _prefer_realesrgan_upscale(img_bgr, s, device=self.cfg.sr_device)
            if out is not None:
                return out
        # fallback bicubic
        h, w = img_bgr.shape[:2]
        return cv2.resize(img_bgr, (w * s, h * s), interpolation=cv2.INTER_CUBIC)

    def _run_single(self, img_bgr: np.ndarray) -> List[Dict]:
        self._ensure_holistic()
        img_proc = self._upscale(img_bgr)
        h, w = img_proc.shape[:2]

        # MediaPipe expects RGB
        rgb = img_proc[:, :, ::-1]
        res = self._mp_holistic.process(rgb)

        # Pose (33)
        body25 = None
        c_body25 = None
        if res.pose_landmarks and res.pose_landmarks.landmark:
            body25, c_body25 = body25_from_mediapipe33(res.pose_landmarks.landmark, w, h)

        # Hands (21/21)
        lhand21 = None
        c_lhand21 = None
        if res.left_hand_landmarks and res.left_hand_landmarks.landmark:
            l_xy, l_c = _to_xy_conf_mp_landmarks(res.left_hand_landmarks.landmark, w, h)
            lhand21, c_lhand21 = l_xy[:21], l_c[:21]
        elif body25 is not None:
            # Fallback to dedicated MP Hands on wrist ROI
            lwrist = body25[7]  # LWrist in BODY_25
            if not np.any(np.isnan(lwrist)):
                out = self._infer_hand_from_roi(img_proc, (lwrist[0], lwrist[1]))
                if out is not None:
                    lhand21, c_lhand21 = out

        rhand21 = None
        c_rhand21 = None
        if res.right_hand_landmarks and res.right_hand_landmarks.landmark:
            r_xy, r_c = _to_xy_conf_mp_landmarks(res.right_hand_landmarks.landmark, w, h)
            rhand21, c_rhand21 = r_xy[:21], r_c[:21]
        elif body25 is not None:
            rwrist = body25[4]  # RWrist in BODY_25
            if not np.any(np.isnan(rwrist)):
                out = self._infer_hand_from_roi(img_proc, (rwrist[0], rwrist[1]))
                if out is not None:
                    rhand21, c_rhand21 = out

        # Face (approximate subset from MP FaceMesh to iBUG-68 indices needed by face19)
        face68 = None
        c_face68 = None
        if res.face_landmarks and res.face_landmarks.landmark:
            face68, c_face68 = face68_from_mp468(res.face_landmarks.landmark, w, h)

        if body25 is None:
            # Nothing detected
            return []

        comp = compose_wb86(body25, c_body25, lhand21, c_lhand21, rhand21, c_rhand21, face68, c_face68)
        kps86 = comp["keypoints"]
        # Robustness: if XY is present but confidence is 0 (e.g., some MP landmarks lack visibility), set conf=1.
        xy = kps86[:, :2]
        conf = kps86[:, 2]
        valid_xy = (~np.isnan(xy[:, 0])) & (~np.isnan(xy[:, 1]))
        conf[(valid_xy) & (conf == 0)] = 1.0
        kps86[:, 2] = conf

        # Person box from available keypoints
        valid = ~np.isnan(kps86[:, 0])
        if np.any(valid):
            mins = np.nanmin(kps86[valid, :2], axis=0)
            maxs = np.nanmax(kps86[valid, :2], axis=0)
            person_box = [float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])]
        else:
            person_box = [0.0, 0.0, float(w - 1), float(h - 1)]

        result = {
            "keypoints": kps86.tolist(),
            "schema": KEYPOINT_NAMES_86,
            "boxes": [person_box],
            "rois": {
                "face": [0.0, 0.0, 0.0, 0.0],
                "lhand": [0.0, 0.0, 0.0, 0.0],
                "rhand": [0.0, 0.0, 0.0, 0.0],
            },
            "meta": {
                "engine": "mediapipe",
                "sr": {
                    "scale": self.cfg.scale,
                    "prefer": self.cfg.sr_prefer,
                },
                "face_mapping": "mp468->ibug68(partial)",
            },
        }
        return [result]

    def infer_image(self, path: str, out_dir: Optional[str] = None, visualize: bool = False) -> List[Dict]:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        res = self._run_single(img)
        if out_dir:
            import os, json
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            for i, r in enumerate(res):
                with open(os.path.join(out_dir, f"{base}_person{i}.json"), "w", encoding="utf-8") as f:
                    json.dump(r, f)
                if visualize:
                    # Draw in the same coordinate space as keypoints (post-upscale)
                    img_vis = self._upscale(img)
                    vis = draw_wb86(img_vis, np.array(r["keypoints"]))
                    cv2.imwrite(os.path.join(out_dir, f"{base}_person{i}.jpg"), vis)
        return res
