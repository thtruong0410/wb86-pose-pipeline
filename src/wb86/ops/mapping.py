from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


# Canonical 86-keypoint schema (OpenPose-style): BODY_25 + left hand 21 + right hand 21 + face-19
KEYPOINT_NAMES_86: List[str] = [
    # BODY_25 (25)
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
    # Left hand 21 (25..45)
    *[f"LHand_{i}" for i in range(21)],
    # Right hand 21 (46..66)
    *[f"RHand_{i}" for i in range(21)],
    # Face-19 (67..85)
    *[f"Face_{i}" for i in range(19)],
]


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def body25_from_wholebody133(kps133: np.ndarray, conf133: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    """Map COCO-WholeBody 133 to OpenPose BODY_25.

    Assumes ordering: 0-16 body(17), 17-22 foot(6), 23-90 face(68), 91-111 left hand(21), 112-132 right hand(21).
    Returns (25x2, 25) arrays.
    """
    assert kps133.shape[-1] == 2
    body = kps133[:17]
    foot = kps133[17:23]
    # Named refs under COCO
    nose = body[0]
    l_eye, r_eye = body[1], body[2]
    l_ear, r_ear = body[3], body[4]
    l_sh, r_sh = body[5], body[6]
    l_el, r_el = body[7], body[8]
    l_wr, r_wr = body[9], body[10]
    l_hip, r_hip = body[11], body[12]
    l_knee, r_knee = body[13], body[14]
    l_ank, r_ank = body[15], body[16]
    # foot order assumed: L big, L small, L heel, R big, R small, R heel
    l_big, l_small, l_heel, r_big, r_small, r_heel = foot

    neck = _midpoint(l_sh, r_sh)
    midhip = _midpoint(l_hip, r_hip)

    k25 = np.stack([
        nose, neck, r_sh, r_el, r_wr,
        l_sh, l_el, l_wr, midhip, r_hip,
        r_knee, r_ank, l_hip, l_knee, l_ank,
        r_eye, l_eye, r_ear, l_ear, l_big,
        l_small, l_heel, r_big, r_small, r_heel
    ], axis=0)

    c25 = None
    if conf133 is not None:
        c_body = conf133[:17]
        c_foot = conf133[17:23]
        c_lsh, c_rsh = c_body[5], c_body[6]
        c_neck = (c_lsh + c_rsh) / 2.0
        c_lhip, c_rhip = c_body[11], c_body[12]
        c_midhip = (c_lhip + c_rhip) / 2.0
        c25 = np.array([
            c_body[0], c_neck, c_body[6], c_body[8], c_body[10],
            c_body[5], c_body[7], c_body[9], c_midhip, c_body[12],
            c_body[14], c_body[16], c_body[11], c_body[13], c_body[15],
            c_body[2], c_body[1], c_body[4], c_body[3], c_foot[0],
            c_foot[1], c_foot[2], c_foot[3], c_foot[4], c_foot[5]
        ], dtype=float)
    return k25, c25


def face19_from_face68(face68: np.ndarray) -> np.ndarray:
    """Select 19 facial landmarks from iBUG-68 layout.

    Selected indices (0-based on 68):
    - Eyebrows: L(17,19,21), R(22,24,26)
    - Eyes corners: L(36,39), R(42,45)
    - Nose: tip(30), left(31), right(35), root(27), base center(33)
    - Mouth: left(48), right(54), top(51), bottom(57)
    """
    idx = [17, 19, 21, 22, 24, 26, 36, 39, 42, 45, 30, 31, 35, 48, 54, 51, 57, 27, 33]
    return face68[idx]


def compose_wb86(
    body25: np.ndarray,
    lhand21: np.ndarray | None,
    rhand21: np.ndarray | None,
    face68: np.ndarray | None,
) -> Dict[str, np.ndarray]:
    """Compose the 86-keypoint vector from body25 + hands + face.

    Each input is (N, 2) or None. Missing parts are filled with NaN.
    Returns dict with 'keypoints' (86,2) and 'mask' (86,) for present joints.
    """
    out = np.full((86, 2), np.nan, dtype=float)

    # BODY_25
    out[0:25] = body25

    # Hands
    if lhand21 is not None:
        out[25:46] = lhand21
    if rhand21 is not None:
        out[46:67] = rhand21

    # Face-19
    if face68 is not None:
        out[67:86] = face19_from_face68(face68)

    mask = ~np.isnan(out[:, 0])
    return {"keypoints": out, "mask": mask.astype(np.uint8)}

