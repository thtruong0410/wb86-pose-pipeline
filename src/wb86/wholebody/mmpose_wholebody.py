from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class WholeBodyEstimator:
    """Wrapper around MMPose inferencer for whole-body (133) keypoints.

    Attempts to use MMPoseInferencer with a model alias from config (e.g., 'rtmpose-wholebody').
    """

    def __init__(self, model_alias: str = "rtmpose-wholebody", device: str | None = None):
        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("mmpose>=1.2.0 is required for WholeBodyEstimator. Install with [wholebody] extra.") from e
        self._Infer = MMPoseInferencer
        kwargs = {"pose2d": model_alias}
        if device is not None:
            kwargs["device"] = device
        self.inf = self._Infer(**kwargs)

    def infer(self, img_bgr: np.ndarray, bboxes_xyxy: List[Tuple[float, float, float, float]] | None = None) -> List[Dict]:
        # MMPoseInferencer accepts numpy image; returns generator of results per image
        inputs = [img_bgr[:, :, ::-1]]  # RGB
        preds = list(self.inf(inputs, batch_size=1))
        if not preds:
            return []
        res = preds[0]
        # Expect res["predictions"][0]["keypoints"][0] etc for whole-body
        persons = []
        if "predictions" not in res:
            return persons
        for inst in res["predictions"][0]:
            # Prefer whole-body if available; else skip
            kps = None
            if "keypoints" in inst and isinstance(inst["keypoints"], list) and len(inst["keypoints"]) > 0:
                kps = inst["keypoints"][0]
            if kps is None:
                continue
            arr = np.asarray(kps, dtype=float)
            if arr.shape[0] < 133:
                continue
            if arr.shape[1] == 3:
                xy = arr[:, :2]
                conf = arr[:, 2]
            else:
                xy = arr[:, :2]
                conf = np.ones((arr.shape[0],), dtype=float)
            persons.append({"keypoints133": xy, "conf": conf})
        return persons

