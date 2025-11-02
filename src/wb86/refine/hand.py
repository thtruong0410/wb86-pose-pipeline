from __future__ import annotations

from typing import List, Tuple
import numpy as np


class HandRefiner:
    """Refine 21 hand keypoints using mmpose hand model.

    model_alias: e.g., 'rtmpose-hand' or a checkpoints alias supported by MMPoseInferencer.
    """

    def __init__(self, model_alias: str = "rtmpose-hand", device: str | None = None):
        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("mmpose>=1.2.0 is required for HandRefiner. Install with [wholebody] extra.") from e
        self._Infer = MMPoseInferencer
        kwargs = {"pose2d": model_alias}
        if device is not None:
            kwargs["device"] = device
        self.inf = self._Infer(**kwargs)

    def infer(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # returns (21x2, 21) arrays
        preds = list(self.inf([img_bgr[:, :, ::-1]], batch_size=1))
        if not preds:
            return np.full((21, 2), np.nan), np.zeros((21,), dtype=float)
        res = preds[0]
        if "predictions" not in res or not res["predictions"][0]:
            return np.full((21, 2), np.nan), np.zeros((21,), dtype=float)
        inst = res["predictions"][0][0]
        arr = np.asarray(inst.get("keypoints", [[[]]]))[0]
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] < 21:
            return np.full((21, 2), np.nan), np.zeros((21,), dtype=float)
        xy = arr[:21, :2]
        conf = arr[:21, 2] if arr.shape[1] >= 3 else np.ones((21,), dtype=float)
        return xy, conf

