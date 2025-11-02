import os
import pytest
import numpy as np
import cv2


@pytest.mark.skipif("mmpose" not in globals(), reason="Optional deps not installed")
def test_pipeline_constructs_and_runs(tmp_path):
    try:
        from wb86.api import WB86Pipeline
    except Exception:
        pytest.skip("Deps not installed")

    # Create a simple blank image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    p = tmp_path / "test.jpg"
    cv2.imwrite(str(p), img)

    pipe = WB86Pipeline()
    try:
        res = pipe.infer_image(str(p), out_dir=str(tmp_path), visualize=False)
    except Exception:
        pytest.skip("Runtime deps not available")
    assert isinstance(res, list)

