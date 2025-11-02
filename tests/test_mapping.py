import numpy as np

from wb86.ops.mapping import body25_from_wholebody133, compose_wb86


def test_body25_from_wholebody133_minimal():
    k = np.zeros((133, 2), dtype=float)
    k[:] = np.nan
    # fill a few required joints
    k[0] = [10, 10]   # nose
    k[5] = [0, 0]     # L shoulder
    k[6] = [10, 0]    # R shoulder
    k[11] = [0, 10]   # L hip
    k[12] = [10, 10]  # R hip
    k[17:23] = [[0,20],[1,20],[0,21],[10,20],[11,20],[10,21]]

    b25, _ = body25_from_wholebody133(k)
    assert b25.shape == (25, 2)
    # neck midpoint
    assert np.allclose(b25[1], [5, 0])
    # midhip midpoint
    assert np.allclose(b25[8], [5, 10])


def test_compose_wb86_shapes():
    body25 = np.zeros((25, 2), dtype=float)
    lhand = np.zeros((21, 2), dtype=float)
    rhand = np.zeros((21, 2), dtype=float)
    face68 = np.zeros((68, 2), dtype=float)
    out = compose_wb86(body25, lhand, rhand, face68)
    assert out["keypoints"].shape == (86, 2)
    assert out["mask"].shape == (86,)

