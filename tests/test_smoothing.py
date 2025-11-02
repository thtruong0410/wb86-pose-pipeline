import numpy as np

from wb86.ops.smoothing import ArrayOneEuro


def test_array_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    signal = np.cumsum(rng.normal(0, 0.1, size=(50, 2)), axis=0)
    noisy = signal + rng.normal(0, 0.3, size=signal.shape)
    f = ArrayOneEuro(freq=25.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    out = np.vstack([f(noisy[i]) for i in range(noisy.shape[0])])
    var_in = np.var(noisy, axis=0).mean()
    var_out = np.var(out, axis=0).mean()
    assert var_out < var_in

