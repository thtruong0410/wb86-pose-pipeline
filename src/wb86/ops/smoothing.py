from __future__ import annotations

import math
import numpy as np


class OneEuroFilter:
    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    @staticmethod
    def _exp_smooth(a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        # estimate derivative
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = self._exp_smooth(a_d, dx, self.dx_prev)
        # filtered value
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = self._exp_smooth(a, x, self.x_prev)
        # update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class ArrayOneEuro:
    """One-Euro filter applied to arrays of shape (..., D).

    Missing values (NaN) are passed through without updating state.
    """

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.filters = None

    def _ensure(self, dim: int):
        if self.filters is None or len(self.filters) != dim:
            self.filters = [OneEuroFilter(self.freq, self.min_cutoff, self.beta, self.d_cutoff) for _ in range(dim)]

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            arr = arr[None, :]
        out = arr.copy()
        d = out.shape[-1]
        self._ensure(d)
        flat = out.reshape(-1, d)
        for i in range(flat.shape[0]):
            for j in range(d):
                v = flat[i, j]
                if np.isnan(v):
                    continue
                flat[i, j] = self.filters[j](float(v))
        return out.reshape(arr.shape)

