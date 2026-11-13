"""Reference-free image quality metrics for decoded PNGs."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def to_ycc(rgb: np.ndarray):
    rgb = rgb.astype(np.float64)
    y = rgb @ np.array([0.299, 0.587, 0.114])
    return y, 128 + (rgb[..., 2] - y) / 1.772, 128 + (rgb[..., 0] - y) / 1.402


def _lap_noise(a: np.ndarray) -> float:
    lap = ndi.laplace(a)
    return float(1.4826 * np.median(np.abs(lap - np.median(lap))))


def quality(rgb: np.ndarray) -> dict:
    """Noise (robust Laplacian MAD) and count of rows far from both neighbours."""
    y, cb, cr = to_ycc(rgb)
    dev = np.abs(y[1:-1] - 0.5 * (y[:-2] + y[2:])).mean(axis=1)
    return {
        "bad_rows": int((dev > 25).sum()),
        "noise_y": _lap_noise(y),
        "noise_c": 0.5 * (_lap_noise(cb) + _lap_noise(cr)),
    }
