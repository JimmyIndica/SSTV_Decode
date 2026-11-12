"""Reference-free and reference-based image quality metrics for decoded PNGs."""

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


def compare(rgb: np.ndarray, ref: np.ndarray) -> dict:
    """Luma/chroma MAE versus a reference after a best sub-pixel horizontal shift.

    The shift only affects the *measurement*: the decoder keeps spec timing.
    """
    y, cb, cr = to_ycc(rgb)
    ry, rcb, rcr = to_ycc(ref)
    best = (1e9, 0.0)
    for dx in np.arange(-3.0, 3.01, 0.25):
        s = ndi.shift(y, (0, dx), order=1, mode="nearest")
        m = float(np.abs(ry[:, 12:-12] - s[:, 12:-12]).mean())
        if m < best[0]:
            best = (m, float(dx))
    m, dx = best
    sh = lambda a: ndi.shift(a, (0, dx), order=1, mode="nearest")  # noqa: E731
    return {
        "y_mae": m,
        "shift_px": dx,
        "cb_mae": float(np.abs(rcb[:, 12:-12] - sh(cb)[:, 12:-12]).mean()),
        "cr_mae": float(np.abs(rcr[:, 12:-12] - sh(cr)[:, 12:-12]).mean()),
    }
