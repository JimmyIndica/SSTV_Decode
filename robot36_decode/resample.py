"""Resampling utilities for audio/IQ signals."""

from __future__ import annotations

from math import gcd

import numpy as np

from .filters import resample_poly


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a real or complex signal to a new sample rate.

    Uses polyphase filtering with the reduced up/down ratio, which is far
    cheaper in time and memory than a whole-signal FFT resample and works
    unchanged for real and complex input.

    Args:
        samples: Input signal (real or complex)
        src_rate: Source sample rate in Hz
        dst_rate: Target sample rate in Hz (typically 48000)

    Returns:
        Resampled signal at dst_rate
    """
    if src_rate == dst_rate:
        return samples
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    divisor = gcd(int(src_rate), int(dst_rate))
    up = int(dst_rate) // divisor
    down = int(src_rate) // divisor
    return resample_poly(samples, up, down)
