"""FIR design, FFT convolution and polyphase resampling on numpy + scipy.fft.

These replace the handful of ``scipy.signal`` functions the decoder used
(``firwin``, ``oaconvolve``, ``resample_poly``, ``medfilt``). Importing
``scipy.signal`` costs about a second (it builds documentation for hundreds
of functions at import time), which was a large part of a single decode;
``scipy.fft`` imports in a fraction of that and is also several times faster
than ``numpy.fft`` on the awkward transform lengths real recordings have.

Each function computes the same thing as its ``scipy.signal`` counterpart
(the tests compare them), differing only by floating-point rounding.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import scipy.fft as sfft

# Batched FFTs (many blocks at once) are split over all CPU cores.
WORKERS = -1


def firwin(
    numtaps: int,
    cutoff: float | list[float] | tuple[float, ...],
    *,
    fs: float,
    window: str | tuple[str, float] = "hamming",
    pass_zero: bool = True,
) -> np.ndarray:
    """Windowed-sinc FIR filter, as ``scipy.signal.firwin`` (scale=True).

    ``window`` is ``"hamming"`` or ``("kaiser", beta)``. A single ``cutoff``
    with ``pass_zero=True`` is a low-pass; two cutoffs with
    ``pass_zero=False`` a band-pass.
    """
    edges = np.atleast_1d(np.asarray(cutoff, dtype=np.float64)) / (0.5 * float(fs))
    if np.any(edges <= 0.0) or np.any(edges >= 1.0):
        raise ValueError("cutoff frequencies must lie strictly between 0 and fs/2")
    pass_nyquist = bool(edges.size & 1) ^ bool(pass_zero)
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError("a filter that passes fs/2 needs an odd number of taps")
    edges = np.hstack(([0.0] * bool(pass_zero), edges, [1.0] * pass_nyquist))
    bands = edges.reshape(-1, 2)
    m = np.arange(numtaps, dtype=np.float64) - 0.5 * (numtaps - 1)
    h = np.zeros(numtaps, dtype=np.float64)
    for left, right in bands:
        h += right * np.sinc(right * m)
        h -= left * np.sinc(left * m)
    if window == "hamming":
        h *= np.hamming(numtaps)
    elif isinstance(window, tuple) and len(window) == 2 and window[0] == "kaiser":
        h *= np.kaiser(numtaps, float(window[1]))
    else:
        raise ValueError(f"unsupported window {window!r}")
    left, right = bands[0]
    if left == 0.0:
        scale_frequency = 0.0
    elif right == 1.0:
        scale_frequency = 1.0
    else:
        scale_frequency = 0.5 * (left + right)
    h /= np.sum(h * np.cos(np.pi * m * scale_frequency))
    return h


def convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution of a long 1-D signal with a short FIR, ``mode="same"``.

    Overlap-add with batched FFTs (``scipy.signal.oaconvolve`` computes the
    same). The working precision follows the inputs, so ``complex64`` data
    and ``float32`` taps stay in single precision.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    n, m = x.size, h.size
    if n == 0 or m == 0:
        return np.zeros(n, dtype=np.result_type(x, h))
    dtype = np.result_type(x, h, np.float32)
    nfft = sfft.next_fast_len(max(8 * m, 1024), real=not np.iscomplexobj(x))
    block = nfft - m + 1
    blocks = -(-n // block)
    padded = np.zeros(blocks * block, dtype=dtype)
    padded[:n] = x
    padded = padded.reshape(blocks, block)
    if np.iscomplexobj(padded):
        spectrum = sfft.fft(padded, nfft, axis=1, workers=WORKERS)
        spectrum *= sfft.fft(h.astype(dtype), nfft)
        y = sfft.ifft(spectrum, axis=1, workers=WORKERS, overwrite_x=True)
    else:
        spectrum = sfft.rfft(padded, nfft, axis=1, workers=WORKERS)
        spectrum *= sfft.rfft(h.astype(dtype), nfft)
        y = sfft.irfft(spectrum, nfft, axis=1, workers=WORKERS, overwrite_x=True)
    full = np.zeros((blocks + 1) * block, dtype=y.dtype)
    full[: blocks * block] = y[:, :block].reshape(-1)
    if m > 1:
        tails = full[block:].reshape(blocks, block)
        tails[:, : m - 1] += y[:, block : block + m - 1]
    start = (m - 1) // 2
    return full[start : start + n]


def _upfirdn_part(h: np.ndarray, x: np.ndarray, up: int, down: int, first: int, count: int) -> np.ndarray:
    """Outputs ``first .. first+count-1`` of ``upfirdn(h, x, up, down)``.

    Polyphase: output ``k`` reads input ``q = k*down // up`` backwards through
    phase ``r = k*down % up`` of the filter. Outputs that share a phase are
    ``up`` apart and read inputs ``down`` apart, so each phase is a single
    strided matrix-vector product.
    """
    taps = -(-h.size // up)
    phases = np.zeros((up, taps), dtype=np.float64)
    hp = np.zeros(up * taps, dtype=np.float64)
    hp[: h.size] = h
    phases[:] = hp.reshape(taps, up).T  # phases[r, l] = h[r + l*up]
    reversed_phases = phases[:, ::-1]
    xpad = np.concatenate((np.zeros(taps - 1, dtype=x.dtype), x, np.zeros(taps, dtype=x.dtype)))
    windows = np.lib.stride_tricks.sliding_window_view(xpad, taps)  # windows[q] = x[q-taps+1 .. q]
    out = np.zeros(count, dtype=np.result_type(x, np.float64))
    for offset in range(min(up, count)):
        k0 = first + offset
        q0, r = divmod(k0 * down, up)
        rows = windows[q0 : q0 + ((count - offset - 1) // up) * down + 1 : down]
        out[offset::up] = rows @ reversed_phases[r]
    return out


def resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Polyphase resampling by ``up/down``, as ``scipy.signal.resample_poly``
    with its default Kaiser (beta 5) anti-aliasing filter and zero padding."""
    x = np.asarray(x)
    divisor = gcd(int(up), int(down))
    up, down = int(up) // divisor, int(down) // divisor
    if up == down == 1:
        return x.copy()
    n_in = x.size
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)
    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, 1.0 / max_rate, fs=2.0, window=("kaiser", 5.0)) * up
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down

    def output_len(len_h: int) -> int:
        return ((n_in - 1) * up + len_h - 1) // down + 1

    while output_len(h.size + n_pre_pad + n_post_pad) < n_out + n_pre_remove:
        n_post_pad += 1
    h = np.concatenate((np.zeros(n_pre_pad), h, np.zeros(n_post_pad)))
    if np.iscomplexobj(x):
        xr = np.ascontiguousarray(x.real, dtype=np.float64)
        xi = np.ascontiguousarray(x.imag, dtype=np.float64)
        return _upfirdn_part(h, xr, up, down, n_pre_remove, n_out) + 1j * _upfirdn_part(
            h, xi, up, down, n_pre_remove, n_out
        )
    return _upfirdn_part(h, np.asarray(x, dtype=np.float64), up, down, n_pre_remove, n_out)


def medfilt(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Running median with zero padding at the ends, as ``scipy.signal.medfilt``."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    x = np.asarray(x, dtype=np.float64)
    half = kernel_size // 2
    padded = np.concatenate((np.zeros(half), x, np.zeros(half)))
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, kernel_size), axis=1)
