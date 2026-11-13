"""Estimate where a Robot 36 image starts in a poor capture.

Used by ``--find-start``, ``--force-start auto`` and the automatic rescue of
files that fail to decode in directory mode. It works on demodulated 48 kHz
audio (or a complex I/Q audio signal) and combines three independent clues:

1. **Sync comb.** Every line starts with a 9 ms 1200 Hz pulse, so folding the
   "is this 1200 Hz?" indicator at the 150 ms line period gives a sharp peak
   even when single pulses are lost in noise. The peak fixes the line phase
   to about a millisecond, but not *which* line is line 0.
2. **Separator parity.** The 4.5 ms separator after each luma line is 1500 Hz
   on even lines and 2300 Hz on odd lines, which fixes whether a candidate
   start is an even or an odd number of lines out.
3. **Header and pulse train.** Line 0 is preceded by the VIS header: 600 ms of
   1900 Hz leader, a 30 ms 1200 Hz start bit two lines before line 0 and a
   30 ms 1200 Hz stop bit just before it. A template of that pattern, plus the
   step up in sync-pulse strength where the picture begins, picks the line
   among the candidates the comb and parity allow. (The pulse step alone is
   not enough: the start bit and data bits look like sync pulses and make the
   train seem to start two lines early.) If a readable VIS header is available
   it decides directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.fft as sfft

from .constants import LINE_MS, ROBOT36_VIS_CODE

LINE_S = LINE_MS / 1000.0
VIS_S = 0.910  # leader start -> first line
IMAGE_S = 36.0
PULSE_S = 0.009
SEPARATOR_WINDOW_S = (0.1008, 0.1037)  # inside the 4.5 ms separator, 100-104.5 ms
SEARCH_SPAN_S = 1.2  # candidate starts this far either side of the expected one
PARITY_LINES = 60
STEP_LINES = 6

MIN_COMB_RATIO = 5.0
GOOD_COMB_RATIO = 10.0
MIN_PARITY_HZ = 150.0
GOOD_PARITY_HZ = 400.0
VIS_SNAP_S = 0.075


@dataclass
class StartEstimate:
    """Result of a start search. Times are seconds on the capture's clock."""

    start_s: float | None  # first line sync; None when nothing was found
    confidence: str  # HIGH / MEDIUM / LOW / NONE
    onset_s: float
    comb_phase_s: float | None = None  # line starts at comb_phase_s + n * 0.150
    comb_ratio: float = 0.0  # sync-comb peak over background
    parity_contrast_hz: float = 0.0  # odd minus even separator tone
    vis_found: bool = False
    alternatives: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _hilbert(x: np.ndarray) -> np.ndarray:
    """Analytic signal without padding (as ``scipy.signal.hilbert``)."""
    n = x.size
    spectrum = sfft.fft(x)
    gain = np.zeros(n)
    if n % 2 == 0:
        gain[0] = gain[n // 2] = 1.0
        gain[1 : n // 2] = 2.0
    else:
        gain[0] = 1.0
        gain[1 : (n + 1) // 2] = 2.0
    return sfft.ifft(spectrum * gain)


def _instantaneous_frequency(samples: np.ndarray, rate: int) -> np.ndarray:
    from scipy import ndimage  # only needed here; keeps the normal decode's imports light

    analytic = samples if np.iscomplexobj(samples) else _hilbert(samples)
    freq = np.gradient(np.unwrap(np.angle(analytic))) * rate / (2.0 * np.pi)
    # A median filter that pads with zeros, like scipy.signal.medfilt.
    return ndimage.median_filter(freq, int(0.0015 * rate) | 1, mode="constant")


def find_start(
    samples: np.ndarray,
    rate: int,
    t0_s: float,
    onset_s: float,
    vis=None,
) -> StartEstimate:
    """Find the start of line 0.

    ``samples`` is audio (real) or an I/Q audio signal (complex) at ``rate``;
    ``t0_s`` is the capture time of ``samples[0]``; ``onset_s`` the capture
    time where the signal appears; ``vis`` an optional detected
    :class:`~robot36_decode.vis.VisInfo` (relative to ``samples``).
    """
    est = StartEstimate(start_s=None, confidence="NONE", onset_s=onset_s)
    line = int(round(LINE_S * rate))
    i0 = max(0, int((onset_s - t0_s - 0.5) * rate))
    i1 = min(samples.size, int((onset_s - t0_s + VIS_S + IMAGE_S + 0.5) * rate))
    if i1 - i0 < 8 * line:
        est.notes.append("too little audio after the signal onset")
        return est

    freq = _instantaneous_frequency(samples, rate)
    indicator = np.exp(-(((freq - 1200.0) / 120.0) ** 2))
    from scipy import ndimage

    indicator = ndimage.uniform_filter1d(indicator, max(1, int(PULSE_S * rate)))

    # --- 1. sync comb -------------------------------------------------------
    folds = (i1 - i0) // line
    fold = indicator[i0 : i0 + folds * line].reshape(folds, line).mean(axis=0)
    peak = int(np.argmax(fold))
    background = max(float(np.median(fold)), 1e-3)
    est.comb_ratio = float(fold[peak] / background)
    if est.comb_ratio < MIN_COMB_RATIO:
        est.notes.append(
            f"no periodic 1200 Hz sync pulses found (comb {est.comb_ratio:.1f}x background)"
        )
        return est
    spread = int(0.006 * rate)
    offsets = np.arange(-spread, spread + 1)
    weights = np.clip(fold[(peak + offsets) % line] - 0.5 * fold[peak], 0.0, None)
    centre = peak + float(np.sum(weights * offsets) / np.sum(weights))
    # The pulse indicator is centred on the 9 ms pulse; line starts 4.5 ms earlier.
    est.comb_phase_s = (i0 + centre) / rate + t0_s - PULSE_S / 2.0

    def pulse_strength(start_s: float) -> float:
        index = int(round((start_s + PULSE_S / 2.0 - t0_s) * rate))
        return float(indicator[index]) if 0 <= index < indicator.size else 0.0

    def separator_contrast(start_s: float) -> float:
        even, odd = [], []
        for m in range(PARITY_LINES):
            a = int(round((start_s + m * LINE_S + SEPARATOR_WINDOW_S[0] - t0_s) * rate))
            b = int(round((start_s + m * LINE_S + SEPARATOR_WINDOW_S[1] - t0_s) * rate))
            if a < 0 or b > freq.size:
                continue
            (odd if m & 1 else even).append(float(np.median(freq[a:b])))
        if len(even) < 5 or len(odd) < 5:
            return 0.0
        return float(np.median(odd) - np.median(even))

    def near_fraction(a_s: float, b_s: float, target_hz: float, tolerance_hz: float = 150.0) -> float:
        a = int(round((a_s - t0_s) * rate))
        b = int(round((b_s - t0_s) * rate))
        if a < 0 or b > freq.size or b <= a:
            return 0.0
        return float(np.mean(np.abs(freq[a:b] - target_hz) < tolerance_hz))

    def header_score(start_s: float) -> float:
        """How much the 910 ms before ``start_s`` looks like a VIS header."""
        leader = 0.5 * (
            near_fraction(start_s - 0.900, start_s - 0.615, 1900.0)
            + near_fraction(start_s - 0.590, start_s - 0.310, 1900.0)
        )
        bits = 0.5 * (
            near_fraction(start_s - 0.297, start_s - 0.273, 1200.0)
            + near_fraction(start_s - 0.027, start_s - 0.003, 1200.0)
        )
        return leader + 0.5 * bits

    def step(start_s: float) -> float:
        after = np.mean([pulse_strength(start_s + j * LINE_S) for j in range(STEP_LINES)])
        before = np.mean([pulse_strength(start_s - j * LINE_S) for j in range(1, STEP_LINES + 1)])
        return float(after - before)

    # --- 2./3. choose the line -----------------------------------------------
    if vis is not None and vis.code == ROBOT36_VIS_CODE and vis.parity_ok is not False:
        vis_start = t0_s + vis.image_start_ms / 1000.0
        n = round((vis_start - est.comb_phase_s) / LINE_S)
        snapped = est.comb_phase_s + n * LINE_S
        est.vis_found = True
        est.start_s = snapped if abs(snapped - vis_start) <= VIS_SNAP_S else vis_start
        est.parity_contrast_hz = separator_contrast(est.start_s)
        est.confidence = "HIGH"
        est.notes.append("start taken from the VIS header and refined with the sync comb")
        return est

    anchor = onset_s + VIS_S
    n_lo = int(np.ceil((anchor - SEARCH_SPAN_S - est.comb_phase_s) / LINE_S))
    n_hi = int(np.floor((anchor + SEARCH_SPAN_S - est.comb_phase_s) / LINE_S))
    candidates = [est.comb_phase_s + n * LINE_S for n in range(n_lo, n_hi + 1)]
    candidates = [c for c in candidates if c >= t0_s]
    if not candidates:
        est.notes.append("no candidate start inside the audio")
        return est
    contrast = {c: separator_contrast(c) for c in candidates}
    steps = {c: header_score(c) + step(c) for c in candidates}
    valid = [c for c in candidates if contrast[c] >= MIN_PARITY_HZ]
    pool = valid or candidates
    ranked = sorted(pool, key=lambda c: steps[c], reverse=True)
    best = ranked[0]
    est.start_s = best
    est.parity_contrast_hz = contrast[best]
    est.alternatives = [c for c in ranked[1:4]]
    second = steps[ranked[1]] if len(ranked) > 1 else 0.0
    unique = steps[best] > 0 and (second <= 0 or steps[best] >= 1.3 * second)
    if not valid:
        est.confidence = "LOW"
        est.notes.append(
            "separator tones do not show a clear even/odd pattern; the start may be "
            "one line out (try the alternatives)"
        )
    elif (
        est.comb_ratio >= GOOD_COMB_RATIO
        and contrast[best] >= GOOD_PARITY_HZ
        and unique
    ):
        est.confidence = "HIGH"
    elif steps[best] > 0:
        est.confidence = "MEDIUM"
        if not unique:
            est.notes.append(
                "the header and the start of the pulse train are not clear-cut; "
                "neighbours two lines away score similarly (see the alternatives)"
            )
    else:
        est.confidence = "LOW"
        est.notes.append("could not see where the pulse train begins")
    return est


def format_report(est: StartEstimate, command: str | None) -> str:
    """Human-readable report for ``--find-start``."""
    lines: list[str] = []
    lines.append(f"Signal:     onset about {est.onset_s:.1f} s")
    if est.comb_phase_s is None:
        lines.append(f"Line sync:  not found ({'; '.join(est.notes)})")
        lines.append("Confidence: NONE")
        return "\n".join(lines)
    lines.append(
        f"Line sync:  comb found, lines start at {est.comb_phase_s % LINE_S:.4f} s "
        f"+ n x {LINE_S:.3f} s (strength {est.comb_ratio:.0f}x background)"
    )
    lines.append(
        f"Parity:     separator contrast {est.parity_contrast_hz:+.0f} Hz "
        "(even lines 1500 Hz, odd lines 2300 Hz)"
    )
    if est.start_s is not None:
        source = "VIS header" if est.vis_found else "header pattern and start of the sync pulse train"
        lines.append(f"Best start: {est.start_s:.4f} s  (from the {source})")
    lines.append(f"Confidence: {est.confidence}")
    for note in est.notes:
        lines.append(f"Note:       {note}")
    if est.alternatives:
        alt = ", ".join(f"{a:.4f} s" for a in est.alternatives)
        lines.append(f"Next best:  {alt}")
    if command and est.start_s is not None:
        lines.append("")
        lines.append(f"Try:  {command}")
        lines.append("Adjust by +/-0.150 s only if the picture looks wrong.")
    return "\n".join(lines)
