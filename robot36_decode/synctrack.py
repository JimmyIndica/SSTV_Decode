"""Sync-pulse tracking for recordings that are not perfectly timed.

Robot 36 sends a 9 ms 1200 Hz sync pulse at the start of every 150 ms line.
Finding those pulses lets the decoder tolerate:

* leading silence or unrelated audio before the image,
* an arbitrary start offset,
* a sample-clock or tape-speed error that stretches or shrinks the line time,
* small line-to-line timing jitter,
* a constant or slowly drifting frequency offset (Doppler shift, tuning
  error): the sync tone is searched on a frequency grid, its frequency is
  measured on every tracked line, and the track is re-run following the
  fitted sync-frequency curve so the whole image is covered.

The tracker returns one start time per image line, in milliseconds relative to
the first sample it was given, plus a per-line frequency offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import FREQ_BLACK, FREQ_SYNC, LINE_MS, LINES, SYNC_MS
from .dsp import (
    centered_frequencies,
    downward_crossing,
    phase,
    phase_values,
    true_runs,
    upward_crossing,
)

# A 9 ms sync pulse appears roughly 2.5 ms longer in the centred-frequency
# detector because the measurement window blends in the neighbouring tones.
SYNC_PULSE_MIN_MS = 4.0
SYNC_PULSE_MAX_MS = 17.0
SYNC_DETECT_TOLERANCE_HZ = 110.0
SYNC_TOLERANCE_MS = 15.0
MIN_SYNC_LINES = 8
GOOD_TRACK_FRACTION = 0.9

# Sync-tone search grid (Doppler / tuning offset recovery).
GRID_HALF_RANGE_HZ = 1500.0
GRID_STEP_HZ = 60.0
GRID_MAX_EVALUATED = 12
MAX_RETRACK_ITERATIONS = 10
# Each re-track round may only extend the confirmed line range by this many
# lines on either side, so an extrapolated fit cannot wander into unrelated
# tones (line separators, chroma) that happen to sit near it.
RETRACK_EXTEND_LINES = 40

# The sync-to-porch step: the porch that follows sync is FREQ_BLACK.
SYNC_PORCH_DELTA_HZ = FREQ_BLACK - FREQ_SYNC

# Interior of the sync pulse used to measure its tone frequency (ms from the
# pulse start, before time-scale correction), and the part of the 3 ms porch
# that follows it, used to tell real sync pulses from look-alikes.
SYNC_MEASURE_START_MS = 2.0
SYNC_MEASURE_END_MS = 7.0
PORCH_MEASURE_START_MS = 9.6
PORCH_MEASURE_END_MS = 11.4


@dataclass
class SyncTrack:
    line_starts_ms: np.ndarray
    matched_lines: int
    pulse_count: int
    drift_ms_per_line: float
    line_scale: float
    # Measured sync-tone frequency (Hz) at the middle of the image.
    sync_hz: float = FREQ_SYNC
    # Per-line frequency offset (Hz) relative to the nominal, time-scaled
    # tones: ``measured sync - FREQ_SYNC / line_scale``. ``None`` if unknown.
    line_offsets_hz: np.ndarray | None = None
    # Line numbers that matched a detected pulse.
    matched_idx: np.ndarray | None = None
    # (poly coefficients, first line, last line) of the sync-frequency fit.
    sync_fit: tuple | None = None

    @property
    def tone_offset_hz(self) -> float:
        """Representative (median) frequency offset of the whole image."""
        if self.line_offsets_hz is None:
            return 0.0
        return float(np.median(self.line_offsets_hz))


# ---------------------------------------------------------------------------
# Pulse detection
# ---------------------------------------------------------------------------


def _refine_sync_start(
    phase_track: np.ndarray,
    sample_rate: int,
    approx_ms: float,
    sync_duration_ms: float = SYNC_MS,
    tone_offset_hz: float = 0.0,
    sync_hz: float | None = None,
) -> float:
    """Refine a coarse pulse time from the sync-to-porch transition.

    Sync is 1200 Hz and the porch that follows it is 1500 Hz. The midpoint
    crossing between those tones is symmetric, so measuring the end of the
    pulse and subtracting the sync duration is much less sensitive to the
    preceding chroma tone than measuring the start directly.

    ``sync_hz`` is the measured/assumed sync tone; the porch is placed the
    nominal sync-to-porch step above it.
    """
    sync_end = approx_ms + sync_duration_ms
    start = max(0.0, sync_end - 3.0)
    end = sync_end + 3.0
    if end - start < 0.5:
        return max(0.0, approx_ms)

    if sync_hz is None:
        sync_hz = FREQ_SYNC + tone_offset_hz
    porch_hz = sync_hz + SYNC_PORCH_DELTA_HZ
    midpoint_hz = (sync_hz + porch_hz) / 2.0
    times, frequencies = centered_frequencies(
        phase_track, sample_rate, start, end, 0.05, 0.2
    )
    crossing = upward_crossing(times, frequencies, midpoint_hz)
    if crossing is not None:
        return max(0.0, crossing - sync_duration_ms)

    # Fallback: look for the downward transition into the pulse.
    start = max(0.0, approx_ms - 3.0)
    end = approx_ms + 3.0
    times, frequencies = centered_frequencies(
        phase_track, sample_rate, start, end, 0.05, 0.2
    )
    crossing = downward_crossing(times, frequencies, midpoint_hz)
    if crossing is not None:
        return max(0.0, crossing)

    below = np.flatnonzero(frequencies < midpoint_hz)
    if below.size:
        return max(0.0, float(times[below[0]]))
    return max(0.0, approx_ms)


def _coarse_pulses(
    series: tuple[np.ndarray, np.ndarray],
    sync_hz,
    tolerance_hz: float = SYNC_DETECT_TOLERANCE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Find approximate pulse positions, before the sync-duration correction.

    ``series`` is ``(times_ms, frequencies_hz)`` from ``centered_frequencies``.
    ``sync_hz`` is a scalar, or a callable mapping times (ms) to the expected
    sync frequency, which lets the search follow a drifting tone. Returns the
    candidate times and the expected sync frequency at each.
    """
    times, frequencies = series
    if callable(sync_hz):
        centre = np.asarray(sync_hz(times), dtype=np.float64)
    else:
        centre = np.full(times.shape, float(sync_hz))
    mask = np.abs(frequencies - centre) < tolerance_hz

    candidates: list[float] = []
    centres: list[float] = []
    for first, last in true_runs(mask):
        duration = float(last - first)
        if SYNC_PULSE_MIN_MS <= duration <= SYNC_PULSE_MAX_MS:
            candidates.append(float(times[first]))
            centres.append(float(centre[first]))
    return np.asarray(candidates, dtype=np.float64), np.asarray(centres, dtype=np.float64)


def _porch_step_ok(
    phase_track: np.ndarray, sample_rate: int, start_ms: float, scale: float
) -> bool:
    """Check that the tone after a pulse is about one sync-to-porch step higher.

    A real sync (1200 Hz) is followed by the 1500 Hz porch, i.e. +300 Hz,
    whatever the Doppler offset. Look-alikes -- a dark picture area that runs
    into the 1500 Hz line separator (+0 Hz), or a pulse before the 2300 Hz
    separator (+800 Hz) -- fail this, which keeps them out of the line grid.
    """
    times = np.array(
        [
            start_ms + SYNC_MEASURE_START_MS * scale,
            start_ms + SYNC_MEASURE_END_MS * scale,
            start_ms + PORCH_MEASURE_START_MS * scale,
            start_ms + PORCH_MEASURE_END_MS * scale,
        ]
    )
    p = phase_values(phase_track, sample_rate, times)
    tau = 1000.0 / (2.0 * np.pi)
    f_sync = (p[1] - p[0]) * tau / ((SYNC_MEASURE_END_MS - SYNC_MEASURE_START_MS) * scale)
    f_porch = (p[3] - p[2]) * tau / ((PORCH_MEASURE_END_MS - PORCH_MEASURE_START_MS) * scale)
    step = SYNC_PORCH_DELTA_HZ / scale
    return 0.5 * step < (f_porch - f_sync) < 1.5 * step


def _refine_pulses(
    phase_track: np.ndarray,
    sample_rate: int,
    candidates: np.ndarray,
    centres: np.ndarray,
    sync_duration_ms: float,
) -> np.ndarray:
    scale = sync_duration_ms / SYNC_MS
    pulses = []
    for value, centre in zip(candidates, centres):
        start = _refine_sync_start(
            phase_track, sample_rate, float(value), sync_duration_ms, sync_hz=float(centre)
        )
        if _porch_step_ok(phase_track, sample_rate, start, scale):
            pulses.append(start)
    if not pulses:
        return np.empty(0, dtype=np.float64)

    # Ignore duplicate detections of the same pulse.
    pulses.sort()
    unique = [pulses[0]]
    for value in pulses[1:]:
        if value - unique[-1] >= 20.0:
            unique.append(value)
    return np.asarray(unique, dtype=np.float64)


def _pulses_from_candidates(
    phase_track: np.ndarray,
    sample_rate: int,
    candidates: np.ndarray,
    centres: np.ndarray,
) -> np.ndarray:
    if candidates.size == 0:
        return np.empty(0, dtype=np.float64)

    pulses = _refine_pulses(phase_track, sample_rate, candidates, centres, SYNC_MS)

    # If the recording is time-scaled, the sync pulse itself is scaled too.
    # Re-refine using the measured line interval so the absolute start times do
    # not acquire a small, constant offset. The expected sync frequency per
    # candidate (``centres``) must be passed through unchanged here.
    interval = _estimate_line_interval(pulses)
    scaled_sync = SYNC_MS * interval / LINE_MS
    if abs(scaled_sync - SYNC_MS) > 0.001:
        pulses = _refine_pulses(
            phase_track, sample_rate, candidates, centres, float(scaled_sync)
        )
    return pulses


def _find_pulses(
    phase_track: np.ndarray,
    sample_rate: int,
    duration_ms: float,
    sync_hz: float,
    tone_offset_hz: float = 0.0,
    series: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return refined sync pulses for a candidate sync tone frequency."""
    if series is None:
        series = centered_frequencies(phase_track, sample_rate, 0.0, duration_ms, 1.0, 2.5)
    candidates, centres = _coarse_pulses(series, sync_hz)
    return _pulses_from_candidates(phase_track, sample_rate, candidates, centres)


def _sync_grid_frequencies(
    series: tuple[np.ndarray, np.ndarray], nominal_hz: float
) -> list[float]:
    """Return sync-tone frequencies worth trying, best first.

    Scans a band around the nominal sync tone and keeps the local maxima of
    the number of pulse-like runs, so neighbouring grid points that see the
    same pulses are not all evaluated.
    """
    scan = nominal_hz + np.arange(
        -GRID_HALF_RANGE_HZ, GRID_HALF_RANGE_HZ + 1e-9, GRID_STEP_HZ
    )
    scan = scan[scan >= 300.0]
    counts = np.array([_coarse_pulses(series, hz)[0].size for hz in scan])
    padded = np.concatenate(([-1], counts, [-1]))
    peaks = [
        i
        for i in range(scan.size)
        if counts[i] >= MIN_SYNC_LINES
        and counts[i] >= padded[i]
        and counts[i] >= padded[i + 2]
        and abs(scan[i] - nominal_hz) > 1e-6
    ]
    peaks.sort(key=lambda i: counts[i], reverse=True)
    return [float(scan[i]) for i in peaks]


def find_sync_pulses(
    samples: np.ndarray,
    sample_rate: int,
    expected_start_ms: float | None = None,
    tone_offset_hz: float = 0.0,
    phase_track: np.ndarray | None = None,
) -> np.ndarray:
    """Return the start times (ms) of candidate Robot 36 line sync pulses.

    ``tone_offset_hz`` shifts the sync tone frequency to follow a Doppler-shifted
    signal. ``expected_start_ms`` is accepted for API symmetry with
    ``track_line_starts`` but is not used: pulse detection is independent of
    where the image starts.
    """
    duration_ms = 1000.0 * samples.size / sample_rate
    if duration_ms < SYNC_PULSE_MIN_MS:
        return np.empty(0, dtype=np.float64)

    if phase_track is None:
        phase_track = phase(samples, sample_rate, tone_offset_hz=tone_offset_hz)
    sync_hz = FREQ_SYNC + tone_offset_hz
    return _find_pulses(phase_track, sample_rate, duration_ms, sync_hz, tone_offset_hz)


# ---------------------------------------------------------------------------
# Pulse grid -> line track
# ---------------------------------------------------------------------------


def _estimate_line_interval(pulses: np.ndarray) -> float:
    """Estimate the average line time, allowing for missing pulses."""
    gaps = np.diff(pulses)
    candidates: list[float] = []
    for gap in gaps:
        multiple = int(round(gap / LINE_MS))
        if multiple < 1 or multiple > 8:
            continue
        interval = gap / multiple
        if abs(interval - LINE_MS) <= 0.12 * LINE_MS:
            candidates.append(interval)
    if not candidates:
        return LINE_MS
    return float(np.median(candidates))


PHASE_AGREE_MS = 8.0


def _choose_base(
    pulses: np.ndarray,
    interval: float,
    expected_start_ms: float | None,
    expected_end_ms: float | None = None,
) -> float:
    """Choose the pulse grid phase that corresponds to image line 0.

    The grid phase is chosen by agreement: every real sync pulse sits at the
    same position modulo the line time, so the phase supported by the most
    pulses wins. Stray look-alike pulses (which have some other phase) cannot
    capture the grid just by being close to the expected start. Only after the
    phase is fixed does ``expected_start_ms`` pick which line is line 0.
    """
    best_support = -1
    best_phase_pulse = float(pulses[0])
    for candidate in pulses[: min(pulses.size, 400)]:
        relative = pulses - candidate
        error = np.abs(relative - np.rint(relative / interval) * interval)
        support = int(np.count_nonzero(error <= PHASE_AGREE_MS))
        if support > best_support:
            best_support = support
            best_phase_pulse = float(candidate)

    relative = pulses - best_phase_pulse
    error = np.abs(relative - np.rint(relative / interval) * interval)
    aligned = pulses[error <= PHASE_AGREE_MS]

    if expected_start_ms is not None:
        # Snap the expected start onto the agreed pulse grid. This works even
        # when line 0 itself is missing (for example when its sync is merged
        # with the VIS stop bit) or when --start-ms is only approximate.
        best_base: float | None = None
        best_distance = float("inf")
        for pulse in aligned:
            multiple = round((pulse - expected_start_ms) / interval)
            candidate = float(pulse - multiple * interval)
            distance = abs(candidate - expected_start_ms)
            if distance < best_distance:
                best_distance = distance
                best_base = candidate
        if best_base is not None and best_distance <= 40.0:
            return best_base

    if expected_end_ms is not None and aligned.size:
        # The end of the image is roughly known (for example from where the
        # carrier disappears). Pick the grid position whose last line ends
        # nearest to it: this fixes the line index even when the first lines
        # of a weak signal were not detected.
        target = expected_end_ms - LINES * interval
        pulse = float(aligned[0])
        return pulse + round((target - pulse) / interval) * interval

    # No usable hint: the earliest pulse of the agreed grid is line 0.
    return float(aligned.min())


def _match_lines(
    pulses: np.ndarray, base: float, interval: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map pulse times onto line indices, keeping the best pulse per line."""
    relative = pulses - base
    indices = np.rint(relative / interval).astype(int)
    error = np.abs(relative - indices * interval)
    valid = (indices >= 0) & (indices < LINES) & (error <= SYNC_TOLERANCE_MS)

    best: dict[int, tuple[float, float]] = {}
    for time_ms, index, err in zip(pulses[valid], indices[valid], error[valid]):
        previous = best.get(int(index))
        if previous is None or err < previous[1]:
            best[int(index)] = (float(time_ms), float(err))

    ordered = sorted(best)
    line_numbers = np.asarray(ordered, dtype=np.float64)
    line_times = np.asarray([best[index][0] for index in ordered], dtype=np.float64)
    return line_numbers, line_times


def _robust_fit(
    line_numbers: np.ndarray, line_times: np.ndarray, interval: float, base: float
) -> tuple[float, float]:
    """Least-squares line fit with a few rounds of outlier rejection."""
    slope = interval
    intercept = base
    inliers = np.ones(line_numbers.size, dtype=bool)

    for _ in range(3):
        if np.count_nonzero(inliers) < 2:
            break
        slope, intercept = np.polyfit(line_numbers[inliers], line_times[inliers], 1)
        residual = line_times - (intercept + slope * line_numbers)
        spread = np.median(np.abs(residual[inliers])) * 1.4826
        limit = max(4.0, 3.0 * spread)
        inliers = np.abs(residual) <= limit

    if np.count_nonzero(inliers) >= 2:
        slope, intercept = np.polyfit(line_numbers[inliers], line_times[inliers], 1)
    return float(slope), float(intercept)


def _track_pulses(
    pulses: np.ndarray,
    expected_start_ms: float | None,
    expected_end_ms: float | None = None,
) -> SyncTrack | None:
    """Turn detected sync pulses into a Robot 36 line-start track."""
    if pulses.size < MIN_SYNC_LINES:
        return None

    interval = _estimate_line_interval(pulses)
    base = _choose_base(pulses, interval, expected_start_ms, expected_end_ms)

    # Refine the interval and base from the matched pulses, then re-match.
    for _ in range(2):
        line_numbers, line_times = _match_lines(pulses, base, interval)
        if line_numbers.size < MIN_SYNC_LINES:
            return None
        interval, base = _robust_fit(line_numbers, line_times, interval, base)

    line_numbers, line_times = _match_lines(pulses, base, interval)
    if line_numbers.size < MIN_SYNC_LINES:
        return None

    slope, intercept = _robust_fit(line_numbers, line_times, interval, base)
    if slope <= 0.5 * LINE_MS or slope >= 2.0 * LINE_MS:
        return None

    line_starts = intercept + slope * np.arange(LINES, dtype=np.float64)

    # Prefer the actual detected time for lines where it agrees with the fit.
    residual = line_times - (intercept + slope * line_numbers)
    close = np.abs(residual) <= 5.0
    line_starts[line_numbers[close].astype(int)] = line_times[close]

    if not np.all(np.diff(line_starts) > 0.0):
        line_starts = intercept + slope * np.arange(LINES, dtype=np.float64)

    return SyncTrack(
        line_starts_ms=line_starts,
        matched_lines=int(line_numbers.size),
        pulse_count=int(pulses.size),
        drift_ms_per_line=slope - LINE_MS,
        line_scale=slope / LINE_MS,
        matched_idx=line_numbers.astype(int),
    )


# ---------------------------------------------------------------------------
# Doppler / frequency-offset measurement
# ---------------------------------------------------------------------------


def _eval_fit(coef: np.ndarray, lines: np.ndarray, first: float, last: float) -> np.ndarray:
    """Evaluate the sync-frequency fit; quadratic fits are not extrapolated far."""
    lines = np.asarray(lines, dtype=np.float64)
    if len(coef) > 2:
        lines = np.clip(lines, first - 15.0, last + 15.0)
    return np.polyval(coef, lines)


def _attach_offsets(
    track: SyncTrack, phase_track: np.ndarray, sample_rate: int
) -> SyncTrack:
    """Measure the sync tone on every matched line and fit its frequency.

    A line fit (or a quadratic once there is enough coverage) follows a
    constant offset or a smooth Doppler drift, and rejects outliers.
    """
    scale = track.line_scale
    nominal = FREQ_SYNC / scale
    idx = track.matched_idx
    if idx is None or idx.size < 3:
        track.line_offsets_hz = np.zeros(LINES)
        track.sync_hz = nominal
        return track

    starts = track.line_starts_ms[idx]
    p_a = phase_values(phase_track, sample_rate, starts + SYNC_MEASURE_START_MS * scale)
    p_b = phase_values(phase_track, sample_rate, starts + SYNC_MEASURE_END_MS * scale)
    span_ms = (SYNC_MEASURE_END_MS - SYNC_MEASURE_START_MS) * scale
    freqs = (p_b - p_a) * 1000.0 / (2.0 * np.pi * span_ms)

    x = idx.astype(np.float64)
    degree = 2 if (x.size >= 60 and np.ptp(x) >= 100) else 1
    inliers = np.ones(x.size, dtype=bool)
    coef: np.ndarray | None = None
    for _ in range(3):
        if np.count_nonzero(inliers) < degree + 2:
            break
        coef = np.polyfit(x[inliers], freqs[inliers], degree)
        residual = freqs - np.polyval(coef, x)
        spread = 1.4826 * np.median(np.abs(residual[inliers]))
        inliers = np.abs(residual) <= max(20.0, 3.0 * spread)
    if coef is None:
        coef = np.array([float(np.median(freqs))])

    first, last = float(x.min()), float(x.max())
    lines = np.arange(LINES, dtype=np.float64)
    fitted = _eval_fit(coef, lines, first, last)
    track.sync_fit = (coef, first, last)
    track.line_offsets_hz = fitted - nominal
    track.sync_hz = float(fitted[LINES // 2])
    return track


def _centre_function(track: SyncTrack):
    """Return a function mapping time (ms) to the expected sync frequency."""
    coef, first, last = track.sync_fit
    start0 = float(track.line_starts_ms[0])
    spacing = LINE_MS * track.line_scale

    def centre(times: np.ndarray) -> np.ndarray:
        lines = (np.asarray(times) - start0) / spacing
        value = _eval_fit(coef, np.clip(lines, -LINES, 2 * LINES), first, last)
        # NaN never satisfies the detection test, so nothing is accepted
        # farther than RETRACK_EXTEND_LINES from the confirmed range.
        outside = (lines < first - RETRACK_EXTEND_LINES) | (lines > last + RETRACK_EXTEND_LINES)
        return np.where(outside, np.nan, value)

    return centre


def _finish_track(
    track: SyncTrack,
    phase_track: np.ndarray,
    sample_rate: int,
    series: tuple[np.ndarray, np.ndarray],
    expected_start_ms: float | None,
    expected_end_ms: float | None = None,
) -> SyncTrack:
    """Attach offsets, then re-track along the fitted sync curve.

    The first lock may only cover the part of the image whose sync tone stays
    within the detection tolerance of one frequency. Re-detecting pulses
    around the fitted (possibly drifting) tone extends the track to the rest
    of the image; each round is kept only if it matches more lines.
    """
    track = _attach_offsets(track, phase_track, sample_rate)
    for _ in range(MAX_RETRACK_ITERATIONS):
        # Line 0's own sync is often merged with the VIS stop bit, so 239 is
        # complete. Anything less is worth another round, which stops by itself
        # as soon as a round adds no lines.
        if track.matched_lines >= LINES - 1 or track.sync_fit is None:
            break
        candidates, centres = _coarse_pulses(series, _centre_function(track))
        pulses = _pulses_from_candidates(phase_track, sample_rate, candidates, centres)
        retracked = _track_pulses(pulses, expected_start_ms, expected_end_ms)
        if retracked is None or retracked.matched_lines <= track.matched_lines:
            break
        track = _attach_offsets(retracked, phase_track, sample_rate)
    return track


def track_line_starts(
    samples: np.ndarray,
    sample_rate: int,
    expected_start_ms: float | None = None,
    tone_offset_hz: float = 0.0,
    phase_track: np.ndarray | None = None,
    expected_end_ms: float | None = None,
) -> SyncTrack | None:
    """Track Robot 36 line sync pulses.

    ``expected_start_ms`` is an optional approximate time for line 0, normally
    taken from a VIS header or from ``--start-ms``. When it is omitted, the
    tracker finds the strongest 150 ms-spaced pulse sequence in the recording.

    ``expected_end_ms`` is an optional approximate time at which the last line
    ends; it fixes the line numbering when there is no start hint and the
    first lines were not detected. ``tone_offset_hz`` shifts the nominal sync
    tone frequency. ``phase_track`` may be supplied to avoid recomputing the
    phase of ``samples``.

    Returns ``None`` when too few sync pulses are found to be trustworthy.

    When the nominal lock is weak (or fails), the tracker scans a grid of
    alternative sync-tone frequencies -- for example when a Doppler-shifted
    capture puts the sync pulse away from the nominal Robot 36 frequency --
    and keeps the strongest track. The returned track carries the measured
    sync frequency of each line (``line_offsets_hz``), so a constant or
    drifting offset can be removed when the pixels are decoded.
    """
    duration_ms = 1000.0 * samples.size / sample_rate
    if duration_ms < SYNC_PULSE_MIN_MS:
        return None

    if phase_track is None:
        phase_track = phase(samples, sample_rate, tone_offset_hz=tone_offset_hz)
    series = centered_frequencies(phase_track, sample_rate, 0.0, duration_ms, 1.0, 2.5)
    nominal = FREQ_SYNC + tone_offset_hz

    def attempt(sync_hz: float) -> SyncTrack | None:
        candidates, centres = _coarse_pulses(series, sync_hz)
        pulses = _pulses_from_candidates(phase_track, sample_rate, candidates, centres)
        track = _track_pulses(pulses, expected_start_ms, expected_end_ms)
        if track is None:
            return None
        # Follow the fitted sync curve *before* comparing candidates: under a
        # Doppler drift the true sync tone only matches part of the image at
        # first, while a false periodic structure (line separators) can match
        # more lines until both have been extended along their own drift.
        return _finish_track(
            track, phase_track, sample_rate, series, expected_start_ms, expected_end_ms
        )

    # Only a near-complete nominal lock is trusted outright. A partial lock can
    # be a false periodic structure (for example the 1500 Hz line separator of
    # even lines landing inside the sync tolerance of a shifted signal), so
    # the grid candidates are evaluated and the strongest track wins.
    best = attempt(nominal)
    if best is None or best.matched_lines < GOOD_TRACK_FRACTION * LINES:
        for sync_hz in _sync_grid_frequencies(series, nominal)[:GRID_MAX_EVALUATED]:
            alt = attempt(sync_hz)
            if alt is not None and (best is None or alt.matched_lines > best.matched_lines):
                best = alt
            if best is not None and best.matched_lines >= GOOD_TRACK_FRACTION * LINES:
                break
    return best
