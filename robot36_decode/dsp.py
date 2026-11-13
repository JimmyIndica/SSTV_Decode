"""Signal-processing helpers for complex IQ baseband FM tone analysis."""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .constants import (
    ANALYSIS_BAND_AUTO_RATIO,
    ANALYSIS_BAND_HIGH_HZ,
    ANALYSIS_BAND_LOW_HZ,
    ANALYSIS_BAND_TAPER_HZ,
    FREQ_SYNC,
)


def analytic_signal(x: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Return the analytic signal using the Hilbert transform."""
    if np.iscomplexobj(x):
        return x
    x = np.asarray(x, dtype=np.float64)  # double precision, as numpy.fft always used
    n = x.shape[0]
    if n == 0:
        return x.astype(np.complex128)
    # Reflected padding greatly reduces Hilbert edge artefacts at the start
    # and end of the signal.
    pad = min(n, max(32, int(0.02 * float(sample_rate))))
    padded = np.pad(x, (pad, pad), mode="reflect")
    nfft = 1 << (padded.size - 1).bit_length()
    # One-sided spectrum: DC and Nyquist kept, positive frequencies doubled,
    # negative ones zero. The real FFT gives the positive half directly.
    half = sfft.rfft(padded, nfft)
    half[1 : nfft // 2] *= 2.0
    spectrum = np.zeros(nfft, dtype=half.dtype)
    spectrum[: half.size] = half
    analytic = sfft.ifft(spectrum, overwrite_x=True)
    return analytic[pad : pad + n]


def _bandpass_gain(
    freqs: np.ndarray, low_hz: float, high_hz: float, taper_hz: float
) -> np.ndarray:
    """Return the FFT-domain bandpass gain for the given bin frequencies.

    The passband is ``[low_hz, high_hz]``; the edges of the transition bands
    are tapered over ``taper_hz`` so the filter does not ring. A taper of zero
    yields a brick-wall mask over the passband.
    """
    taper = float(taper_hz)
    if taper <= 0.0:
        return ((freqs >= low_hz) & (freqs <= high_hz)).astype(np.float64)
    ramp_up = np.clip((freqs - (low_hz - taper)) / taper, 0.0, 1.0)
    ramp_down = np.clip(((high_hz + taper) - freqs) / taper, 0.0, 1.0)
    gain = np.clip(ramp_up * ramp_down, 0.0, 1.0)
    gain[(freqs < low_hz - taper) | (freqs > high_hz + taper)] = 0.0
    return gain


def _out_of_band_fraction(spectrum: np.ndarray, gain: np.ndarray) -> float:
    """Return the fraction of signal energy falling outside the passband."""
    energy = float(np.sum(np.abs(spectrum) ** 2))
    if energy <= 0.0:
        return 0.0
    kept = float(np.sum(np.abs(spectrum) ** 2 * (gain * gain)))
    return 1.0 - kept / energy


def band_limit(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float = ANALYSIS_BAND_LOW_HZ,
    high_hz: float = ANALYSIS_BAND_HIGH_HZ,
    taper_hz: float = ANALYSIS_BAND_TAPER_HZ,
    auto: bool = True,
    tone_offset_hz: float = 0.0,
    band_span_hz: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Band-limit a signal before frequency/phase analysis.

    Applies a linearly tapered bandpass over ``[low_hz, high_hz]`` (a zero taper gives a brick-wall mask). With
    ``auto`` enabled (the default) the filter is only applied when the
    out-of-band energy exceeds ``ANALYSIS_BAND_AUTO_RATIO`` of the total energy,
    so clean or already-filtered recordings keep their original response.

    ``tone_offset_hz`` shifts the passband together with the signal's tone
    frequencies. Use it when the recording carries a Doppler shift so the filter
    window follows the signal instead of sitting at the nominal frequencies.

    ``band_span_hz`` is ``(low_extra, high_extra)`` added to the lower and upper
    band edges (after ``tone_offset_hz``). A drifting Doppler shift moves the
    signal through a range of frequencies during one image, so the window must
    cover the whole sweep rather than one instantaneous offset.

    Real-valued signals are filtered through the real FFT; complex signals go
    through the full complex FFT.
    """
    # scipy.fft keeps single precision for single-precision input; analysis
    # is done in double precision (as numpy.fft always did).
    samples = np.asarray(
        samples, dtype=np.complex128 if np.iscomplexobj(samples) else np.float64
    )
    n = int(samples.shape[0])
    low = max(float(low_hz) + float(tone_offset_hz) + float(band_span_hz[0]), 50.0)
    high = float(high_hz) + float(tone_offset_hz) + float(band_span_hz[1])

    if np.iscomplexobj(samples):
        spectrum = sfft.fft(samples, axis=0)
        # Signed bin frequencies: negative baseband frequencies are negative
        # here, so only the positive-frequency (analytic) passband is kept.
        freqs = np.fft.fftfreq(n, d=1.0 / float(sample_rate))
        gain = _bandpass_gain(freqs, low, high, taper_hz)
        should_filter = auto and _out_of_band_fraction(spectrum, gain) > ANALYSIS_BAND_AUTO_RATIO
        if not should_filter:
            return samples
        return sfft.ifft(spectrum * gain, axis=0)

    spectrum = sfft.rfft(samples, axis=0)
    freqs = np.arange(spectrum.size, dtype=np.float64) * (float(sample_rate) / n)
    gain = _bandpass_gain(freqs, low, high, taper_hz)
    should_filter = auto and _out_of_band_fraction(spectrum, gain) > ANALYSIS_BAND_AUTO_RATIO
    if not should_filter:
        return samples
    return sfft.irfft(spectrum * gain, n=n, axis=0)


def phase(
    iq_samples: np.ndarray,
    sample_rate: int,
    tone_offset_hz: float = 0.0,
    band_span_hz: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return the unwrapped phase of the complex IQ signal.

    The signal is band-limited to the SSTV analysis band first (when its
    out-of-band energy is significant) so that broadband or out-of-band noise
    does not dominate the phase estimate. ``tone_offset_hz`` shifts the analysis
    band along with the signal's tone frequencies, which keeps a Doppler-shifted
    recording inside the filter window; ``band_span_hz`` widens the window to
    cover a drifting offset.
    """
    limited = band_limit(
        iq_samples, sample_rate, tone_offset_hz=tone_offset_hz, band_span_hz=band_span_hz
    )
    analytic = analytic_signal(limited, sample_rate)
    return np.unwrap(np.angle(analytic))


def phase_values(
    phase_track: np.ndarray, sample_rate: int, times_ms: np.ndarray
) -> np.ndarray:
    """Interpolate unwrapped phase at several times, extrapolating at edges.

    Positions outside the track are extrapolated linearly from the first or
    last segment, which falls out of clamping the segment index and letting
    the fractional part leave [0, 1].
    """
    positions = np.asarray(times_ms, dtype=np.float64) * (sample_rate / 1000.0)
    last = phase_track.size - 1
    if last < 1:
        return np.full(positions.shape, float(phase_track[0]) if last == 0 else 0.0)
    index = np.clip(np.floor(positions), 0, last - 1).astype(np.int64)
    frac = positions - index
    left = phase_track[index]
    return left + frac * (phase_track[index + 1] - left)


def phase_at(phase_track: np.ndarray, sample_rate: int, time_ms: float) -> float:
    """Linearly interpolate the unwrapped phase at a time in milliseconds."""
    return float(phase_values(phase_track, sample_rate, np.array([time_ms]))[0])


def frequency_between(
    phase_track: np.ndarray, sample_rate: int, start_ms: float, end_ms: float
) -> float:
    """Average tone frequency over a time interval, in Hz."""
    delta_ms = end_ms - start_ms
    if delta_ms <= 0.0:
        raise ValueError("frequency interval must have positive duration")
    delta_phase = phase_at(phase_track, sample_rate, end_ms) - phase_at(
        phase_track, sample_rate, start_ms
    )
    return delta_phase * 1000.0 / (2.0 * np.pi * delta_ms)


def segment_frequencies(
    phase_track: np.ndarray,
    sample_rate: int,
    start_ms: float,
    pixel_ms: float,
    count: int,
    lead_guard_ms: float = 0.0,
    tail_guard_ms: float = 0.0,
    min_window_ms: float = 0.15,
) -> np.ndarray:
    """Measure the average frequency of each pixel-sized time segment.

    The first and last pixels of a scan line sit next to a tone change (sync
    porch, separator, next sync pulse). The demodulator and band-pass filter
    smear such a step over several tenths of a millisecond, which can be longer
    than a pixel (0.275 ms), so averaging a whole edge pixel mixes in the
    neighbouring tone. With a guard, no window may start before
    ``start + lead_guard_ms`` or end after ``end - tail_guard_ms`` (``end`` is
    the end of the last pixel); pixels that fall inside a guard are measured
    over the nearest clean stretch of at least ``min_window_ms`` instead.
    Nothing is interpolated or filtered: it is the same signal, sampled where
    it is trustworthy. Without guards every pixel is averaged over its own
    time slot.
    """
    starts = start_ms + np.arange(count, dtype=np.float64) * pixel_ms
    ends = starts + pixel_ms
    if lead_guard_ms > 0.0:
        starts = np.maximum(starts, start_ms + lead_guard_ms)
        ends = np.maximum(ends, starts + min_window_ms)
    if tail_guard_ms > 0.0:
        limit = start_ms + count * pixel_ms - tail_guard_ms
        ends = np.minimum(ends, limit)
        starts = np.minimum(starts, ends - min_window_ms)
    p0 = phase_values(phase_track, sample_rate, starts)
    p1 = phase_values(phase_track, sample_rate, ends)
    return (p1 - p0) * 1000.0 / (2.0 * np.pi * (ends - starts))


def centered_frequencies(
    phase_track: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
    step_ms: float,
    width_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure frequencies in windows centred on evenly spaced times."""
    centres = np.arange(start_ms, end_ms + step_ms * 0.5, step_ms)
    before = centres - width_ms / 2.0
    after = centres + width_ms / 2.0
    p0 = phase_values(phase_track, sample_rate, before)
    p1 = phase_values(phase_track, sample_rate, after)
    return centres, (p1 - p0) * 1000.0 / (2.0 * np.pi * width_ms)


def first_true_run(mask: np.ndarray, minimum: int) -> int | None:
    """Return the first index of a run of True values of at least `minimum`."""
    start: int | None = None
    count = 0
    for index, value in enumerate(mask):
        if value:
            if start is None:
                start = index
            count += 1
            if count >= minimum:
                return start
        else:
            start = None
            count = 0
    return None


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each run of True values."""
    flags = np.asarray(mask, dtype=bool)
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def downward_crossing(
    times: np.ndarray, frequencies: np.ndarray, threshold: float
) -> float | None:
    """Find the first time the frequency falls through a threshold."""
    above = frequencies >= threshold
    for index in range(frequencies.size - 1):
        if above[index] and not above[index + 1]:
            span = frequencies[index] - frequencies[index + 1]
            frac = (frequencies[index] - threshold) / span if span else 0.0
            return float(times[index] + frac * (times[index + 1] - times[index]))
    return None


def upward_crossing(
    times: np.ndarray, frequencies: np.ndarray, threshold: float
) -> float | None:
    """Find the first time the frequency rises through a threshold."""
    below = frequencies < threshold
    for index in range(frequencies.size - 1):
        if below[index] and not below[index + 1]:
            span = frequencies[index + 1] - frequencies[index]
            frac = (threshold - frequencies[index]) / span if span else 0.0
            return float(times[index] + frac * (times[index + 1] - times[index]))
    return None


def energy_envelope(
    samples: np.ndarray, sample_rate: int, window_ms: float = 10.0
) -> np.ndarray:
    """Return mean signal magnitude per time window.

    Index ``i`` holds the mean complex magnitude of the samples falling in
    ``[i * window_ms, (i + 1) * window_ms)`` milliseconds. Used to locate the
    signal inside a recording that contains leading/trailing noise.
    """
    n = int(samples.shape[0])
    window_ms = max(0.5, float(window_ms))
    frame = max(1, int(round(window_ms * sample_rate / 1000.0)))
    usable = (n // frame) * frame
    if usable == 0 or frame == 0:
        return np.zeros(0, dtype=np.float64)
    magnitudes = np.abs(samples[:usable].reshape(-1, frame))
    return magnitudes.mean(axis=1)


def find_signal_region(
    envelope: np.ndarray, window_ms: float, threshold_ratio: float = 0.2
) -> tuple[float, float] | None:
    """Return ``(start_ms, end_ms)`` of the loudest contiguous region.

    The region is the longest run of envelope bins whose magnitude exceeds
    ``threshold_ratio`` times the peak. Returns ``None`` when the envelope is
    empty or the surviving region is implausibly short.
    """
    if envelope.size == 0:
        return None
    peak = float(envelope.max())
    if peak <= 0.0:
        return None
    floor = float(np.median(envelope))
    threshold = floor + (peak - floor) * float(threshold_ratio)
    if threshold <= floor:
        threshold = peak * float(threshold_ratio)
    mask = envelope > threshold
    best: tuple[int, int] | None = None
    start: int | None = None
    stop = 0
    for index, value in enumerate(mask):
        if value:
            if start is None:
                start = index
            stop = index + 1
        elif start is not None:
            if best is None or (stop - start) > (best[1] - best[0]):
                best = (start, stop)
            start = None
    if start is not None and (best is None or (stop - start) > (best[1] - best[0])):
        best = (start, stop)
    if best is None:
        return None
    start_ms = float(best[0]) * float(window_ms)
    end_ms = float(best[1]) * float(window_ms)
    if end_ms - start_ms < 10.0:
        return None
    return (start_ms, end_ms)


def dominant_frequency(
    samples: np.ndarray,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
    low_hz: float,
    high_hz: float,
    taper_hz: float = 0.0,
    search_center_hz: float | None = None,
    search_width_hz: float | None = None,
) -> float:
    """Return the strongest frequency between ``start_ms`` and ``end_ms``.

    Only bins inside ``[low_hz, high_hz]`` (with ``taper_hz`` transition) are
    considered; the returned value is the peak bin magnitude's frequency, in Hz.

    ``search_center_hz``/``search_width_hz`` optionally override the search
    window, which lets callers anchor the estimate on a particular tone (for
    example the sync pulse) instead of the whole analysis band.
    """
    low = float(low_hz)
    high = float(high_hz)
    if search_center_hz is not None and search_width_hz is not None:
        half = float(search_width_hz) / 2.0
        low = float(search_center_hz) - half
        high = float(search_center_hz) + half
    start = int(round(start_ms * sample_rate / 1000.0))
    end = int(round(end_ms * sample_rate / 1000.0))
    start = max(0, start)
    end = min(int(samples.shape[0]), end)
    if end - start < 2:
        return 0.0
    segment = np.asarray(samples[start:end], dtype=np.complex128)
    spectrum = sfft.fft(segment, axis=0)
    n = int(segment.shape[0])
    freqs = np.arange(n, dtype=np.float64) * (float(sample_rate) / n)
    gain = _bandpass_gain(freqs, low, high, float(taper_hz))
    magnitude = np.abs(spectrum) * gain
    index = int(np.argmax(magnitude))
    return float(freqs[index])


def locate_signal(
    samples: np.ndarray,
    sample_rate: int,
    window_ms: float = 10.0,
    low_hz: float = ANALYSIS_BAND_LOW_HZ,
    high_hz: float = ANALYSIS_BAND_HIGH_HZ,
    taper_hz: float = ANALYSIS_BAND_TAPER_HZ,
    sync_hz: float = FREQ_SYNC,
    sync_width_hz: float = 1200.0,
    search_high_hz: float = 12000.0,
) -> tuple[float, float, float]:
    """Locate a signal inside a recording using its energy.

    Noise before/after the signal is tolerated because the region is found from
    the magnitude envelope rather than from a fixed time offset.

    The envelope and the tone-centre estimate are computed on a band that is
    wide enough to contain the (possibly Doppler-shifted) signal, so recording
    noise outside the nominal SSTV band does not destroy the energy location.

    Returns ``(start_ms, end_ms, center_hz)`` where ``center_hz`` is the
    strongest frequency of the located region. Callers can use the returned
    center to place the analysis filter on the actual (possibly Doppler-shifted)
    signal instead of the nominal tone frequencies.
    """
    band_low = float(low_hz)
    band_high = max(float(high_hz), float(search_high_hz))
    limited = band_limit(samples, sample_rate, band_low, band_high, taper_hz, auto=True)
    envelope = energy_envelope(limited, sample_rate, window_ms)
    region = find_signal_region(envelope, window_ms)
    if region is None:
        return (0.0, 0.0, 0.0)
    start_ms, end_ms = region
    center_hz = dominant_frequency(
        limited,
        sample_rate,
        start_ms,
        end_ms,
        band_low,
        band_high,
        taper_hz,
    )
    return (start_ms, end_ms, float(center_hz))
