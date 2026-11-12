"""FM demodulation of raw SDR baseband captures (SDR#/SDR++ style recordings).

A baseband recording of a satellite SSTV downlink is a complex I/Q stream at
the SDR sample rate (e.g. 78.125 kHz) holding an *FM carrier* whose audio is
the SSTV subcarrier. Compared with the audio the decoder needs, the capture
has three problems that this module solves:

* long stretches of dead air before and after the ~38 s signal,
* a Doppler-shifted carrier that is not at 0 Hz and drifts during the pass,
* a low signal level with bursts, clicks and AGC steps.

Pipeline (all FFT/vectorised, a few seconds for a two-minute capture):

1. ``locate_fm_signal``: time/frequency power map -> sliding-band excess
   power finds *when* the FM signal is present and *where* its carrier is,
   immune to wideband bursts and fixed spurs.
2. ``demodulate_fm``: mix the tracked carrier to 0 Hz, channel-filter,
   FM-discriminate, then measure the residual carrier error from the
   discriminator output itself and repeat. The final carrier curve is the
   Doppler shift over the signal.
3. Resample the audio to 48 kHz for the SSTV decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .resample import resample

AUDIO_RATE = 48000

# Time/frequency analysis.
TF_NFFT = 1024
TF_AVG = 8  # FFT blocks averaged per time frame (~0.1 s at 78 kHz)
BAND_WIDTH_HZ = 12000.0  # sliding window used to find the FM signal
MIN_SCORE_RATIO = 1.2  # smoothed band score must exceed this times the noise level
SCORE_SMOOTH_S = 1.0
GAP_S = 4.0  # dropouts (interference, fades) shorter than this do not split the signal
DEFAULT_EXPECTED_S = 38.0  # a Robot 36 transmission incl. VIS header and tail
MAX_F0_JUMP_HZ = 3000.0
POLY_DEGREE = 3

# Demodulation.
CHANNEL_HALF_BW_HZ = 7000.0  # wide enough for the full deviation of a strong signal
MIN_HALF_BW_HZ = 5000.0  # narrowest automatic choice, for weak (noise-limited) signals
CHANNEL_TAPS = 301
# Feedback (threshold-extension) passes: re-mix with the estimated audio phase,
# then filter much narrower so less noise reaches the discriminator. The audio
# estimate is low-passed just above the highest SSTV tone (2300 Hz): anything
# higher is pure discriminator noise. (Tuned against a reference decode; 3200 Hz
# / 3000 Hz / 2 passes left roughly twice the noise.)
# The passes are annealed: early passes use a wide filter so the loop stays
# locked while the audio estimate is still poor; later passes narrow to the
# final 2400 Hz. (half bandwidth Hz, audio low-pass Hz) per pass.
FEEDBACK_SCHEDULE: tuple[tuple[float, float], ...] = (
    (4200.0, 3000.0),
    (3600.0, 2800.0),
    (3000.0, 2600.0),
    (2600.0, 2400.0),
    (2400.0, 2400.0),
    (2400.0, 2400.0),
)
# The demodulated audio is band-passed to the SSTV tone range before it leaves
# this module. VIS bits are 1100/1300 Hz, sync 1200 Hz, picture 1500-2300 Hz, so
# 1000-2500 Hz keeps all of them while dropping ~1.2 kHz of noise that the
# decoder's wider 300-3000 Hz analysis band would otherwise pass to every pixel.
AUDIO_BAND_HZ = (1000.0, 2500.0)
AUDIO_BAND_TAPS = 201
DC_WINDOW_S = 0.025  # moving-average high-pass applied to the audio
CLICK_FACTOR = 1.3  # |instantaneous frequency| above this x channel half-bandwidth is a click
CLICK_GUARD = 2  # samples blanked on each side of a click
MARGIN_S = 1.5
REFINE_PASSES = 2
RESIDUAL_BLOCK_S = 0.5


@dataclass
class FmLocation:
    """Where the FM signal is in the capture."""

    start_s: float
    end_s: float
    track_t_s: np.ndarray  # frame centre times inside the signal
    track_hz: np.ndarray  # carrier estimate per frame (relative to the tuned centre)
    snr_ratio: float  # median in-band/out-of-band power ratio inside the signal

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class FmAudio:
    """Demodulated audio plus the Doppler curve that was removed."""

    audio: np.ndarray  # float64, AUDIO_RATE
    sample_rate: int
    t0_s: float  # capture time of audio[0]
    doppler_t_s: np.ndarray  # evaluation times (capture seconds)
    doppler_hz: np.ndarray  # carrier offset from the tuned centre frequency
    signal_start_s: float
    signal_end_s: float

    @property
    def doppler_rate_hz_per_s(self) -> float:
        if self.doppler_t_s.size < 2:
            return 0.0
        return float(np.polyfit(self.doppler_t_s, self.doppler_hz, 1)[0])


# ---------------------------------------------------------------------------
# Locating the signal
# ---------------------------------------------------------------------------


def _time_frequency_power(x: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Return ``(P[t, f], frame_seconds)`` with ``f`` in fftshift order."""
    frames = (x.size // TF_NFFT) // TF_AVG * TF_AVG
    if frames < TF_AVG:
        raise ValueError("recording is too short to analyse")
    window = np.hanning(TF_NFFT).astype(np.float32)
    power = np.empty((frames // TF_AVG, TF_NFFT), dtype=np.float32)
    # Process in chunks to keep the working set small for long captures.
    chunk = TF_AVG * 256
    for first in range(0, frames, chunk):
        last = min(frames, first + chunk)
        blocks = x[first * TF_NFFT : last * TF_NFFT].reshape(-1, TF_NFFT) * window
        spec = np.fft.fftshift(np.abs(np.fft.fft(blocks, axis=1)) ** 2, axes=1)
        power[first // TF_AVG : last // TF_AVG] = spec.reshape(-1, TF_AVG, TF_NFFT).mean(axis=1)
    return power, TF_AVG * TF_NFFT / float(fs)


def _robust_poly(
    t: np.ndarray, y: np.ndarray, degree: int, limit_hz: float, rounds: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial fit with outlier rejection; returns (coefficients, inliers)."""
    inliers = np.ones(t.size, dtype=bool)
    degree = min(degree, max(0, t.size - 1))
    centre = float(np.mean(t))
    coef = np.polyfit(t - centre, y, degree)
    for _ in range(rounds):
        residual = y - np.polyval(coef, t - centre)
        spread = 1.4826 * np.median(np.abs(residual[inliers])) if inliers.any() else limit_hz
        inliers = np.abs(residual) <= max(limit_hz * 0.25, 3.0 * spread)
        if np.count_nonzero(inliers) <= degree + 1:
            break
        coef = np.polyfit(t[inliers] - centre, y[inliers], degree)
    # Re-express around zero so callers can evaluate with plain ``t``.
    poly = np.poly1d(coef)
    shifted = np.polyfit(t, poly(t - centre), degree)
    return shifted, inliers


def _refine_centroids(
    q: np.ndarray,
    frames: np.ndarray,
    t_frames: np.ndarray,
    coarse: np.ndarray,
    frame_s: float,
    fs: float,
) -> np.ndarray:
    """Carrier estimate per frame: an excess-power centroid around the coarse
    Doppler curve, averaged over ~1 s so bin noise does not dominate weak
    signals."""
    half = int(round(7000.0 / (fs / TF_NFFT)))
    freqs = (np.arange(TF_NFFT) - TF_NFFT / 2.0) * fs / TF_NFFT
    smooth = max(1, int(round(1.0 / frame_s)))
    centroid = np.empty(frames.size)
    for n, frame in enumerate(frames):
        lo, hi = max(0, frame - smooth // 2), min(q.shape[0], frame + smooth // 2 + 1)
        mean_q = q[lo:hi].mean(axis=0)
        centre = int(np.argmin(np.abs(freqs - np.polyval(coarse, t_frames[frame]))))
        a, b = max(0, centre - half), min(TF_NFFT, centre + half + 1)
        weights = np.clip(mean_q[a:b] - 1.15, 0.0, None)
        centroid[n] = (
            float(np.sum(weights * freqs[a:b]) / np.sum(weights))
            if weights.sum() > 0
            else float(np.polyval(coarse, t_frames[frame]))
        )
    return centroid


def _window_location(
    q: np.ndarray,
    score: np.ndarray,
    f0: np.ndarray,
    frame_s: float,
    fs: float,
    duration_s: float,
    threshold: float,
    window: tuple[float, float],
) -> FmLocation | None:
    """Carrier track for a signal the caller says occupies ``window``."""
    start_s = max(0.0, window[0])
    end_s = min(duration_s, window[1])
    first = int(start_s / frame_s)
    last = min(score.size, int(np.ceil(end_s / frame_s)))
    if last - first < 10:
        return None
    frames = np.arange(first, last)
    t_frames = (np.arange(score.size) + 0.5) * frame_s
    strong = frames[score[first:last] > threshold]
    if strong.size < 5:
        return None
    # A line through the clear frames extrapolates safely over fades; a
    # curve is only trusted when those frames span most of the window.
    covered = (t_frames[strong[-1]] - t_frames[strong[0]]) / max(end_s - start_s, 1e-9)
    degree = 2 if covered >= 0.5 else 1
    coarse, ok = _robust_poly(t_frames[strong], f0[strong], degree, MAX_F0_JUMP_HZ)
    near = strong[np.abs(f0[strong] - np.polyval(coarse, t_frames[strong])) < MAX_F0_JUMP_HZ]
    if near.size < 5:
        return None
    centroid = _refine_centroids(q, near, t_frames, coarse, frame_s, fs)
    return FmLocation(
        start_s=start_s,
        end_s=end_s,
        track_t_s=t_frames[near],
        track_hz=centroid,
        snr_ratio=float(np.median(score[near])),
    )


def locate_fm_signal(
    x: np.ndarray,
    fs: float,
    expected_seconds: float | None = DEFAULT_EXPECTED_S,
    window: tuple[float, float] | None = None,
) -> FmLocation | None:
    """Find the FM signal in a baseband capture.

    The score of a frame is the strongest excess of power in any
    ``BAND_WIDTH_HZ`` window relative to that frame's own noise floor, after
    removing fixed spurs. Wideband bursts and clicks raise the whole frame and
    so do not score; a modulated carrier does.

    With ``window=(start_s, end_s)`` the search is skipped: the signal is taken
    to occupy exactly that stretch of the capture (clipped to the file) and
    only the carrier track is estimated, from the frames inside it that show a
    clear carrier. Use this for captures so poor (deep fades, bursts) that
    the automatic search cannot find a whole transmission.
    """
    power, frame_s = _time_frequency_power(x, fs)
    # Remove fixed spurs / the DC spike: normalise each bin by its quiet level.
    floor = np.percentile(power, 25, axis=0)
    floor = np.maximum(floor, np.median(floor) * 1e-3)
    q = power / floor
    q /= np.median(q, axis=1, keepdims=True)

    bins = max(8, int(round(BAND_WIDTH_HZ / (fs / TF_NFFT))))
    cumulative = np.cumsum(np.pad(q, ((0, 0), (1, 0))), axis=1, dtype=np.float64)
    ratio = (cumulative[:, bins:] - cumulative[:, :-bins]) / bins
    centres_hz = (np.arange(ratio.shape[1]) + bins / 2.0 - TF_NFFT / 2.0) * fs / TF_NFFT
    best = ratio.argmax(axis=1)
    score = ratio.max(axis=1)
    f0 = centres_hz[best]

    # Smooth over ~1 s: a weak signal dips below the noise-relative threshold
    # frame by frame during fades and interference, but not on average.
    # (np.convolve "same" returns the longer input, so the kernel must not be
    # longer than the data: very short captures have only a few frames.)
    k = min(max(1, int(round(SCORE_SMOOTH_S / frame_s))), score.size)
    score = np.convolve(score, np.ones(k) / k, mode="same")
    noise = float(np.percentile(score, 20))
    threshold = noise * MIN_SCORE_RATIO
    if window is not None:
        return _window_location(q, score, f0, frame_s, fs, x.size / fs, threshold, window)
    active = np.flatnonzero(score > threshold)
    if active.size < 5:
        return None

    # Split into segments on time gaps, then keep the strongest one.
    gap_frames = max(1, int(round(GAP_S / frame_s)))
    splits = np.flatnonzero(np.diff(active) > gap_frames) + 1
    segments = np.split(active, splits)
    mass = [float(np.sum(score[s] - noise)) for s in segments]
    seg = segments[int(np.argmax(mass))]
    if seg.size < 5:
        return None

    # Keep frames whose carrier estimate agrees with a smooth Doppler curve.
    t_frames = (np.arange(score.size) + 0.5) * frame_s
    in_seg = np.arange(seg[0], seg[-1] + 1)
    coarse, ok = _robust_poly(t_frames[seg], f0[seg], POLY_DEGREE, MAX_F0_JUMP_HZ)
    consistent = np.abs(f0[in_seg] - np.polyval(coarse, t_frames[in_seg])) < MAX_F0_JUMP_HZ
    run = in_seg[consistent & (score[in_seg] > threshold)]
    if run.size < 5:
        return None
    splits = np.flatnonzero(np.diff(run) > gap_frames) + 1
    run = max(np.split(run, splits), key=lambda r: float(np.sum(score[r] - noise)))

    centroid = _refine_centroids(q, run, t_frames, coarse, frame_s, fs)

    start_s = float(run[0]) * frame_s
    end_s = float(run[-1] + 1) * frame_s
    if expected_seconds is not None and end_s - start_s > 1.25 * expected_seconds:
        # Longer than a transmission: keep the strongest stretch of about the
        # expected length (plus slack) rather than a long noisy tail.
        length = int((expected_seconds + 4.0) / frame_s)
        weights = np.convolve(score[run] - noise, np.ones(length), mode="valid")
        first = int(np.argmax(weights))
        keep = slice(first, first + length)
        run, centroid = run[keep], centroid[keep]
        start_s, end_s = float(run[0]) * frame_s, float(run[-1] + 1) * frame_s

    inband = float(np.median(score[run]))
    return FmLocation(
        start_s=start_s,
        end_s=end_s,
        track_t_s=t_frames[run],
        track_hz=centroid,
        snr_ratio=inband,
    )


# ---------------------------------------------------------------------------
# Demodulation
# ---------------------------------------------------------------------------


def _discriminate(y: np.ndarray, fs: float) -> np.ndarray:
    """FM discriminator: instantaneous frequency (Hz) from phase differences."""
    out = np.empty(y.size, dtype=np.float64)
    out[1:] = np.angle(y[1:] * np.conj(y[:-1]))
    out[0] = out[1] if y.size > 1 else 0.0
    return out * (fs / (2.0 * np.pi))


def _mix_filter_discriminate(
    x: np.ndarray, fs: float, carrier_hz: np.ndarray, half_bw_hz: float
) -> np.ndarray:
    phase = 2.0 * np.pi * np.cumsum(carrier_hz, dtype=np.float64) / fs
    mixed = x * np.exp(-1j * phase).astype(np.complex64)
    taps = sps.firwin(CHANNEL_TAPS, half_bw_hz, fs=fs, window=("kaiser", 6.0)).astype(np.float32)
    filtered = sps.oaconvolve(mixed, taps, mode="same")
    return _discriminate(filtered, fs)


def _declick(disc: np.ndarray, limit_hz: float) -> np.ndarray:
    """Replace FM click noise by interpolation.

    When noise momentarily swamps the carrier the discriminator emits a spike
    up to +-fs/2. Since the channel filter only passes +-``half_bw``, an
    instantaneous frequency well beyond it cannot be signal. Those samples (and
    a small guard either side) are blanked and filled in linearly.
    """
    bad = np.abs(disc) > limit_hz
    if not bad.any():
        return disc
    kernel = np.ones(2 * CLICK_GUARD + 1)
    if disc.size < kernel.size:
        return disc
    bad = np.convolve(bad.astype(np.float32), kernel, mode="same") > 0
    good = np.flatnonzero(~bad)
    if good.size < 2:
        return np.zeros_like(disc)
    out = disc.copy()
    out[bad] = np.interp(np.flatnonzero(bad), good, disc[good])
    return out


def _remove_dc(audio: np.ndarray, fs: float) -> np.ndarray:
    """Subtract a short moving average (a high-pass) to drop any residual carrier error."""
    n = max(3, int(round(DC_WINDOW_S * fs)) | 1)
    cumulative = np.concatenate(([0.0], np.cumsum(audio)))
    mean = (cumulative[n:] - cumulative[:-n]) / n
    pad = n // 2
    mean = np.pad(mean, (pad, audio.size - mean.size - pad), mode="edge")
    return audio - mean


def auto_half_bandwidth(snr_ratio: float) -> float:
    """Channel half-bandwidth from the signal strength.

    A strong signal needs the full FM deviation (an overly narrow filter clips
    it); a weak one is noise-limited, so a narrower filter wins. ``snr_ratio``
    is the median in-band/noise power ratio found by ``locate_fm_signal``.
    """
    fraction = (float(snr_ratio) - 2.0) / 6.0
    return float(MIN_HALF_BW_HZ + np.clip(fraction, 0.0, 1.0) * (CHANNEL_HALF_BW_HZ - MIN_HALF_BW_HZ))


def demodulate_fm(
    x: np.ndarray,
    fs: float,
    location: FmLocation,
    half_bw_hz: float | None = None,
    margin_s: float = MARGIN_S,
    audio_rate: int = AUDIO_RATE,
    feedback_schedule: tuple[tuple[float, float], ...] | None = FEEDBACK_SCHEDULE,
    audio_band_hz: tuple[float, float] | None = AUDIO_BAND_HZ,
) -> FmAudio:
    """Demodulate the located FM signal to audio at ``audio_rate``.

    ``half_bw_hz=None`` picks the channel width from the signal strength.
    ``feedback_schedule`` is a sequence of (half bandwidth, audio low-pass) Hz per
    feedback pass (``None``/empty disables). ``audio_band_hz`` band-passes the output audio (``None`` disables it).
    """
    if half_bw_hz is None:
        half_bw_hz = auto_half_bandwidth(location.snr_ratio)
    first = max(0, int(round((location.start_s - margin_s) * fs)))
    last = min(x.size, int(round((location.end_s + margin_s) * fs)))
    seg = np.ascontiguousarray(x[first:last], dtype=np.complex64)
    t_s = first / fs + np.arange(seg.size) / fs

    # Smooth Doppler curve from the spectral centroid track.
    coef, _ = _robust_poly(location.track_t_s, location.track_hz, POLY_DEGREE, 2000.0)
    carrier = np.polyval(coef, np.clip(t_s, location.start_s, location.end_s))

    # Residual carrier error is the mean of the discriminator output; measure
    # it per block (median: robust to clicks), fit it, fold it into the
    # carrier, and repeat.
    block = max(1, int(round(RESIDUAL_BLOCK_S * fs)))
    in_signal = (t_s >= location.start_s) & (t_s <= location.end_s)
    for _ in range(REFINE_PASSES):
        disc = _mix_filter_discriminate(seg, fs, carrier, half_bw_hz)
        usable = np.flatnonzero(in_signal)
        n_blocks = usable.size // block
        if n_blocks < 4:
            break
        idx = usable[: n_blocks * block].reshape(n_blocks, block)
        mid_t = t_s[idx].mean(axis=1)
        resid = np.median(disc[idx], axis=1)
        rcoef, _ = _robust_poly(mid_t, resid, POLY_DEGREE, 2000.0)
        carrier = carrier + np.polyval(rcoef, np.clip(t_s, location.start_s, location.end_s))

    disc = _declick(_mix_filter_discriminate(seg, fs, carrier, half_bw_hz), CLICK_FACTOR * half_bw_hz)

    if feedback_schedule:
        # FM feedback: the audio estimate is turned back into a phase and
        # removed from the signal, leaving a nearly unmodulated carrier that
        # can be filtered far more narrowly than the full FM channel.
        taps: dict[tuple[str, float], np.ndarray] = {}

        def _lp(hz: float) -> np.ndarray:
            if ("lp", hz) not in taps:
                taps[("lp", hz)] = sps.firwin(201, hz, fs=fs).astype(np.float32)
            return taps[("lp", hz)]

        def _narrow(hz: float) -> np.ndarray:
            if ("bw", hz) not in taps:
                taps[("bw", hz)] = sps.firwin(
                    CHANNEL_TAPS, hz, fs=fs, window=("kaiser", 6.0)
                ).astype(np.float32)
            return taps[("bw", hz)]

        estimate = disc - np.median(disc)
        for bw_hz, lp_hz in feedback_schedule:
            smooth = sps.oaconvolve(estimate, _lp(lp_hz), mode="same")
            phase = 2.0 * np.pi * np.cumsum(carrier + smooth, dtype=np.float64) / fs
            residual = sps.oaconvolve(
                seg * np.exp(-1j * phase).astype(np.complex64), _narrow(bw_hz), mode="same"
            )
            correction = _declick(_discriminate(residual, fs), CLICK_FACTOR * bw_hz)
            estimate = smooth + correction
        disc = estimate
    audio = _remove_dc(disc, fs)
    if audio_band_hz is not None:
        band = sps.firwin(
            AUDIO_BAND_TAPS, list(audio_band_hz), pass_zero=False, fs=fs
        ).astype(np.float32)
        audio = sps.oaconvolve(audio, band, mode="same")
    if int(round(fs)) != audio_rate:
        audio = resample(audio, int(round(fs)), audio_rate)
    scale = np.percentile(np.abs(audio), 99.0)
    if scale > 0:
        audio = audio / (1.5 * scale)

    doppler_t = np.linspace(location.start_s, location.end_s, 64)
    final, _ = _robust_poly(t_s[in_signal][:: max(1, block // 8)],
                            carrier[in_signal][:: max(1, block // 8)], POLY_DEGREE, 2000.0)
    return FmAudio(
        audio=np.ascontiguousarray(audio, dtype=np.float64),
        sample_rate=audio_rate,
        t0_s=first / fs,
        doppler_t_s=doppler_t,
        doppler_hz=np.polyval(final, doppler_t),
        signal_start_s=location.start_s,
        signal_end_s=location.end_s,
    )
