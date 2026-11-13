"""SSTV VIS header detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    FREQ_SYNC,
    FREQ_VIS_BIT_0,
    FREQ_VIS_BIT_1,
    FREQ_VIS_LEADER,
    VIS_BIT_MS,
    VIS_BREAK_MS,
    VIS_HEADER_MS,
    VIS_LEADER_MS,
)
from .filters import medfilt
from .dsp import (
    centered_frequencies,
    downward_crossing,
    first_true_run,
    frequency_between,
    phase,
)


# How far into the recording to look for the VIS header. Real recordings have
# dead air or a calibration preamble before it, so this is generous.
VIS_SEARCH_MS = 15000.0


@dataclass
class VisInfo:
    code: int | None
    image_start_ms: float
    start_bit_ms: float
    parity_ok: bool | None


def _detect_vis_phase(
    samples: np.ndarray,
    sample_rate: int,
    tone_offset_hz: float = 0.0,
    phase_track: np.ndarray | None = None,
) -> VisInfo | None:
    """Detect and decode an SSTV VIS header from the instantaneous frequency.

    ``tone_offset_hz`` shifts the VIS tone frequencies to follow a Doppler-shifted
    signal. Returns None when no plausible VIS header is present. The returned
    image start is just after the VIS stop bit. ``phase_track`` may be the
    precomputed phase of ``samples`` (only its first ``VIS_SEARCH_MS`` are used).
    """
    search_ms = min(VIS_SEARCH_MS, 1000.0 * samples.size / sample_rate)
    if search_ms < VIS_HEADER_MS:
        return None

    count = int(search_ms * sample_rate / 1000.0)
    if phase_track is None:
        phase_track = phase(samples[:count], sample_rate, tone_offset_hz=tone_offset_hz)
    else:
        phase_track = phase_track[:count]

    leader_hz = FREQ_VIS_LEADER + tone_offset_hz
    sync_hz = FREQ_SYNC + tone_offset_hz
    bit_mid_hz = (FREQ_VIS_BIT_0 + FREQ_VIS_BIT_1) / 2.0 + tone_offset_hz

    # Coarse pass: find a sustained ~1900 Hz leader.
    centres, coarse = centered_frequencies(
        phase_track, sample_rate, 5.0, search_ms - 5.0, 5.0, 5.0
    )
    # A median filter bridges the isolated glitches that click noise punches
    # into a real leader tone without blurring the leader/start-bit edge.
    leader_mask = np.abs(medfilt(coarse, 9) - leader_hz) < 150.0
    run_start = first_true_run(leader_mask, minimum=int(200.0 / 5.0))
    if run_start is None:
        return None

    candidate_ms = float(centres[run_start])

    # Refine the transition from the 1900 Hz leader into the 1200 Hz VIS start
    # bit. It is nominally 610 ms after the leader starts.
    expected_start = candidate_ms + 2.0 * VIS_LEADER_MS + VIS_BREAK_MS
    fine_start = max(0.0, expected_start - 25.0)
    fine_end = min(search_ms - 1.0, expected_start + 25.0)
    fine_times, fine_freqs = centered_frequencies(
        phase_track, sample_rate, fine_start, fine_end, 0.25, 4.0
    )
    crossing = downward_crossing(fine_times, fine_freqs, (leader_hz + sync_hz) / 2.0)

    if crossing is not None:
        start_bit_ms = crossing
    else:
        # Coarse fallback: the centre of the first sustained 1200 Hz run.
        sync_mask = np.abs(fine_freqs - sync_hz) < 100.0
        sync_index = first_true_run(sync_mask, minimum=int(15.0 / 0.25))
        start_bit_ms = (
            float(fine_times[sync_index])
            if sync_index is not None
            else expected_start
        )

    def bit_frequency(bit_index: int) -> float:
        begin = start_bit_ms + bit_index * VIS_BIT_MS + 7.0
        return frequency_between(phase_track, sample_rate, begin, begin + 16.0)

    # Logic 1 is the lower tone (1100 Hz), logic 0 the higher (1300 Hz).
    code = 0
    for bit_index in range(7):
        if bit_frequency(bit_index + 1) < bit_mid_hz:
            code |= 1 << bit_index

    parity_frequency = bit_frequency(8)
    parity_bit = 1 if parity_frequency < bit_mid_hz else 0
    expected_parity = bin(code).count("1") & 1
    parity_ok = parity_bit == expected_parity

    image_start_ms = start_bit_ms + 10.0 * VIS_BIT_MS
    if image_start_ms >= search_ms:
        return None

    return VisInfo(
        code=code,
        image_start_ms=image_start_ms,
        start_bit_ms=start_bit_ms,
        parity_ok=parity_ok,
    )


# ---------------------------------------------------------------------------
# Spectral (matched-template) detector for weak signals
# ---------------------------------------------------------------------------

ROBOT36_BITS = 0x08
SPECTRAL_FRAME_MS = 5.0
SPECTRAL_WINDOW_MS = 20.0
# Real headers score 3.6-4.0x the best position elsewhere (z-score 19-33); look-alikes
# in picture data or noise reach at most ~1.2x (z < 9).
SPECTRAL_MIN_PEAK_RATIO = 2.0  # best template score over the best one elsewhere
SPECTRAL_MAX_BIT_ERRORS = 1
SPECTRAL_MIN_Z = 10.0  # robust z-score of the best score over all positions
SPECTRAL_EXCLUDE_MS = 350.0


def _tone_energies(
    samples: np.ndarray, sample_rate: int, tones_hz: list[float], frame: int, window: int
) -> np.ndarray:
    """Narrowband energy at each tone per frame: shape (len(tones), frames).

    Each tone is a heterodyne plus boxcar integration over ``window`` samples,
    i.e. a sliding single-bin DFT, so it integrates the tone over the whole
    window instead of deciding per sample. Frame ``i`` covers samples
    ``[i * frame, i * frame + window)``.

    When the window is a whole number of frames, each frame's heterodyne sum
    is one matrix product for all tones at once, and the windows are sums of
    consecutive frames (a few thousand values instead of one per sample).
    """
    n = samples.size
    frames = (n - window) // frame
    if frames <= 0:
        return np.empty((len(tones_hz), 0))
    omega = 2.0 * np.pi * np.asarray(tones_hz, dtype=np.float64) / sample_rate
    if window % frame == 0:
        per = window // frame
        blocks = frames + per - 1
        x = samples[: blocks * frame].reshape(blocks, frame)
        local = np.exp(-1j * np.outer(np.arange(frame), omega))  # (frame, tones)
        sums = x @ local  # heterodyne sum of each block, block-relative phase
        sums *= np.exp(-1j * np.outer(np.arange(blocks) * frame, omega))
        csum = np.concatenate((np.zeros((1, omega.size), dtype=sums.dtype), np.cumsum(sums, axis=0)))
        return (np.abs(csum[per : per + frames] - csum[:frames]) ** 2).T
    t = np.arange(n, dtype=np.float64)
    starts = np.arange(frames) * frame
    out = np.empty((len(tones_hz), frames))
    for k, w in enumerate(omega):
        z = samples * np.exp(-1j * w * t)
        csum = np.concatenate(([0.0], np.cumsum(z)))
        out[k] = np.abs(csum[starts + window] - csum[starts]) ** 2
    return out


SPECTRAL_OFFSETS_HZ = (-225.0, -150.0, -75.0, 0.0, 75.0, 150.0, 225.0)


def _detect_vis_spectral(
    samples: np.ndarray, sample_rate: int, tone_offset_hz: float = 0.0
) -> VisInfo | None:
    """Find a Robot 36 VIS header by matching its whole timeline against tone energy.

    Header: 300 ms 1900 Hz, 10 ms 1200, 300 ms 1900, a 30 ms 1200 Hz start bit,
    the eight bits of ``0001000`` + parity (LSB first: 1300 Hz x3, 1100, 1300 Hz
    x3, 1100), and a 30 ms 1200 Hz stop bit. For each candidate start-bit time
    the expected tone's energy is summed over every part of that ~0.9 s
    timeline, which is far more sensitive than deciding the tone of any single
    short window, and the exact bit pattern makes a wrong match very unlikely.
    A small range of tone offsets around ``tone_offset_hz`` is scanned so a
    detuned signal still matches. Only Robot 36 headers are recognised.
    """
    search = int(min(VIS_SEARCH_MS, 1000.0 * samples.size / sample_rate) * sample_rate / 1000.0)
    seg = samples[:search]
    frame = int(round(SPECTRAL_FRAME_MS * sample_rate / 1000.0))
    window = int(round(SPECTRAL_WINDOW_MS * sample_rate / 1000.0))
    if seg.size < window + frame * 200:
        return None

    per_ms = 1.0 / SPECTRAL_FRAME_MS
    lead = int(round(600 * per_ms))
    bit_frames = int(round(30 * per_ms))
    half = window / (2.0 * frame)
    shift = int(round(half))
    # LSB first: data bits of 0x08, then even-parity bit (1 for a single 1).
    pattern = [(ROBOT36_BITS >> i) & 1 for i in range(7)]
    pattern.append(bin(ROBOT36_BITS).count("1") & 1)

    def prefix(a: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(a)))

    # All the tones an SSTV signal uses; the header's tones must dominate them.
    tones = [1100.0, 1200.0, 1300.0, 1500.0, 1900.0, 2300.0]
    best_overall: tuple | None = None
    for extra in SPECTRAL_OFFSETS_HZ:
        offset = tone_offset_hz + extra
        energy = _tone_energies(seg, sample_rate, [f + offset for f in tones], frame, window)
        # Purity: share of the SSTV-tone energy that sits in each tone, per
        # frame. Picture content mixes tones (and noise spreads evenly), so only
        # a genuine header keeps one tone dominant for the whole timeline.
        purity = energy / np.maximum(energy.sum(axis=0, keepdims=True), 1e-30)
        e1100, e1200, e1300, _e1500, e1900, _e2300 = purity
        p1100, p1200, p1300, p1900 = prefix(e1100), prefix(e1200), prefix(e1300), prefix(e1900)
        frames = e1900.size
        starts = np.arange(lead + 2 + shift, frames - 10 * bit_frames - 2 - shift)
        if starts.size == 0:
            continue
        s_ = starts - shift

        def span(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return p[b] - p[a]

        score = span(p1900, s_ - lead, s_ - 2) / (lead - 2)
        score = score + span(p1200, s_ + 1, s_ + bit_frames - 1) / (bit_frames - 2)
        for i, bit in enumerate(pattern):
            a = s_ + (i + 1) * bit_frames + 1
            b = a + bit_frames - 2
            score = score + span(p1100 if bit else p1300, a, b) / (bit_frames - 2)
        score = score + span(p1200, s_ + 9 * bit_frames + 1, s_ + 10 * bit_frames - 1) / (bit_frames - 2)

        best = int(np.argmax(score))
        # Significance: the header must stand out from every other position,
        # not just from the typical one (flat tones elsewhere also score well).
        median = float(np.median(score))
        sigma = 1.4826 * float(np.median(np.abs(score - median)))
        keep = np.abs(np.arange(score.size) - best) > int(SPECTRAL_EXCLUDE_MS * per_ms)
        elsewhere = float(score[keep].max()) if keep.any() else 0.0
        if (
            score[best] < SPECTRAL_MIN_PEAK_RATIO * elsewhere
            or (score[best] - median) < SPECTRAL_MIN_Z * max(sigma, 1e-9)
        ):
            continue
        if best_overall is None or score[best] > best_overall[0]:
            best_overall = (float(score[best]), int(starts[best]) - shift, offset, e1100, e1300)

    if best_overall is None:
        return None
    start_frame, e1100, e1300 = best_overall[1], best_overall[3], best_overall[4]
    # The template alone cannot tell neighbouring modes apart (Robot 72's code
    # differs from Robot 36's in two bits), so read the bits and insist on the
    # Robot 36 pattern, tolerating one bit error for noisy signals.
    mismatches = 0
    for i, bit in enumerate(pattern):
        a = start_frame + (i + 1) * bit_frames + 1
        b = a + bit_frames - 2
        read = 1 if float(np.sum(e1100[a:b])) > float(np.sum(e1300[a:b])) else 0
        mismatches += int(read != bit)
    if mismatches > SPECTRAL_MAX_BIT_ERRORS:
        return None
    start_bit_ms = (start_frame + half) * SPECTRAL_FRAME_MS
    return VisInfo(
        code=ROBOT36_BITS,
        image_start_ms=start_bit_ms + 10.0 * VIS_BIT_MS,
        start_bit_ms=start_bit_ms,
        parity_ok=True,
    )


def detect_vis(
    samples: np.ndarray,
    sample_rate: int,
    tone_offset_hz: float = 0.0,
    phase_track: np.ndarray | None = None,
) -> VisInfo | None:
    """Detect and decode an SSTV VIS header.

    The instantaneous-frequency detector is tried first (precise on clean
    signals). If it finds no Robot 36 header, a matched-template spectral
    detector is used, which still locks on signals too noisy or detuned for
    per-window frequency estimates.
    ``tone_offset_hz`` shifts the expected tones to follow a Doppler-shifted
    signal; ``phase_track`` may be the precomputed phase of ``samples``.
    """
    info = _detect_vis_phase(samples, sample_rate, tone_offset_hz, phase_track)
    if info is not None and info.code == ROBOT36_BITS and info.parity_ok:
        return info
    # Nothing found, or something that is not Robot 36. A detuned or noisy
    # Robot 36 header can be misread by the precise detector (it judges bits
    # against nominal tone midpoints), so let the offset-scanning template
    # confirm it; if the template finds nothing, report what was read (a
    # genuine header of another mode is then rejected by the caller).
    spectral = _detect_vis_spectral(samples, sample_rate, tone_offset_hz)
    if spectral is not None:
        return spectral
    if info is not None and not _header_tones_ok(samples, sample_rate, info, tone_offset_hz, phase_track):
        # Picture content can look like a leader (mid-grey is 1900 Hz) and its
        # pixels then read as "bits"; that is no header at all.
        return None
    return info


HEADER_TONE_TOLERANCE_HZ = 80.0


def _header_tones_ok(
    samples: np.ndarray,
    sample_rate: int,
    info: VisInfo,
    tone_offset_hz: float,
    phase_track: np.ndarray | None,
) -> bool:
    """True when ``info`` sits on real VIS tones: 1200 Hz start and stop bits
    and data bits at 1100 or 1300 Hz (any mode's header has them)."""
    count = int(min(VIS_SEARCH_MS, 1000.0 * samples.size / sample_rate) * sample_rate / 1000.0)
    if phase_track is None:
        track = phase(samples[:count], sample_rate, tone_offset_hz=tone_offset_hz)
    else:
        track = phase_track[:count]
    end_ms = 1000.0 * track.size / sample_rate

    def tone(bit: int) -> float | None:
        begin = info.start_bit_ms + bit * VIS_BIT_MS + 7.0
        if begin < 0.0 or begin + 16.0 > end_ms:
            return None
        return frequency_between(track, sample_rate, begin, begin + 16.0) - tone_offset_hz

    edges = [tone(0), tone(9)]
    if any(f is None or abs(f - FREQ_SYNC) > HEADER_TONE_TOLERANCE_HZ for f in edges):
        return False
    for bit in range(1, 9):
        f = tone(bit)
        if f is None or min(abs(f - FREQ_VIS_BIT_0), abs(f - FREQ_VIS_BIT_1)) > HEADER_TONE_TOLERANCE_HZ:
            return False
    return True
