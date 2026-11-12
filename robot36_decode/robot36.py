"""Robot 36 colour SSTV decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    CHROMA_PIXEL_MS,
    CHROMA_START_MS,
    FREQ_BLACK,
    FREQ_SEPARATOR_THRESHOLD,
    FREQ_WHITE,
    LINE_MS,
    LINES,
    SEPARATOR_MS,
    SEPARATOR_START_MS,
    WIDTH,
    Y_PIXEL_MS,
    Y_START_MS,
)
from .dsp import frequency_between, phase, segment_frequencies
from .image import ycbcr_to_rgb

# Edge pixels sit next to a tone change that the demodulator and band-pass
# filter smear over ~0.5 ms (see dsp.segment_frequencies). Pixels are 0.275 ms.
Y_LEAD_GUARD_MS = 0.12  # after the 1500 Hz porch
Y_TAIL_GUARD_MS = 0.60  # before the separator tone (odd lines: 2300 Hz)
CHROMA_TAIL_GUARD_MS = 0.60  # before the next 1200 Hz sync pulse


@dataclass
class DecodeStats:
    separator_mismatches: int
    line_scale: float = 1.0
    repaired_lines: int = 0


REPAIR_MIN_DEVIATION = 30.0  # levels (0-255) of mean deviation before a line is suspect
REPAIR_MEDIAN_FACTOR = 4.0  # ... and this many times the picture's typical deviation
REPAIR_NEIGHBOUR_AGREEMENT = 0.5  # neighbours must agree to within this x the deviation


def _repair_rows(rows: np.ndarray, step: int) -> int:
    """Interpolate rows that were hit by an interference burst, in place.

    A row is replaced by the mean of the rows ``step`` above and below only if
    it differs strongly from both while those two agree with each other. A real
    horizontal edge or text line has neighbours that differ from each other and
    so is left alone, as is a region that is noisy throughout. Returns how many
    rows were replaced.
    """
    count = rows.shape[0]
    if count <= 2 * step:
        return 0
    above, below = rows[: count - 2 * step], rows[2 * step :]
    middle = rows[step : count - step]
    deviation = np.mean(np.abs(middle - 0.5 * (above + below)), axis=1)
    agreement = np.mean(np.abs(above - below), axis=1)
    limit = max(REPAIR_MIN_DEVIATION, REPAIR_MEDIAN_FACTOR * float(np.median(deviation)))
    bad = (deviation > limit) & (agreement < REPAIR_NEIGHBOUR_AGREEMENT * deviation)
    index = np.flatnonzero(bad) + step
    rows[index] = 0.5 * (rows[index - step] + rows[index + step])
    return int(index.size)


def _frequencies_to_levels(
    frequencies: np.ndarray, black_hz: float, white_hz: float
) -> np.ndarray:
    """Map tone frequencies to full-range 0-255 levels (black -> 0, white -> 255).

    ``ycbcr_to_rgb`` expects full-range YCbCr, so the levels are clipped to
    the full 0-255 range rather than to a video range.
    """
    levels = (frequencies - black_hz) * 255.0 / (white_hz - black_hz)
    return np.clip(levels, 0.0, 255.0).astype(np.float64)


def decode_robot36(
    samples: np.ndarray,
    sample_rate: int,
    black_hz: float = FREQ_BLACK,
    white_hz: float = FREQ_WHITE,
    line_starts_ms: np.ndarray | None = None,
    tone_offset_hz: float = 0.0,
    line_offsets_hz: np.ndarray | None = None,
    phase_track: np.ndarray | None = None,
    repair_lines: bool = False,
) -> tuple[np.ndarray, DecodeStats]:
    """Decode Robot 36 audio.

    By default the first sample is assumed to be the first line's sync. Pass
    ``line_starts_ms`` (one entry per line, relative to the first sample) to
    decode a tracked recording instead. The median line spacing is used to
    compensate for a sample-clock or tape-speed error.

    ``tone_offset_hz`` shifts the analysis band and, when ``line_offsets_hz``
    is not given, the black/white tone references so they follow a constant
    Doppler shift. ``line_offsets_hz`` (one entry per line, in Hz) overrides
    that with a per-line offset, so a drifting Doppler shift is removed line by
    line; the offsets are measured against the time-scaled nominal tones, as
    produced by the sync tracker. ``phase_track`` may be the precomputed phase
    of ``samples``.
    """
    if phase_track is None:
        phase_track = phase(samples, sample_rate, tone_offset_hz=tone_offset_hz)

    if line_starts_ms is None:
        starts = np.arange(LINES, dtype=np.float64) * LINE_MS
        line_scale = 1.0
    else:
        starts = np.asarray(line_starts_ms, dtype=np.float64)
        if starts.shape != (LINES,):
            raise ValueError(
                f"line_starts_ms must contain {LINES} entries, got {starts.size}"
            )
        spacings = np.diff(starts)
        line_scale = float(np.median(spacings) / LINE_MS) if spacings.size else 1.0
        line_scale = float(np.clip(line_scale, 0.5, 2.0))

    # A sample-clock or tape-speed error stretches the audio in time and
    # lowers the tone frequencies by the same factor. Once sync tracking has
    # measured the line scale, divide the reference tones by it, then add the
    # per-line frequency (Doppler) offset.
    if line_offsets_hz is None:
        offsets = np.full(LINES, tone_offset_hz / line_scale, dtype=np.float64)
    else:
        offsets = np.asarray(line_offsets_hz, dtype=np.float64)
        if offsets.shape != (LINES,):
            raise ValueError(
                f"line_offsets_hz must contain {LINES} entries, got {offsets.size}"
            )

    luma = np.empty((LINES, WIDTH), dtype=np.float64)
    chroma = np.empty((LINES, WIDTH), dtype=np.float64)
    separator_is_cb = np.zeros(LINES, dtype=bool)

    y_pixel_ms = Y_PIXEL_MS * line_scale
    # Robot 36 chroma is only WIDTH // 2 pixels wide; measure it at that
    # resolution (twice the time per pixel, so half the noise) and repeat.
    chroma_pixel_ms = 2.0 * CHROMA_PIXEL_MS * line_scale
    chroma_start_offset = CHROMA_START_MS * line_scale
    separator_offset = SEPARATOR_START_MS * line_scale
    separator_ms = SEPARATOR_MS * line_scale

    for line, line_start in enumerate(starts):
        offset = offsets[line]
        black = black_hz / line_scale + offset
        white = white_hz / line_scale + offset
        separator_threshold_hz = FREQ_SEPARATOR_THRESHOLD / line_scale + offset

        luma_start = line_start + Y_START_MS * line_scale
        y_freqs = segment_frequencies(
            phase_track,
            sample_rate,
            luma_start,
            y_pixel_ms,
            WIDTH,
            lead_guard_ms=Y_LEAD_GUARD_MS * line_scale,
            tail_guard_ms=Y_TAIL_GUARD_MS * line_scale,
        )
        luma[line] = _frequencies_to_levels(y_freqs, black, white)

        chroma_freqs = segment_frequencies(
            phase_track,
            sample_rate,
            line_start + chroma_start_offset,
            chroma_pixel_ms,
            WIDTH // 2,
            tail_guard_ms=CHROMA_TAIL_GUARD_MS * line_scale,
        )
        chroma[line] = np.repeat(_frequencies_to_levels(chroma_freqs, black, white), 2)

        separator_frequency = frequency_between(
            phase_track,
            sample_rate,
            line_start + separator_offset,
            line_start + separator_offset + separator_ms,
        )
        separator_is_cb[line] = separator_frequency > separator_threshold_hz

    repaired = 0
    if repair_lines:
        # Luma rows are independent; chroma rows alternate Cr/Cb, so a row's
        # same-component neighbours are two lines away.
        repaired += _repair_rows(luma, 1)
        repaired += _repair_rows(chroma, 2)

    # The header alternates Cr and Cb, but use a majority vote over the whole
    # frame so a single bad separator cannot swap the colours.
    even_is_cb = bool(
        np.count_nonzero(separator_is_cb[0::2])
        > np.count_nonzero(~separator_is_cb[0::2])
    )
    expected = np.zeros(LINES, dtype=bool)
    expected[0::2] = even_is_cb
    expected[1::2] = not even_is_cb
    separator_mismatches = int(np.count_nonzero(separator_is_cb != expected))

    # Robot 36 uses 4:2:0 colour: each Cr/Cb pair is shared by both luma rows
    # in the line pair.
    cb = np.empty_like(luma)
    cr = np.empty_like(luma)
    for pair in range(LINES // 2):
        first = pair * 2
        second = first + 1
        if even_is_cb:
            cb[first] = cb[second] = chroma[first]
            cr[first] = cr[second] = chroma[second]
        else:
            cr[first] = cr[second] = chroma[first]
            cb[first] = cb[second] = chroma[second]

    stats = DecodeStats(
        separator_mismatches=separator_mismatches,
        line_scale=line_scale,
        repaired_lines=repaired,
    )
    return ycbcr_to_rgb(luma, cb, cr), stats
