"""Robot 36 timing and tone constants shared across the decoder."""

from __future__ import annotations

__all__ = [
    "ROBOT36_VIS_CODE",
    "WIDTH",
    "HEIGHT",
    "LINES",
    "LINE_MS",
    "SYNC_MS",
    "SYNC_PORCH_MS",
    "Y_START_MS",
    "Y_SCAN_MS",
    "SEPARATOR_START_MS",
    "SEPARATOR_MS",
    "CHROMA_PORCH_START_MS",
    "CHROMA_PORCH_MS",
    "CHROMA_START_MS",
    "CHROMA_SCAN_MS",
    "Y_PIXEL_MS",
    "CHROMA_PIXEL_MS",
    "IMAGE_MS",
    "FREQ_SYNC",
    "FREQ_BLACK",
    "FREQ_WHITE",
    "FREQ_CHROMA_PORCH",
    "FREQ_VIS_BIT_0",
    "FREQ_VIS_BIT_1",
    "FREQ_VIS_LEADER",
    "FREQ_SEPARATOR_THRESHOLD",
    "VIS_LEADER_MS",
    "VIS_BREAK_MS",
    "VIS_BIT_MS",
    "VIS_HEADER_MS",
    "ANALYSIS_BAND_LOW_HZ",
    "ANALYSIS_BAND_HIGH_HZ",
    "ANALYSIS_BAND_TAPER_HZ",
    "ANALYSIS_BAND_AUTO_RATIO",
]

# ---------------------------------------------------------------------------
# Robot 36 colour mode
# ---------------------------------------------------------------------------

ROBOT36_VIS_CODE = 0x08

WIDTH = 320
HEIGHT = 240
LINES = HEIGHT

LINE_MS = 150.0
SYNC_MS = 9.0
SYNC_PORCH_MS = 3.0
Y_START_MS = 12.0
Y_SCAN_MS = 88.0
SEPARATOR_START_MS = 100.0
SEPARATOR_MS = 4.5
CHROMA_PORCH_START_MS = 104.5
CHROMA_PORCH_MS = 1.5
CHROMA_START_MS = 106.0
CHROMA_SCAN_MS = 44.0

Y_PIXEL_MS = Y_SCAN_MS / WIDTH
CHROMA_PIXEL_MS = CHROMA_SCAN_MS / WIDTH
IMAGE_MS = LINES * LINE_MS

# ---------------------------------------------------------------------------
# SSTV tones
# ---------------------------------------------------------------------------

FREQ_SYNC = 1200.0
FREQ_BLACK = 1500.0
FREQ_WHITE = 2300.0
FREQ_CHROMA_PORCH = 1900.0
# VIS data bits: logic 1 is 1100 Hz and logic 0 is 1300 Hz (not the other
# way round), LSB first.
FREQ_VIS_BIT_0 = 1300.0
FREQ_VIS_BIT_1 = 1100.0
FREQ_VIS_LEADER = 1900.0
FREQ_SEPARATOR_THRESHOLD = 1900.0

# ---------------------------------------------------------------------------
# Analysis band
# ---------------------------------------------------------------------------
# Real recordings often contain broadband noise, adjacent-channel audio, or
# high-frequency recording artefacts. Band-limiting before the Hilbert
# transform keeps that energy from corrupting the phase estimate. The band is
# much wider than the SSTV tones so a modest tuning or tape-speed error is
# still accepted; the filter is brick-wall (zero taper) over the passband.
ANALYSIS_BAND_LOW_HZ = 300.0
ANALYSIS_BAND_HIGH_HZ = 3000.0
ANALYSIS_BAND_TAPER_HZ = 0.0

# Band-limiting is applied automatically only when out-of-band energy is a
# meaningful fraction of the in-band energy. Clean, synthetic, or already
# filtered recordings therefore keep their original response.
ANALYSIS_BAND_AUTO_RATIO = 0.02

# ---------------------------------------------------------------------------
# VIS header timing
# ---------------------------------------------------------------------------

VIS_LEADER_MS = 300.0
VIS_BREAK_MS = 10.0
VIS_BIT_MS = 30.0
VIS_HEADER_MS = VIS_LEADER_MS + VIS_BREAK_MS + VIS_LEADER_MS + 10 * VIS_BIT_MS
