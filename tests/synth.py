"""Synthetic Robot 36 SSTV encoder and test-signal writers.

Generates phase-continuous FM tone sequences (VIS header + 240 lines) so the
decoder can be round-trip tested on clean, noisy, time-scaled and
Doppler-shifted signals, as real audio (mono WAV), complex baseband IQ
(stereo WAV, float or PCM) and raw interleaved float32 IQ.
"""

from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot36_decode.image import ycbcr_to_rgb  # noqa: E402

W, H = 320, 240
FS = 48000


# ---------------------------------------------------------------------------
# Test picture
# ---------------------------------------------------------------------------


def make_test_image() -> np.ndarray:
    """Colour bars, a grey ramp and a smooth colour field (even-sized blocks)."""
    img = np.zeros((H, W, 3), dtype=np.float64)
    bars = [
        (255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
        (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0),
    ]
    for i, colour in enumerate(bars):
        img[:96, i * 40 : (i + 1) * 40] = colour
    ramp = np.arange(W) * 255.0 / (W - 1)
    img[96:144] = ramp[None, :, None]
    y, x = np.mgrid[144:H, 0:W]
    img[144:, :, 0] = 127 + 120 * np.sin(x / 41.0)
    img[144:, :, 1] = 127 + 120 * np.sin(y / 17.0 + x / 90.0)
    img[144:, :, 2] = 127 + 120 * np.cos((x + y) / 33.0)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def rgb_to_ycbcr(rgb: np.ndarray):
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 + (b - y) / 1.772
    cr = 128.0 + (r - y) / 1.402
    return y, cb, cr


# ---------------------------------------------------------------------------
# Robot 36 tone sequence
# ---------------------------------------------------------------------------


def _level_to_hz(level: np.ndarray) -> np.ndarray:
    return 1500.0 + np.clip(level, 0.0, 255.0) * 800.0 / 255.0


class _Painter:
    def __init__(self, n: int, fs: int, scale: float):
        self.f = np.full(n, 1900.0)
        self.n, self.fs, self.scale = n, fs, scale

    def _span(self, start_ms: float, dur_ms: float) -> tuple[int, int]:
        a = int(round(start_ms * self.fs / 1000.0))
        b = int(round((start_ms + dur_ms) * self.fs / 1000.0))
        return max(a, 0), min(b, self.n)

    def tone(self, start_ms: float, dur_ms: float, hz: float) -> None:
        a, b = self._span(start_ms, dur_ms)
        self.f[a:b] = hz / self.scale

    def pixels(self, start_ms: float, pixel_ms: float, hz: np.ndarray) -> None:
        a, b = self._span(start_ms, pixel_ms * hz.size)
        t = np.arange(a, b) * 1000.0 / self.fs
        px = np.clip(np.floor((t - start_ms) / pixel_ms).astype(int), 0, hz.size - 1)
        self.f[a:b] = hz[px] / self.scale


def encode_robot36(
    rgb: np.ndarray,
    *,
    fs: int = FS,
    lead_ms: float = 600.0,
    tail_ms: float = 400.0,
    vis: bool = True,
    vis_code: int = 0x08,
    bad_parity: bool = False,
    time_scale: float = 1.0,
) -> dict:
    """Return ``{freq, active, line_starts_ms, expected_rgb, ...}`` at ``fs``.

    ``time_scale`` > 1 stretches time and lowers every tone by the same factor
    (tape-speed / sample-clock error).
    """
    s = float(time_scale)
    vis_ms = (300 + 10 + 300 + 300) * s if vis else 0.0
    image_start = lead_ms + vis_ms
    total_ms = image_start + 240 * 150.0 * s + tail_ms
    n = int(round(total_ms * fs / 1000.0))
    paint = _Painter(n, fs, s)

    if vis:
        t = lead_ms
        paint.tone(t, 300 * s, 1900.0); t += 300 * s
        paint.tone(t, 10 * s, 1200.0); t += 10 * s
        paint.tone(t, 300 * s, 1900.0); t += 300 * s
        paint.tone(t, 30 * s, 1200.0); t += 30 * s            # start bit
        parity = bin(vis_code).count("1") & 1
        if bad_parity:
            parity ^= 1
        for k in range(7):
            paint.tone(t, 30 * s, 1100.0 if (vis_code >> k) & 1 else 1300.0); t += 30 * s
        paint.tone(t, 30 * s, 1100.0 if parity else 1300.0); t += 30 * s
        paint.tone(t, 30 * s, 1200.0); t += 30 * s            # stop bit
        assert abs(t - image_start) < 1e-6

    y, cb, cr = rgb_to_ycbcr(rgb)
    cr_ds = cr.reshape(120, 2, 160, 2).mean(axis=(1, 3))
    cb_ds = cb.reshape(120, 2, 160, 2).mean(axis=(1, 3))
    expected = ycbcr_to_rgb(
        y,
        np.repeat(np.repeat(cb_ds, 2, axis=0), 2, axis=1),
        np.repeat(np.repeat(cr_ds, 2, axis=0), 2, axis=1),
    )

    starts = []
    for line in range(240):
        t0 = image_start + line * 150.0 * s
        starts.append(t0)
        paint.tone(t0, 9 * s, 1200.0)
        paint.tone(t0 + 9 * s, 3 * s, 1500.0)
        paint.pixels(t0 + 12 * s, 88.0 / 320 * s, _level_to_hz(y[line]))
        even = line % 2 == 0
        paint.tone(t0 + 100 * s, 4.5 * s, 1500.0 if even else 2300.0)
        paint.tone(t0 + 104.5 * s, 1.5 * s, 1900.0)
        chroma = (cr_ds if even else cb_ds)[line // 2]
        paint.pixels(t0 + 106 * s, 44.0 / 160 * s, _level_to_hz(chroma))

    active = np.zeros(n, dtype=bool)
    a = int(round(lead_ms * fs / 1000.0))
    b = int(round((total_ms - tail_ms) * fs / 1000.0))
    active[a:b] = True
    return {
        "freq": paint.f,
        "active": active,
        "fs": fs,
        "line_starts_ms": np.asarray(starts),
        "image_start_ms": image_start,
        "expected_rgb": expected,
        "time_scale": s,
    }


# ---------------------------------------------------------------------------
# Channel models
# ---------------------------------------------------------------------------


def doppler_curve(offset_hz: float = 0.0, rate_hz_per_s: float = 0.0, accel_hz_per_s2: float = 0.0):
    """Return ``offset(t_seconds)`` in Hz."""
    def curve(t: np.ndarray) -> np.ndarray:
        return offset_hz + rate_hz_per_s * t + 0.5 * accel_hz_per_s2 * t * t
    return curve


def to_iq(sig: dict, doppler=None, snr_db: float | None = None, seed: int = 1) -> np.ndarray:
    """Analytic (SSB-style) complex baseband with an optional Doppler curve."""
    fs = sig["fs"]
    f = sig["freq"].copy()
    t = np.arange(f.size) / fs
    if doppler is not None:
        f = f + doppler(t)
    phi = 2.0 * np.pi * np.cumsum(f) / fs
    iq = np.where(sig["active"], 1.0, 0.0) * np.exp(1j * phi)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(10 ** (-snr_db / 10.0) / 2.0)
        iq = iq + sigma * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    return iq


def to_audio(sig: dict, offset_hz: float = 0.0, snr_db: float | None = None, seed: int = 1) -> np.ndarray:
    """Real audio; ``offset_hz`` is a constant tuning error added to every tone."""
    fs = sig["fs"]
    phi = 2.0 * np.pi * np.cumsum(sig["freq"] + offset_hz) / fs
    x = np.where(sig["active"], 0.8, 0.0) * np.cos(phi)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        power = 0.8 ** 2 / 2.0
        x = x + np.sqrt(power * 10 ** (-snr_db / 10.0)) * rng.standard_normal(x.size)
    return x


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_wav_pcm16(path: Path, data: np.ndarray, fs: int) -> None:
    """``data`` is (n,) or (n, channels), float in [-1, 1]."""
    data = np.atleast_2d(data.T).T if data.ndim == 1 else data
    pcm = np.clip(np.rint(data * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


def write_wav_float32(path: Path, data: np.ndarray, fs: int) -> None:
    data = np.atleast_2d(data.T).T if data.ndim == 1 else data
    payload = data.astype("<f4").tobytes()
    channels = data.shape[1]
    fmt = struct.pack("<HHIIHH", 3, channels, fs, fs * channels * 4, channels * 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_iq_wav(path: Path, iq: np.ndarray, fs: int, kind: str = "float32") -> None:
    scale = 1.0 / max(1.0, float(np.max(np.abs(iq))) * 1.05)
    stereo = np.stack([iq.real, iq.imag], axis=1) * scale
    (write_wav_float32 if kind == "float32" else write_wav_pcm16)(path, stereo, fs)


def write_raw_iq(path: Path, iq: np.ndarray) -> None:
    inter = np.empty(iq.size * 2, dtype="<f4")
    inter[0::2], inter[1::2] = iq.real, iq.imag
    path.write_bytes(inter.tobytes())


def resample_to(x: np.ndarray, fs_from: int, fs_to: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(fs_from, fs_to)
    return resample_poly(x, fs_to // g, fs_from // g)


def read_png_rgb(path: Path) -> np.ndarray:
    import zlib
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(raw):
        ln = struct.unpack(">I", raw[pos : pos + 4])[0]
        tag = raw[pos + 4 : pos + 8]
        body = raw[pos + 8 : pos + 8 + ln]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + ln
    data = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * 3)
    assert not data[:, 0].any(), "decoder PNG writer is expected to use filter 0"
    return data[:, 1:].reshape(h, w, 3)


# ---------------------------------------------------------------------------
# FM baseband (SDR-recording style)
# ---------------------------------------------------------------------------


def to_fm_baseband(
    sig: dict,
    *,
    fs_out: int = 78125,
    dead_before_s: float = 20.0,
    dead_after_s: float = 25.0,
    carrier=None,
    peak_deviation_hz: float = 4000.0,
    inband_snr_db: float | None = 14.0,
    burst: tuple[float, float, float] | None = None,
    seed: int = 7,
) -> np.ndarray:
    """FM-modulate the SSTV audio onto a Doppler-shifted carrier at ``fs_out``.

    ``carrier(t_seconds)`` is the carrier offset in Hz from the tuned centre.
    The noise level is set so the signal-to-noise ratio inside a 12 kHz channel
    is ``inband_snr_db``. ``burst=(start_s, length_s, level_db)`` adds a wideband
    noise burst (relative to the noise floor) like the ones in real captures.
    """
    audio = to_audio(sig)
    audio = resample_to(audio, sig["fs"], fs_out)
    audio = audio / 0.8  # unit peak
    n_sig = audio.size
    n_before = int(round(dead_before_s * fs_out))
    n_after = int(round(dead_after_s * fs_out))
    n = n_before + n_sig + n_after
    t = np.arange(n) / fs_out
    f = np.zeros(n)
    f[n_before : n_before + n_sig] = peak_deviation_hz * audio
    if carrier is not None:
        f += carrier(t)
    phase = 2.0 * np.pi * np.cumsum(f) / fs_out
    active = np.zeros(n)
    active[n_before : n_before + n_sig] = 1.0
    iq = active * np.exp(1j * phase)

    rng = np.random.default_rng(seed)
    if inband_snr_db is not None:
        sigma2 = 10 ** (-inband_snr_db / 10.0) * (fs_out / 12000.0)  # unit-power signal
    else:
        sigma2 = 1e-6
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(sigma2 / 2.0)
    if burst is not None:
        b0, bl, level_db = burst
        a, b = int(b0 * fs_out), int((b0 + bl) * fs_out)
        noise[a:b] *= 10 ** (level_db / 20.0)
    return iq + noise
