"""RIFF/WAVE input handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .resample import resample


@dataclass
class WavData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    bits: int
    encoding: str

    @property
    def duration_ms(self) -> float:
        return 1000.0 * self.samples.size / self.sample_rate


def _float_from_pcm(data: bytes, bits: int, encoding: str) -> np.ndarray:
    """Convert interleaved PCM/float sample bytes to float64 in [-1, 1]."""
    bytes_per_sample = bits // 8
    usable = (len(data) // bytes_per_sample) * bytes_per_sample
    data = data[:usable]

    if encoding == "float":
        if bits == 32:
            return np.frombuffer(data, dtype="<f4").astype(np.float64)
        if bits == 64:
            return np.frombuffer(data, dtype="<f8").astype(np.float64)
        raise ValueError(f"unsupported floating-point WAV depth: {bits} bits")

    if bits == 8:
        return (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if bits == 16:
        return np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    if bits == 24:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        value = np.where(value & 0x800000, value - 0x1000000, value)
        return value.astype(np.float64) / 8388608.0
    if bits == 32:
        return np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(f"unsupported PCM WAV depth: {bits} bits")


def _iter_chunks(raw: bytes):
    """Yield ``(chunk_id, body)`` pairs for a RIFF or RF64 WAVE payload.

    Both container flavours share the RIFF chunk layout in practice (chunk
    size is a 32-bit little-endian field); RF64 files just carry the magic and
    use an ``0xFFFFFFFF`` size placeholder plus a ``ds64`` chunk for the real
    (possibly >4 GiB) values. Oversized placeholder sizes are clamped to the
    remaining buffer.
    """
    pos = 12
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        chunk_size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        body_start = pos + 8
        end = body_start + chunk_size
        yield chunk_id, raw[body_start : min(end, len(raw))]
        pos = end + (chunk_size & 1)
        if chunk_id == b"data":
            return


def read_wav(path: Path) -> WavData:
    """Read a RIFF/WAVE file.

    Supports PCM 8/16/24/32-bit and IEEE float 32/64-bit, including
    WAVE_FORMAT_EXTENSIBLE. Multi-channel audio is mixed down to mono.

    Also accepts RF64 containers (``RF64`` magic with 64-bit chunk sizes),
    as written by several SDR decoders for large or float32 payloads.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc

    if (
        len(raw) < 16
        or raw[8:12] != b"WAVE"
        or raw[:4] not in (b"RIFF", b"RF64")
    ):
        raise ValueError(f"{path} is not a RIFF/WAVE file")

    fmt_chunk: bytes | None = None
    data_chunk: bytes | None = None
    for chunk_id, chunk in _iter_chunks(raw):
        if chunk_id == b"fmt ":
            fmt_chunk = chunk
        elif chunk_id == b"data":
            data_chunk = chunk
            break

    if fmt_chunk is None or len(fmt_chunk) < 16:
        raise ValueError(f"{path} has no valid fmt chunk")
    if data_chunk is None:
        raise ValueError(f"{path} has no data chunk")

    audio_format = int.from_bytes(fmt_chunk[0:2], "little")
    channels = int.from_bytes(fmt_chunk[2:4], "little")
    sample_rate = int.from_bytes(fmt_chunk[4:8], "little")
    bits = int.from_bytes(fmt_chunk[14:16], "little")

    if audio_format == 0xFFFE and len(fmt_chunk) >= 40:
        # WAVE_FORMAT_EXTENSIBLE: the first two bytes of the SubFormat GUID
        # hold the actual format tag.
        audio_format = int.from_bytes(fmt_chunk[24:26], "little")

    if channels < 1:
        raise ValueError(f"{path} has an invalid channel count ({channels})")
    if sample_rate <= 0:
        raise ValueError(f"{path} has an invalid sample rate ({sample_rate})")
    if bits <= 0 or bits % 8:
        raise ValueError(f"{path} has an invalid sample depth ({bits} bits)")

    if audio_format == 1:
        encoding = "PCM"
    elif audio_format == 3:
        encoding = "float"
    else:
        raise ValueError(
            f"{path} uses unsupported WAV format {audio_format} "
            "(only PCM and IEEE float are supported)"
        )

    samples = _float_from_pcm(data_chunk, bits, encoding)
    if samples.size == 0:
        raise ValueError(f"{path} contains no audio samples")

    if samples.size % channels:
        samples = samples[: samples.size - (samples.size % channels)]

    samples = samples.reshape(-1, channels)
    if channels > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples[:, 0]

    # Resample to 48000 Hz if needed
    if sample_rate != 48000:
        samples = resample(samples, sample_rate, 48000)

    return WavData(
        samples=np.ascontiguousarray(samples, dtype=np.float64),
        sample_rate=48000,
        channels=channels,
        bits=bits,
        encoding=encoding,
    )
