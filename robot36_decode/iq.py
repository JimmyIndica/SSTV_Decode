"""IQ-based input handling for Software-Defined Radio (SDR) data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dsp import analytic_signal
from .resample import resample
from .wav import _float_from_pcm, _iter_chunks


@dataclass
class IQData:
    samples: np.ndarray  # Complex64 or Complex128 IQ samples
    sample_rate: int
    bits: int
    encoding: str
    channels: int = 1

    @property
    def duration_ms(self) -> float:
        return 1000.0 * self.samples.size / self.sample_rate


TARGET_RATE = 48000


RAW_FORMATS = ("f32", "s16")


def read_iq(
    path: Path,
    raw_sample_rate: int = TARGET_RATE,
    target_rate: int | None = TARGET_RATE,
    raw_format: str = "f32",
) -> IQData:
    """Read a raw IQ or complex WAV file.

    Signal model: the complex samples are treated as the *analytic signal of
    the SSTV audio* (i.e. SSB-style baseband, where the instantaneous
    frequency of the IQ is the SSTV tone). A Doppler shift or tuning error
    therefore moves every tone by the same number of Hz, which the decoder
    tracks from the line sync pulses. Captures of an FM carrier must be
    FM-demodulated before use; they are not handled here. For lower-sideband
    captures (inverted spectrum) conjugate the samples (``--invert-iq``).

    Accepted inputs:

    * RIFF/RF64 ``WAVE`` files. Two channels are interpreted as complex
      baseband (I in channel 0, Q in channel 1); mono files are converted to a
      complex analytic signal.
    * Raw binary (no WAV header) with interleaved I/Q samples at
      ``raw_sample_rate``: 32-bit float (``raw_format="f32"``, ``.iq`` /
      ``.raw`` style captures) or 16-bit signed integer (``"s16"``, e.g. an
      SDR WAV recording whose header was lost).

    Returns complex samples so the rest of the pipeline (sync tracking, VIS,
    decode) can run unchanged. ``target_rate`` is the rate the samples are
    resampled to; ``None`` keeps the capture's native rate (needed for FM
    baseband captures, which are demodulated before any resampling) and
    returns ``complex64`` to halve the memory of long recordings.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc

    if len(raw) == 0:
        raise ValueError(f"{path} contains no data")

    is_wave = raw[:4] in (b"RIFF", b"RF64") and raw[8:12] == b"WAVE"
    if is_wave:
        fmt_chunk: bytes | None = None
        data_chunk: bytes | None = None
        for chunk_id, chunk in _iter_chunks(raw):
            if chunk_id == b"fmt ":
                fmt_chunk = chunk
            elif chunk_id == b"data":
                data_chunk = chunk
                break

        if fmt_chunk is None or data_chunk is None or len(fmt_chunk) < 16:
            raise ValueError(f"{path} has no valid IQ WAV chunks")
        if int.from_bytes(fmt_chunk[2:4], "little") < 1:
            raise ValueError(f"{path} has an invalid channel count")
        if int.from_bytes(fmt_chunk[4:8], "little") <= 0:
            raise ValueError(f"{path} has an invalid sample rate")
        if int.from_bytes(fmt_chunk[14:16], "little") % 8 or not int.from_bytes(fmt_chunk[14:16], "little"):
            raise ValueError(f"{path} has an invalid sample depth")

        audio_format = int.from_bytes(fmt_chunk[0:2], "little")
        channels = int.from_bytes(fmt_chunk[2:4], "little")
        sample_rate = int.from_bytes(fmt_chunk[4:8], "little")
        bits = int.from_bytes(fmt_chunk[14:16], "little")
        if audio_format == 0xFFFE and len(fmt_chunk) >= 40:
            audio_format = int.from_bytes(fmt_chunk[24:26], "little")

        if audio_format == 1:
            encoding = "PCM"
        elif audio_format == 3:
            encoding = "float"
        else:
            raise ValueError(
                f"{path} uses unsupported WAV format {audio_format} "
                "(only PCM and IEEE float are supported)"
            )

        native = target_rate is None
        if encoding == "PCM" and bits == 16:
            # Fast path for the common SDR recording format: no float copy of
            # the whole file, I and Q go straight into the complex array.
            usable = (len(data_chunk) // 2) * 2
            samples = np.frombuffer(data_chunk[:usable], dtype="<i2")
            scale = 1.0 / 32768.0
        elif encoding == "float" and bits == 32:
            usable = (len(data_chunk) // 4) * 4
            samples = np.frombuffer(data_chunk[:usable], dtype="<f4")
            scale = 1.0
        else:
            samples = _float_from_pcm(data_chunk, bits, encoding)
            scale = 1.0
        if samples.size == 0:
            raise ValueError(f"{path} contains no audio samples")
        if samples.size % channels:
            samples = samples[: samples.size - (samples.size % channels)]
        interleaved = samples.reshape(-1, channels)

        if channels > 1:
            iq = np.empty(interleaved.shape[0], dtype=np.complex64 if native else np.complex128)
            iq.real = interleaved[:, 0]
            iq.imag = interleaved[:, 1]
            if scale != 1.0:
                iq *= scale
        else:
            iq = analytic_signal(interleaved[:, 0].astype(np.float64) * scale, sample_rate)

        out_rate = sample_rate if native else target_rate
        if sample_rate != out_rate:
            iq = resample(iq, sample_rate, out_rate)

        return IQData(
            samples=np.ascontiguousarray(iq, dtype=np.complex64 if native else np.complex128),
            sample_rate=out_rate,
            bits=bits,
            encoding=encoding,
            channels=channels,
        )

    # Raw binary: interleaved I/Q samples without a header.
    if raw_sample_rate <= 0:
        raise ValueError("raw sample rate must be positive")
    if raw_format not in RAW_FORMATS:
        raise ValueError(f"unknown raw format {raw_format!r} (expected one of {', '.join(RAW_FORMATS)})")
    native = target_rate is None
    if raw_format == "s16":
        usable = (len(raw) // 4) * 4
        pairs = np.frombuffer(raw[:usable], dtype="<i2").reshape(-1, 2)
        if pairs.shape[0] < 2:
            raise ValueError(f"{path} contains no I/Q samples")
        iq = np.empty(pairs.shape[0], dtype=np.complex64 if native else np.complex128)
        iq.real = pairs[:, 0]
        iq.imag = pairs[:, 1]
        iq *= 1.0 / 32768.0
        bits, encoding = 16, "int16-iq"
    else:
        usable = (len(raw) // 4) * 4
        samples_f4 = np.frombuffer(raw[:usable], dtype="<f4")
        if samples_f4.size % 2:
            samples_f4 = samples_f4[:-1]
        if samples_f4.size < 2 or not np.all(np.isfinite(samples_f4)):
            raise ValueError(
                f"{path}: unsupported IQ data format (expected a WAV file or interleaved "
                "float32 I/Q; for headerless 16-bit I/Q, such as an SDR WAV recording "
                "whose header is damaged, use --raw-format s16 with --raw-sample-rate)"
            )
        iq = np.empty(samples_f4.size // 2, dtype=np.complex64 if native else np.complex128)
        iq.real = samples_f4[0::2]
        iq.imag = samples_f4[1::2]
        bits, encoding = 32, "float32-iq"
    out_rate = raw_sample_rate if native else target_rate
    if raw_sample_rate != out_rate:
        iq = resample(iq, raw_sample_rate, out_rate)
    return IQData(
        samples=np.ascontiguousarray(iq, dtype=np.complex64 if native else np.complex128),
        sample_rate=out_rate,
        bits=bits,
        encoding=encoding,
    )
