"""Round-trip tests for the Robot 36 decoder using a synthetic encoder.

Run directly (``python tests/test_roundtrip.py``) for a results table, or with
pytest. Each image case encodes the test picture, writes it as a WAV/IQ file,
decodes it through the real CLI entry point and compares the PNG with the
picture the encoder *intended* to send (after Robot 36's own chroma
subsampling), reporting the mean absolute error per channel value (0-255).
"""

from __future__ import annotations

import atexit
import contextlib
import io
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import synth  # noqa: E402
from robot36_decode.cli import main  # noqa: E402
from robot36_decode.dsp import band_limit, phase_values  # noqa: E402
from robot36_decode.iq import read_iq  # noqa: E402
from robot36_decode.resample import resample  # noqa: E402
from robot36_decode.robot36 import _frequencies_to_levels  # noqa: E402
from robot36_decode.synctrack import track_line_starts  # noqa: E402

FS = synth.FS
IMG = synth.make_test_image()
_SIG_CACHE: dict = {}
RESULTS: list[tuple[str, str, float | None, float]] = []


def _signal(**kw) -> dict:
    key = tuple(sorted(kw.items()))
    if key not in _SIG_CACHE:
        _SIG_CACHE[key] = synth.encode_robot36(IMG, **kw)
    return _SIG_CACHE[key]


def _decode(path: Path, *flags: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main([str(path), *flags])
    return rc, out.getvalue(), err.getvalue()


def _check(name: str, path: Path, sig: dict, max_mae: float, *flags: str) -> None:
    t0 = time.time()
    png = path.with_suffix(".png")
    if png.exists():
        png.unlink()
    rc, out, err = _decode(path, *flags)
    assert rc == 0, f"{name}: decoder failed: {err.strip()}"
    got = synth.read_png_rgb(png).astype(np.float64)
    mae = float(np.abs(got - sig["expected_rgb"]).mean())
    RESULTS.append((name, "ok" if mae <= max_mae else "FAIL", mae, max_mae))
    assert mae <= max_mae, f"{name}: MAE {mae:.2f} > {max_mae} (decoder said:\n{out})"
    _ = time.time() - t0


# Every test writes under one folder that is deleted when the run ends (a full
# run writes several GB of synthetic captures). Set R36_KEEP_TMP=1 to keep it.
_TMP_ROOT = tempfile.mkdtemp(prefix="r36_tests_")
if not os.environ.get("R36_KEEP_TMP"):
    atexit.register(shutil.rmtree, _TMP_ROOT, True)


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="r36_", dir=_TMP_ROOT))


# ---------------------------------------------------------------------------
# Audio WAV inputs
# ---------------------------------------------------------------------------


def test_audio_clean():
    d, sig = _tmp(), _signal()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    _check("audio clean 48k", d / "a.wav", sig, 2.0)


def test_audio_44k1_resampled():
    d, sig = _tmp(), _signal()
    x = synth.resample_to(synth.to_audio(sig), FS, 44100)
    synth.write_wav_pcm16(d / "a.wav", x, 44100)
    _check("audio 44.1k -> 48k", d / "a.wav", sig, 3.0)


def test_audio_float32_stereo_mixdown():
    d, sig = _tmp(), _signal()
    x = synth.to_audio(sig)
    synth.write_wav_float32(d / "a.wav", np.stack([x, x], axis=1), FS)
    _check("audio stereo float32", d / "a.wav", sig, 2.0)


def test_audio_noise_and_leading_silence():
    d, sig = _tmp(), _signal(lead_ms=3500.0, tail_ms=1500.0)
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig, snr_db=14.0), FS)
    _check("audio noisy (14 dB) + 3.5 s lead", d / "a.wav", sig, 16.0)


def test_audio_tape_speed_error():
    d, sig = _tmp(), _signal(time_scale=1.004)
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    _check("audio time-scaled x1.004", d / "a.wav", sig, 3.0)


def test_audio_tuning_offset():
    d, sig = _tmp(), _signal()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig, offset_hz=150.0), FS)
    _check("audio tuning offset +150 Hz", d / "a.wav", sig, 3.0)


def test_audio_no_vis():
    d, sig = _tmp(), _signal(vis=False)
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    _check("audio without VIS header", d / "a.wav", sig, 3.0)


def test_audio_bad_vis_parity_falls_back():
    d, sig = _tmp(), _signal(bad_parity=True)
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    _check("audio VIS parity error -> fallback", d / "a.wav", sig, 3.0)


def test_audio_wrong_vis_code_is_rejected():
    d, sig = _tmp(), _signal(vis_code=0x0C)  # Robot 72 colour
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    rc, _, err = _decode(d / "a.wav")
    assert rc == 1 and "not Robot 36" in err


def test_audio_no_sync_has_no_spurious_offset():
    """--no-sync used to force a second pass with an offset taken from the
    spectrum's strongest bin (picture dependent)."""
    d, sig = _tmp(), _signal(lead_ms=0.0, vis=False)
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    _check("audio --no-sync (fixed 150 ms)", d / "a.wav", sig, 3.0, "--no-sync", "--no-vis")


# ---------------------------------------------------------------------------
# IQ inputs
# ---------------------------------------------------------------------------


def test_iq_clean_float32_wav():
    d, sig = _tmp(), _signal()
    synth.write_iq_wav(d / "i.wav", synth.to_iq(sig), FS)
    _check("iq clean float32 wav", d / "i.wav", sig, 0.5, "--format", "iq")


def test_iq_clean_pcm16_wav():
    d, sig = _tmp(), _signal()
    synth.write_iq_wav(d / "i.wav", synth.to_iq(sig), FS, kind="pcm16")
    _check("iq clean pcm16 wav", d / "i.wav", sig, 1.5, "--format", "iq")


def test_iq_96k_wav():
    d, sig = _tmp(), _signal(fs=96000)
    synth.write_iq_wav(d / "i.wav", synth.to_iq(sig), 96000)
    _check("iq 96 kHz wav -> 48k", d / "i.wav", sig, 1.5, "--format", "iq")


def test_iq_mono_wav_analytic_path():
    d, sig = _tmp(), _signal()
    synth.write_wav_pcm16(d / "i.wav", synth.to_audio(sig), FS)
    _check("iq mono wav (Hilbert path)", d / "i.wav", sig, 2.0, "--format", "iq")


def test_iq_raw_float32():
    d, sig = _tmp(), _signal()
    synth.write_raw_iq(d / "i.iq", synth.to_iq(sig))
    _check("iq raw f32 48k", d / "i.iq", sig, 0.5, "--format", "iq")


def test_iq_raw_float32_other_rate():
    d, sig = _tmp(), _signal(fs=96000)
    synth.write_raw_iq(d / "i.iq", synth.to_iq(sig))
    _check("iq raw f32 96k (--raw-sample-rate)", d / "i.iq", sig, 1.5,
           "--format", "iq", "--raw-sample-rate", "96000")


def test_iq_inverted_spectrum():
    d, sig = _tmp(), _signal()
    synth.write_iq_wav(d / "i.wav", np.conj(synth.to_iq(sig)), FS)
    _check("iq inverted (--invert-iq)", d / "i.wav", sig, 0.5, "--format", "iq", "--invert-iq")


def test_iq_noise_and_leading_silence():
    d, sig = _tmp(), _signal(lead_ms=3500.0, tail_ms=1500.0)
    synth.write_iq_wav(d / "i.wav", synth.to_iq(sig, snr_db=14.0), FS)
    _check("iq noisy (14 dB) + 3.5 s lead", d / "i.wav", sig, 14.0, "--format", "iq")


def test_iq_tape_speed_error():
    d, sig = _tmp(), _signal(time_scale=1.004)
    synth.write_iq_wav(d / "i.wav", synth.to_iq(sig), FS)
    _check("iq time-scaled x1.004", d / "i.wav", sig, 1.5, "--format", "iq")


def test_iq_no_sync_has_no_spurious_offset():
    d, sig = _tmp(), _signal(lead_ms=0.0, vis=False)
    synth.write_iq_wav(d / "i.wav", synth.to_iq(sig), FS)
    _check("iq --no-sync (fixed 150 ms)", d / "i.wav", sig, 1.0,
           "--format", "iq", "--no-sync", "--no-vis")


# ---------------------------------------------------------------------------
# Doppler shift (IQ)
# ---------------------------------------------------------------------------

DOPPLER_CASES = [
    ("+100 Hz", synth.doppler_curve(100), 1.0),
    ("+300 Hz", synth.doppler_curve(300), 1.0),
    ("-250 Hz", synth.doppler_curve(-250), 1.0),
    ("+900 Hz", synth.doppler_curve(900), 1.0),
    ("-800 Hz", synth.doppler_curve(-800), 1.0),
    ("+500 Hz, +10 Hz/s", synth.doppler_curve(500, 10), 2.0),
    ("+1200 Hz, -40 Hz/s (1.4 kHz sweep)", synth.doppler_curve(1200, -40), 4.0),
    ("+800 Hz, -30 Hz/s, +1.5 Hz/s^2 (curved)", synth.doppler_curve(800, -30, 1.5), 2.0),
]


def test_iq_doppler_cases():
    failures = []
    for label, curve, limit in DOPPLER_CASES:
        d, sig = _tmp(), _signal()
        synth.write_iq_wav(d / "i.wav", synth.to_iq(sig, doppler=curve), FS)
        try:
            _check(f"iq Doppler {label}", d / "i.wav", sig, limit, "--format", "iq")
        except AssertionError as exc:
            failures.append(str(exc).splitlines()[0])
    assert not failures, "; ".join(failures)


def test_iq_doppler_with_noise_and_lead():
    d, sig = _tmp(), _signal(lead_ms=3500.0, tail_ms=1500.0)
    curve = synth.doppler_curve(-600, 12)
    synth.write_iq_wav(d / "i.wav", synth.to_iq(sig, doppler=curve, snr_db=14.0), FS)
    _check("iq Doppler -600 Hz, +12 Hz/s, 14 dB, 3.5 s lead", d / "i.wav", sig, 14.0,
           "--format", "iq")


def test_iq_doppler_plus_time_scale():
    d, sig = _tmp(), _signal(time_scale=1.003)
    synth.write_iq_wav(d / "i.wav", synth.to_iq(sig, doppler=synth.doppler_curve(400, -8)), FS)
    _check("iq Doppler +400 Hz, -8 Hz/s + clock x1.003", d / "i.wav", sig, 3.0, "--format", "iq")


def test_doppler_measurement_accuracy():
    """The tracker's per-line offset should follow the true Doppler curve."""
    sig = _signal()
    worst = 0.0
    for label, curve, _ in DOPPLER_CASES:
        iq = synth.to_iq(sig, doppler=curve)
        track = track_line_starts(iq, FS, expected_start_ms=sig["image_start_ms"])
        assert track is not None and track.matched_lines >= 235, f"{label}: weak track"
        t_s = (sig["line_starts_ms"] + 5.0) / 1000.0  # middle of each sync pulse
        err = np.abs(track.line_offsets_hz - curve(t_s))
        worst = max(worst, float(err.max()))
        assert err.max() < 20.0, f"{label}: max offset error {err.max():.1f} Hz"
    RESULTS.append(("Doppler offset error (worst, Hz)", "ok", worst, 20.0))


# ---------------------------------------------------------------------------
# Unit-level checks for the individual fixes
# ---------------------------------------------------------------------------


def test_sync_refinement_survives_time_scale():
    """Re-refinement of a time-scaled recording used to receive the sync
    *duration* where the sync *frequency* belongs, silently losing the
    sub-millisecond start refinement."""
    sig = _signal(time_scale=1.003)
    iq = synth.to_iq(sig)
    track = track_line_starts(iq, FS, expected_start_ms=sig["image_start_ms"])
    assert track is not None
    err = np.abs(track.line_starts_ms - sig["line_starts_ms"])
    assert abs(track.line_scale - 1.003) < 2e-4, track.line_scale
    assert np.median(err) < 0.1 and err.max() < 0.5, (np.median(err), err.max())
    RESULTS.append(("line-start error, time-scaled (max ms)", "ok", float(err.max()), 0.5))


def test_complex_band_limit_respects_sign():
    n, fs = 48000, 48000
    t = np.arange(n) / fs
    positive = np.exp(2j * np.pi * 1500 * t)
    negative = np.exp(-2j * np.pi * 1500 * t)
    kept = band_limit(positive + 0.5 * negative, fs, auto=True)
    assert np.abs(np.fft.fft(kept)[1500]) > 0.9 * n
    assert np.abs(np.fft.fft(kept)[-1500]) < 1e-6 * n
    wiped = band_limit(negative, fs, auto=True)
    assert np.abs(wiped).max() < 1e-6


def test_phase_values_matches_interp():
    ph = np.cumsum(np.random.default_rng(0).uniform(0.1, 0.5, 5000))
    t = np.linspace(0.0, 5000 / 48.0 - 1.0, 777)
    ref = np.interp(t * 48.0, np.arange(ph.size), ph)
    assert np.allclose(phase_values(ph, 48000, t), ref)


def test_level_mapping_is_full_range():
    levels = _frequencies_to_levels(np.array([1500.0, 1900.0, 2300.0]), 1500.0, 2300.0)
    assert np.allclose(levels, [0.0, 127.5, 255.0])


def test_resample_poly_keeps_tone_and_length():
    fs_in = 250000
    t = np.arange(fs_in) / fs_in
    x = np.exp(2j * np.pi * 1700 * t)
    y = resample(x, fs_in, 48000)
    assert abs(y.size - 48000) <= 1
    f = np.fft.fftfreq(y.size, 1 / 48000)[np.argmax(np.abs(np.fft.fft(y)))]
    assert abs(f - 1700) < 2


def test_read_iq_rejects_bad_wav_header(tmp=None):
    d = _tmp()
    bad = d / "bad.wav"
    header = (b"RIFF" + (36).to_bytes(4, "little") + b"WAVE" + b"fmt " + (16).to_bytes(4, "little")
              + (3).to_bytes(2, "little") + (0).to_bytes(2, "little") + (48000).to_bytes(4, "little")
              + (0).to_bytes(4, "little") + (0).to_bytes(2, "little") + (32).to_bytes(2, "little")
              + b"data" + (8).to_bytes(4, "little") + bytes(8))
    bad.write_bytes(header)
    try:
        read_iq(bad)
    except ValueError:
        return
    raise AssertionError("zero-channel WAV should raise ValueError, not ZeroDivisionError")


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Directory input
# ---------------------------------------------------------------------------


def _h(seconds: float) -> str:
    from robot36_decode.cli import _human_time
    return _human_time(seconds * 1000)


def _mae_of(png: Path, sig: dict) -> float:
    return float(np.abs(synth.read_png_rgb(png).astype(np.float64) - sig["expected_rgb"]).mean())


def _make_folder(sig: dict) -> Path:
    """a.wav + b.WAV valid, bad.wav corrupt, notes.txt, .hidden/, sub/c.wav."""
    d = _tmp() / "caps"
    (d / "sub").mkdir(parents=True)
    (d / ".hidden").mkdir()
    audio = synth.to_audio(sig)
    synth.write_wav_pcm16(d / "a.wav", audio, FS)
    synth.write_wav_pcm16(d / "b.WAV", audio, FS)
    synth.write_wav_pcm16(d / "sub" / "c.wav", audio, FS)
    synth.write_wav_pcm16(d / ".hidden" / "h.wav", audio, FS)
    (d / "bad.wav").write_bytes(b"this is not a wave file")
    (d / "notes.txt").write_text("not audio")
    return d


def _run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = main(list(argv))
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue(), err.getvalue()


def test_directory_input_default_outputs_and_filters():
    sig = _signal()
    d = _make_folder(sig)
    rc, out, err = _run_cli(str(d))
    assert rc == 0, (out, err)
    assert (d / "a.png").exists() and (d / "b.png").exists()
    assert not (d / "sub" / "c.png").exists(), "subfolders need --recursive"
    assert not (d / ".hidden" / "h.png").exists(), "hidden folders are ignored"
    assert not (d / "bad.png").exists() and "not a valid WAV file" in out
    assert "2 decoded, 0 failed, 1 skipped" in out, out
    mae = max(_mae_of(d / "a.png", sig), _mae_of(d / "b.png", sig))
    RESULTS.append(("directory input: decoded images", "ok" if mae <= 2.0 else "FAIL", mae, 2.0))
    assert mae <= 2.0


def test_directory_recursive_output_dir_mirrors_layout():
    sig = _signal()
    d = _make_folder(sig)
    out_dir = d.parent / "decoded"
    rc, out, err = _run_cli(str(d), "-r", "-o", str(out_dir), "--quiet")
    assert rc == 0, (out, err)
    assert {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*.png")} == {"a.png", "b.png", "sub/c.png"}
    assert not list(d.rglob("*.png")), "inputs folder must stay untouched when -o is given"


def test_directory_skip_existing_and_exit_status():
    sig = _signal()
    d = _make_folder(sig)
    assert _run_cli(str(d), "--quiet")[0] == 0
    stamp = (d / "a.png").stat().st_mtime_ns
    rc, out, _ = _run_cli(str(d), "--skip-existing")
    assert rc == 0 and "0 decoded, 0 failed" in out and (d / "a.png").stat().st_mtime_ns == stamp


def test_directory_one_failure_does_not_stop_the_rest():
    sig = _signal()
    d = _make_folder(sig)
    noise = np.random.default_rng(4).standard_normal(FS * 40) * 0.02
    synth.write_wav_pcm16(d / "aaa_noise.wav", noise, FS)  # sorts first, nothing to decode
    rc, out, err = _run_cli(str(d))
    assert rc == 1, "any failed file gives a non-zero status"
    assert "1 failed" in out and "FAILED   aaa_noise.wav" in out
    assert (d / "a.png").exists() and (d / "b.png").exists(), "later files still decode"


def test_directory_truncated_recording_is_rescued_as_partial_image():
    sig = _signal()
    d = _make_folder(sig)
    synth.write_wav_pcm16(d / "aaa_short.wav", synth.to_audio(sig)[: FS * 4], FS)  # ~16 lines
    rc, out, err = _run_cli(str(d))
    assert rc == 0, (out, err)
    assert "image lines are in the audio" in err, "the normal attempt still reports why it failed"
    assert "rescued  aaa_short.wav" in out and (d / "aaa_short_forced.png").exists()
    got = synth.read_png_rgb(d / "aaa_short_forced.png")
    assert (got[200:] == 128).all(), "lines beyond the recording are grey"
    # A rescued file counts as done for --skip-existing ...
    rc, out, _ = _run_cli(str(d), "--skip-existing")
    assert rc == 0 and "skipped  aaa_short.wav" in out, out
    # ... and without its rescued picture, --no-rescue reports the failure.
    (d / "aaa_short_forced.png").unlink()
    rc, out, _ = _run_cli(str(d), "--no-rescue", "--skip-existing")
    assert rc == 1 and "FAILED   aaa_short.wav" in out


def test_directory_name_collisions_and_iq_extensions():
    sig = _signal()
    d = _tmp() / "mix"
    d.mkdir()
    iq = synth.to_iq(sig)
    synth.write_iq_wav(d / "pass.wav", iq, FS, kind="float32")
    synth.write_raw_iq(d / "pass.iq", iq)
    rc, out, err = _run_cli(str(d), "--format", "iq", "--quiet")
    assert rc == 0, (out, err)
    assert (d / "pass_wav.png").exists() and (d / "pass_iq.png").exists()
    # With the default --format wav only .wav files are considered.
    d2 = _tmp() / "only_wav"
    d2.mkdir()
    synth.write_wav_pcm16(d2 / "x.wav", synth.to_audio(sig), FS)
    synth.write_raw_iq(d2 / "y.iq", iq)
    assert _run_cli(str(d2), "--quiet")[0] == 0 and (d2 / "x.png").exists() and not (d2 / "y.png").exists()


def test_directory_argument_errors():
    sig = _signal()
    d = _make_folder(sig)
    empty = _tmp() / "empty"
    empty.mkdir()
    rc, _, err = _run_cli(str(empty))
    assert rc == 1 and "no input files" in err
    afile = d.parent / "afile.png"
    afile.write_bytes(b"x")
    assert _run_cli(str(d), "-o", str(afile))[0] == 2
    assert _run_cli(str(d / "a.wav"), "-r")[0] == 2, "--recursive needs a directory"
    assert _run_cli(str(d), "--format", "fm", "--save-audio", str(d))[0] == 2, "would overwrite inputs"


# ---------------------------------------------------------------------------
# --force-start
# ---------------------------------------------------------------------------


def test_force_start_audio_matches_normal_decode():
    d, sig = _tmp(), _signal(lead_ms=3500.0, tail_ms=1500.0)
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    start_s = float(sig["line_starts_ms"][0]) / 1000.0
    rc, out, err = _decode(d / "a.wav", "--force-start", f"{start_s:.4f}")
    assert rc == 0, err
    assert "VIS and sync tracking skipped" in out and "VIS:" not in out and "Sync:   not tracked" in out
    mae = _mae_of(d / "a.png", sig)
    RESULTS.append(("--force-start audio (exact start)", "ok" if mae <= 2.0 else "FAIL", mae, 2.0))
    assert mae <= 2.0


def test_force_start_small_start_error_is_only_a_shift():
    d, sig = _tmp(), _signal(lead_ms=3500.0, tail_ms=1500.0)
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    start_s = float(sig["line_starts_ms"][0]) / 1000.0 + 0.0005  # 0.5 ms late (~2 pixels)
    assert _decode(d / "a.wav", "--force-start", f"{start_s:.4f}")[0] == 0
    mae = _mae_of(d / "a.png", sig)
    RESULTS.append(("--force-start audio (0.5 ms late)", "ok" if mae <= 12.0 else "FAIL", mae, 12.0))
    assert mae <= 12.0


def test_force_start_fm_capture():
    d, sig = _tmp(), _signal()
    iq = synth.to_fm_baseband(
        sig, dead_before_s=6.0, dead_after_s=6.0, carrier=synth.doppler_curve(6000, -30), inband_snr_db=14.0
    ).astype(np.complex64)
    synth.write_raw_iq(d / "c.raw", iq)
    start_s = 6.0 + float(sig["line_starts_ms"][0]) / 1000.0
    rc, out, err = _decode(d / "c.raw", "--format", "fm", "--raw-sample-rate", "78125", "--force-start", f"{start_s:.4f}")
    assert rc == 0, err
    assert f"Image:  starts at {_h(start_s)}" in out, out
    mae = _mae_of(d / "c.png", sig)
    RESULTS.append(("--force-start fm capture", "ok" if mae <= 14.0 else "FAIL", mae, 14.0))
    assert mae <= 14.0


def test_force_start_truncated_file_greys_missing_lines():
    d, sig = _tmp(), _signal()
    keep_s = float(sig["line_starts_ms"][0]) / 1000.0 + 120 * 0.150 + 0.050  # ~120 lines of audio
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig)[: int(keep_s * FS)], FS)
    start_s = float(sig["line_starts_ms"][0]) / 1000.0
    # v3.5: a normal decode of a recording that stops early keeps the lines it
    # has (it used to be refused), and so does a forced one.
    for flags in (("--no-vis", "--start-ms", f"{start_s * 1000:.1f}"), ("--force-start", f"{start_s:.4f}")):
        rc, out, err = _decode(d / "a.wav", *flags)
        assert rc == 0, err
        got = synth.read_png_rgb(d / "a.png")
        assert (got[121:] == 128).all(), "lines with no audio must be grey"
        top = float(np.abs(got[:100].astype(np.float64) - sig["expected_rgb"][:100]).mean())
        assert top <= 3.0, top
        assert "the recording ends after line 120 of 240" in out, out


def test_force_start_argument_errors_and_hint():
    d, sig = _tmp(), _signal()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    f = str(d / "a.wav")
    assert _run_cli(f, "--force-start", "-1")[0] == 2
    assert _run_cli(f, "--force-start", "1", "--start-ms", "1000")[0] == 2
    assert _run_cli(str(d), "--force-start", "1")[0] == 2, "single file only"
    rc, _, err = _run_cli(f, "--force-start", "9999")
    assert rc == 1 and "beyond the end" in err
    # The normal FM failure message points at --force-start.
    iq = synth.to_fm_baseband(sig, dead_before_s=5.0, dead_after_s=5.0, inband_snr_db=14.0).astype(np.complex64)
    synth.write_raw_iq(d / "short.raw", iq[: int(78125 * 9.0)])  # ~16 lines: too few to keep
    rc, _, err = _run_cli(str(d / "short.raw"), "--format", "fm", "--raw-sample-rate", "78125")
    assert rc == 1 and "--force-start" in err, err


# ---------------------------------------------------------------------------
# Trailing separators in path arguments
# ---------------------------------------------------------------------------


def test_path_argument_parsing_trailing_separators():
    from pathlib import Path as P
    from robot36_decode.cli import parse_path_argument as parse

    assert parse("caps/", windows=False) == (P("caps"), True)
    assert parse("caps//", windows=False) == (P("caps"), True)
    assert parse("caps", windows=False) == (P("caps"), False)
    assert parse("a\\b\\", windows=False)[1] is False, "backslash is an ordinary character on POSIX"
    assert parse("caps\\", windows=True) == (P("caps"), True)
    assert parse("caps/", windows=True) == (P("caps"), True)
    # cmd.exe turns "caps\" into  caps"  (the backslash escapes the quote)
    assert parse('caps"', windows=True) == (P("caps"), True)
    assert parse('my captures"', windows=True) == (P("my captures"), True)
    assert parse('caps"', windows=False)[0] == P('caps"'), "a quote is a legal POSIX file name character"
    assert parse("", windows=False)[0] == P(".")
    assert parse("/", windows=False) == (P("/"), True), "the root is left alone"
    assert parse("C:\\", windows=True)[1] is True and str(parse("C:\\", windows=True)[0]).endswith("\\"), "drive root not rewritten to C:"


def test_output_directory_for_single_file():
    sig = _signal()
    d = _tmp()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    f = str(d / "a.wav")
    # Trailing slash: a (new, nested) directory receiving <input name>.png
    assert _run_cli(f, "-o", f"{d}/new/sub/", "--quiet")[0] == 0
    assert (d / "new" / "sub" / "a.png").is_file()
    # Existing directory without a slash behaves the same
    assert _run_cli(f, "-o", str(d / "new"), "--quiet")[0] == 0
    assert (d / "new" / "a.png").is_file()
    # A plain file path is still a file path
    assert _run_cli(f, "-o", str(d / "plain.png"), "--quiet")[0] == 0
    assert (d / "plain.png").is_file()
    # Directory input with trailing slash on input and output, relative paths
    import os
    before = os.getcwd()
    try:
        os.chdir(d)
        (d / "caps").mkdir()
        synth.write_wav_pcm16(d / "caps" / "x.wav", synth.to_audio(sig), FS)
        assert _run_cli("caps/", "-o", "decoded/", "--quiet")[0] == 0
        assert (d / "decoded" / "x.png").is_file()
    finally:
        os.chdir(before)


def test_save_audio_directory_for_single_file():
    sig = _signal()
    d = _tmp()
    iq = synth.to_fm_baseband(sig, dead_before_s=5.0, dead_after_s=5.0, inband_snr_db=16.0).astype(np.complex64)
    synth.write_raw_iq(d / "c.raw", iq)
    rc, out, err = _run_cli(
        str(d / "c.raw"), "--format", "fm", "--raw-sample-rate", "78125",
        "-o", f"{d}/img/", "--save-audio", f"{d}/aud/", "--quiet",
    )
    assert rc == 0, err
    assert (d / "img" / "c.png").is_file() and (d / "aud" / "c.wav").is_file()


# ---------------------------------------------------------------------------
# --find-start, --force-start auto and directory rescue
# ---------------------------------------------------------------------------


def _reported_start(out: str) -> float:
    import re
    m = re.search(r"Best start:\s+([0-9.]+) s", out)
    assert m, out
    return float(m.group(1))


def test_find_start_function_without_vis_header():
    """Comb + parity + header template, with the VIS reader switched off."""
    from robot36_decode.findstart import find_start

    sig = _signal(lead_ms=3500.0, tail_ms=1500.0)
    true_start = float(sig["line_starts_ms"][0]) / 1000.0
    worst = 0.0
    for snr in (None, 14.0, 9.0):
        audio = synth.to_audio(sig, snr_db=snr) if snr else synth.to_audio(sig)
        est = find_start(audio, FS, 0.0, onset_s=3.5, vis=None)
        assert est.start_s is not None and est.confidence in ("HIGH", "MEDIUM"), (snr, est)
        worst = max(worst, abs(est.start_s - true_start) * 1000.0)
    RESULTS.append(("find-start (no VIS) worst error ms", "ok" if worst <= 1.5 else "FAIL", worst, 1.5))
    assert worst <= 1.5, worst


def test_find_start_cli_audio_and_fm():
    sig = _signal(lead_ms=3500.0, tail_ms=1500.0)
    true_start = float(sig["line_starts_ms"][0]) / 1000.0
    d = _tmp()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig, snr_db=14.0), FS)
    rc, out, err = _run_cli(str(d / "a.wav"), "--find-start")
    assert rc == 0 and "Confidence: HIGH" in out and abs(_reported_start(out) - true_start) < 0.002, (out, err)
    assert "--force-start" in out, "the report ends with a ready-to-run command"
    assert not (d / "a.png").exists(), "--find-start must not decode"
    iq = synth.to_fm_baseband(sig, dead_before_s=6.0, dead_after_s=6.0, carrier=synth.doppler_curve(6000, -30), inband_snr_db=14.0).astype(np.complex64)
    synth.write_raw_iq(d / "c.raw", iq)
    rc, out, err = _run_cli(str(d / "c.raw"), "--format", "fm", "--raw-sample-rate", "78125", "--find-start")
    assert rc == 0 and abs(_reported_start(out) - (6.0 + true_start)) < 0.002, (out, err)


def test_force_start_auto_decodes():
    sig = _signal(lead_ms=3500.0, tail_ms=1500.0)
    d = _tmp()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    rc, out, err = _decode(d / "a.wav", "--force-start", "auto")
    assert rc == 0, err
    assert "Finding the start" in out and "VIS and sync tracking skipped" in out
    mae = _mae_of(d / "a.png", sig)
    RESULTS.append(("--force-start auto (audio)", "ok" if mae <= 2.5 else "FAIL", mae, 2.5))
    assert mae <= 2.5


def test_find_start_nothing_found_and_argument_errors():
    d = _tmp()
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(48000 * 40) * 0.02)
    synth.write_wav_pcm16(d / "n.wav", noise, FS)
    rc, out, err = _run_cli(str(d / "n.wav"), "--find-start")
    assert rc == 1 and "Confidence: NONE" in out
    rc, _, err = _run_cli(str(d / "n.wav"), "--force-start", "auto")
    assert rc == 1 and "could not find the start" in err
    assert _run_cli(str(d / "n.wav"), "--find-start", "--force-start", "5")[0] == 2
    assert _run_cli(str(d / "n.wav"), "--force-start", "soon")[0] == 2
    assert _run_cli(str(d), "--force-start", "auto")[0] == 2, "directory mode rescues automatically instead"


def test_directory_rescue_of_failed_capture():
    sig = _signal()
    d = _tmp() / "caps"
    d.mkdir()
    good = synth.to_fm_baseband(sig, dead_before_s=5.0, dead_after_s=5.0, inband_snr_db=16.0).astype(np.complex64)
    synth.write_raw_iq(d / "a_good.raw", good)
    cut_s = 9.2
    synth.write_raw_iq(d / "b_cut.raw", good[: int(78125 * cut_s)])  # ~18 lines: too few for a normal decode
    flags = ("--format", "fm", "--raw-sample-rate", "78125")
    rc, out, err = _run_cli(str(d), *flags)
    assert rc == 0, (out, err)
    assert (d / "a_good.png").exists() and (d / "b_cut_forced.png").exists() and not (d / "b_cut.png").exists()
    assert "1 rescued with --force-start" in out and "rescued  b_cut.raw" in out, out
    got = synth.read_png_rgb(d / "b_cut_forced.png")
    start = 5.0 + float(sig["line_starts_ms"][0]) / 1000.0
    lines_with_audio = int((cut_s - start) / 0.150)
    top = float(np.abs(got[: lines_with_audio - 8].astype(np.float64) - sig["expected_rgb"][: lines_with_audio - 8]).mean())
    assert top <= 14.0, top
    assert (got[-5:] == 128).all(), "lines beyond the end of the file are grey"
    # --no-rescue keeps the old behaviour
    (d / "b_cut_forced.png").unlink()
    rc, out, err = _run_cli(str(d), *flags, "--no-rescue", "--skip-existing")
    assert rc == 1 and "FAILED   b_cut.raw" in out and not (d / "b_cut_forced.png").exists()


def test_find_start_directory_listing():
    sig = _signal(lead_ms=3500.0, tail_ms=1500.0)
    d = _tmp() / "caps"
    d.mkdir()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    synth.write_wav_pcm16(d / "b.wav", synth.to_audio(sig, snr_db=14.0), FS)
    rc, out, err = _run_cli(str(d), "--find-start")
    assert rc == 0 and out.count("HIGH") == 2 and "a.wav" in out and "b.wav" in out, (out, err)


# ---------------------------------------------------------------------------
# Very short captures and unexpected errors
# ---------------------------------------------------------------------------


def test_very_short_fm_capture_gives_a_clear_error():
    """A 200 kB stereo capture (0.64 s at 78125 Hz) used to crash the locator."""
    d = _tmp()
    rng = np.random.default_rng(2)
    worst = ""
    for seconds in (0.3, 0.64, 1.2):
        n = int(78125 * seconds)
        noise = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.01).astype(np.complex64)
        synth.write_raw_iq(d / "short.raw", noise)
        rc, out, err = _run_cli(str(d / "short.raw"), "--format", "fm", "--raw-sample-rate", "78125")
        assert rc == 1 and "the capture is only" in err and "Traceback" not in err, (seconds, err)
        worst = err
    assert "nothing to decode" in worst
    # The old crash point itself: the locator on a 6-frame capture returns None.
    from robot36_decode.fmdemod import locate_fm_signal
    noise = ((rng.standard_normal(50_000) + 1j * rng.standard_normal(50_000)) * 0.01).astype(np.complex64)
    assert locate_fm_signal(noise, 78125.0) is None
    for frames_seconds in (0.45, 0.63, 0.8, 1.0):
        n = int(78125 * frames_seconds)
        locate_fm_signal(noise[:n] if n <= noise.size else noise, 78125.0)  # must not raise
    assert _run_cli(str(d / "short.raw"), "--format", "fm", "--raw-sample-rate", "78125", "--find-start")[0] == 1


def test_one_bad_file_cannot_crash_a_batch():
    from robot36_decode import cli

    sig = _signal()
    d = _tmp() / "caps"
    d.mkdir()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    synth.write_wav_pcm16(d / "b.wav", synth.to_audio(sig), FS)
    real = cli.decode_robot36
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise IndexError("injected bug")
        return real(*args, **kwargs)

    cli.decode_robot36 = flaky
    try:
        rc, out, err = _run_cli(str(d), "--no-rescue")
    finally:
        cli.decode_robot36 = real
    assert rc == 1 and "internal error (IndexError: injected bug)" in err, err
    assert "FAILED   a.wav" in out and "ok       b.wav" in out, out
    assert (d / "b.png").exists(), "the next file is still processed"


def test_edge_columns_are_not_smeared_by_neighbouring_tones():
    """v3.4.2: the first/last columns used to pick up the sync pulse (green stripe)."""
    sig = synth.encode_robot36(synth.make_test_image())
    d = _tmp()
    p = d / "edge.wav"
    synth.write_wav_pcm16(p, synth.to_audio(sig), FS)
    rc, out, err = _decode(p, "--quiet")
    assert rc == 0, err
    got = synth.read_png_rgb(p.with_suffix(".png")).astype(np.float64)
    want = sig["expected_rgb"]
    col_mae = np.abs(got - want).mean(axis=(0, 2))
    assert col_mae[-2:].mean() < 7.0, f"right edge columns MAE {col_mae[-2:]}"
    assert col_mae[:2].mean() < 5.0, f"left edge columns MAE {col_mae[:2]}"
    # signed green bias at the right edge (the visible artefact) must be tiny
    bias = (got[:, -6:, 1] - want[:, -6:, 1]).mean(axis=0)
    assert np.abs(bias).max() < 4.0, f"green bias at the right edge: {bias}"
    RESULTS.append(("edge columns (last 2) MAE", "ok", float(col_mae[-2:].mean()), 7.0))


def test_neutral_grey_has_no_coloured_edge_stripe():
    """Neutral content must stay neutral right up to the last column."""
    grey = np.full((240, 320, 3), 90, dtype=np.uint8)
    grey[:, :, :] = np.linspace(40, 180, 240, dtype=np.uint8)[:, None, None]
    sig = synth.encode_robot36(grey)
    d = _tmp()
    p = d / "grey.wav"
    synth.write_wav_pcm16(p, synth.to_audio(sig, snr_db=25), FS)
    rc, out, err = _decode(p, "--quiet")
    assert rc == 0, err
    got = synth.read_png_rgb(p.with_suffix(".png")).astype(np.float64)
    tint = got[:, -4:, 1] - 0.5 * (got[:, -4:, 0] + got[:, -4:, 2])
    inner = got[:, 100:200, 1] - 0.5 * (got[:, 100:200, 0] + got[:, 100:200, 2])
    assert abs(tint.mean() - inner.mean()) < 4.0, (tint.mean(), inner.mean())


def test_segment_frequencies_tail_guard_avoids_a_smeared_step():
    from robot36_decode import dsp

    fs = 48000
    t = np.arange(int(0.02 * fs)) / fs
    # 1900 Hz tone that slides down to 1200 Hz around 10 ms (0.6 ms wide smear)
    f = 1900.0 - 700.0 / (1.0 + np.exp(-(t - 0.0100) / 0.00015))
    track = 2 * np.pi * np.cumsum(f) / fs
    pix, n = 0.275, 12  # 12 pixels end at 3.3 ms + start
    start = 10.0 - n * pix + 0.1  # last pixel ends 0.1 ms after the step centre
    plain = dsp.segment_frequencies(track, fs, start, pix, n)
    guarded = dsp.segment_frequencies(track, fs, start, pix, n, tail_guard_ms=0.8)
    assert plain[-1] < 1750.0, plain[-1]
    assert abs(guarded[-1] - 1900.0) < 25.0, guarded[-1]
    assert np.allclose(plain[:3], guarded[:3])


def test_filters_match_scipy_signal():
    """v3.4.3: the decoder's own FIR/convolution/resampling equal scipy.signal's."""
    import scipy.signal as ss

    from robot36_decode import filters

    rng = np.random.default_rng(3)
    for args, kw in (
        ((301, 7000.0), dict(fs=78125.0, window=("kaiser", 6.0))),
        ((201, 2400.0), dict(fs=78125.0)),
        ((201, [1000.0, 2500.0]), dict(fs=78125.0, pass_zero=False)),
    ):
        assert np.allclose(filters.firwin(*args, **kw), ss.firwin(*args, **kw), atol=1e-15)
    x = rng.standard_normal(50_001)
    xc = (x + 1j * rng.standard_normal(x.size)).astype(np.complex64)
    for m in (1, 2, 201, 301):
        h = rng.standard_normal(m)
        assert np.allclose(filters.convolve_same(x, h), np.convolve(x, h, mode="same"), atol=1e-9)
    h32 = ss.firwin(301, 0.2).astype(np.float32)
    ref = ss.oaconvolve(xc, h32, mode="same")
    got = filters.convolve_same(xc, h32)
    assert got.dtype == np.complex64 and np.abs(got - ref).max() < 1e-5
    for data in (x, x + 1j * x[::-1]):
        for up, down in ((384, 625), (48000, 44100), (1, 2), (3, 1), (160, 147)):
            want = ss.resample_poly(data, up, down)
            have = filters.resample_poly(data, up, down)
            assert have.shape == want.shape and np.abs(have - want).max() < 1e-12, (up, down)
    c = rng.standard_normal(997)
    assert np.array_equal(filters.medfilt(c, 9), ss.medfilt(c, 9))


def test_tone_energies_fast_path_matches_reference():
    from robot36_decode.vis import _tone_energies

    rng = np.random.default_rng(4)
    x = rng.standard_normal(100_000)
    tones = [1100.0, 1200.0, 1900.0]
    for data, rate, frame, window in ((x, 48000, 240, 960), (x + 0.5j * x, 48000, 240, 960), (x, 44100, 220, 882)):
        got = _tone_energies(data, rate, tones, frame, window)
        n = np.arange(data.size)
        frames = (data.size - window) // frame
        for k, f in enumerate(tones):
            z = np.concatenate(([0.0], np.cumsum(data * np.exp(-2j * np.pi * f * n / rate))))
            starts = np.arange(frames) * frame
            ref = np.abs(z[starts + window] - z[starts]) ** 2
            assert np.allclose(got[k], ref, rtol=1e-8, atol=1e-6 * ref.max())


def test_normal_decode_does_not_import_scipy_signal():
    """scipy.signal costs ~1 s to import; the decoder no longer needs it."""
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r); import robot36_decode; "
        "print('scipy.signal' in sys.modules, 'scipy.ndimage' in sys.modules)"
    ) % str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"], out


def test_late_vis_in_long_audio_recording():
    """v3.4.3: a header more than 15 s into an audio file is found (was: start 6-11 lines off)."""
    sig = _signal(lead_ms=40000.0)
    d = _tmp()
    p = d / "late.wav"
    synth.write_wav_pcm16(p, synth.to_audio(sig, snr_db=15), FS)
    rc, out, err = _decode(p)
    assert rc == 0, err
    assert "VIS:    code 0x08" in out and "Start:  40.910 s" in out, out
    mae = _mae_of(p.with_suffix(".png"), sig)
    RESULTS.append(("audio, VIS 40 s into the file", "ok" if mae <= 14.0 else "FAIL", mae, 14.0))
    assert mae <= 14.0, mae


def test_stereo_sdr_wav_read_as_audio_gets_a_format_hint():
    sig = _signal()
    d = _tmp()
    iq = synth.to_fm_baseband(sig, dead_before_s=2.0, dead_after_s=2.0)
    synth.write_iq_wav(d / "sdr.wav", iq, 78125, kind="pcm16")
    rc, out, err = _run_cli(str(d / "sdr.wav"), "--quiet")
    # Read as audio it fails or yields noise; either way the hint is shown, even with --quiet.
    assert "--format fm" in err and "78125 Hz" in err, (rc, err)
    # the same file with the right format decodes
    rc, out, err = _run_cli(str(d / "sdr.wav"), "--format", "fm", "--quiet")
    assert rc == 0, err


def test_single_file_option_checks():
    sig = _signal()
    d = _tmp()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    assert _run_cli(str(d / "a.wav"), "--save-audio", str(d / "x.wav"))[0] == 2, "needs --format fm"
    assert _run_cli(str(d / "a.wav"), "--jobs", "2")[0] == 2, "directory only"
    assert _run_cli(str(d), "--jobs", "-1")[0] == 2


def test_rf64_ds64_size_and_trailing_chunk():
    """The RF64 data size comes from ds64; a chunk after the data is not read as samples."""
    import struct

    iq = (np.exp(2j * np.pi * 0.01 * np.arange(4000))).astype(np.complex64)
    pcm = np.empty(iq.size * 2, dtype="<i2")
    pcm[0::2] = np.round(iq.real * 20000)
    pcm[1::2] = np.round(iq.imag * 20000)
    data = pcm.tobytes()
    fmt = struct.pack("<HHIIHH", 1, 2, 78125, 78125 * 4, 4, 16)
    trailer = b"LIST" + struct.pack("<I", 8) + b"INFOjunk"
    ds64 = struct.pack("<QQQI", 0, len(data), iq.size, 0)
    body = (
        b"WAVE" + b"ds64" + struct.pack("<I", len(ds64)) + ds64
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", 0xFFFFFFFF) + data + trailer
    )
    path = _tmp() / "rf64.wav"
    path.write_bytes(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + body)
    got = read_iq(path, target_rate=None)
    assert got.samples.size == iq.size and got.samples.dtype == np.complex64
    assert np.allclose(got.samples * 32768.0 / 20000.0, iq, atol=1e-3)


def test_directory_parallel_jobs_match_serial():
    sig = _signal()
    d = _make_folder(sig)
    serial, parallel = _tmp() / "serial", _tmp() / "parallel"
    rc1, out1, err1 = _run_cli(str(d), "-r", "-o", str(serial))
    rc2, out2, err2 = _run_cli(str(d), "-r", "-o", str(parallel), "--jobs", "2")
    assert rc1 == rc2 == 0, (out1, out2)  # bad.wav (not a WAV file) is skipped, not failed
    names = sorted(p.relative_to(serial).as_posix() for p in serial.rglob("*.png"))
    assert names == ["a.png", "b.png", "sub/c.png"], names
    for name in names:
        assert np.array_equal(synth.read_png_rgb(serial / name), synth.read_png_rgb(parallel / name)), name
    assert "2 at a time" in out2 and "Summary: 3 decoded" in out2, out2


def test_skip_existing_counts_rescued_output():
    sig = _signal()
    d = _tmp() / "caps"
    d.mkdir()
    synth.write_wav_pcm16(d / "a.wav", synth.to_audio(sig), FS)
    (d / "a_forced.png").write_bytes(b"placeholder")
    rc, out, err = _run_cli(str(d), "--skip-existing")
    assert rc == 0 and "skipped  a.wav" in out and not (d / "a.png").exists(), out


def test_fm_capture_trimmed_close_to_the_signal():
    """v3.4.3: 1 s of dead air either side used to locate only half the signal."""
    sig = _signal()
    d = _tmp()
    iq = synth.to_fm_baseband(sig, dead_before_s=1.0, dead_after_s=1.0).astype(np.complex64)
    synth.write_raw_iq(d / "trimmed.raw", iq)
    rc, out, err = _run_cli(str(d / "trimmed.raw"), "--format", "fm", "--raw-sample-rate", "78125")
    assert rc == 0, err
    mae = _mae_of(d / "trimmed.png", sig)
    RESULTS.append(("fm capture, 1 s dead air", "ok" if mae <= 8.0 else "FAIL", mae, 8.0))
    assert mae <= 8.0, mae


# ---------------------------------------------------------------------------
# v3.5.0: traps from a large set of real captures
# ---------------------------------------------------------------------------

FM_FS = 78125


def _fm_capture(sig: dict, **kw) -> np.ndarray:
    kw.setdefault("dead_before_s", 5.0)
    kw.setdefault("dead_after_s", 5.0)
    return synth.to_fm_baseband(sig, **kw).astype(np.complex64)


def _picture_rows(png: Path) -> tuple[np.ndarray, np.ndarray]:
    got = synth.read_png_rgb(png).astype(np.float64)
    return got, ~(got == 128).all(axis=(1, 2))


def test_fm_recording_that_starts_mid_image():
    """The recording begins 10 s into the picture: the missing top is grey and
    the rest sits where it belongs (v3.4: 'image start (-11.5 s) is beyond the end')."""
    sig = _signal(tail_ms=1000.0)
    # The ISS keeps transmitting for about a second after the last line (0.9-1.8 s
    # in real passes); give the synthetic tail a tone so it is signal, not silence.
    tailed = dict(sig)
    tailed["freq"], tailed["active"] = sig["freq"].copy(), sig["active"].copy()
    end = int(round((sig["image_start_ms"] + 240 * 150.0) * FS / 1000.0))
    tailed["freq"][end : end + FS] = 1900.0
    tailed["active"][end : end + FS] = True
    img0 = sig["image_start_ms"] / 1000.0
    iq = _fm_capture(tailed)[int((5.0 + img0 + 10.0) * FM_FS) :]
    d = _tmp()
    synth.write_raw_iq(d / "late.raw", iq)
    rc, out, err = _run_cli(str(d / "late.raw"), "--format", "fm", "--raw-sample-rate", "78125")
    assert rc == 0, err
    got, rows = _picture_rows(d / "late.png")
    first = int(np.flatnonzero(rows)[0])
    assert 65 <= first <= 69, first  # 10 s = 66.7 lines
    mae = float(np.abs(got[rows] - sig["expected_rgb"][rows]).mean())
    RESULTS.append(("fm, recording starts 10 s into the image", "ok" if mae <= 9.0 else "FAIL", mae, 9.0))
    assert mae <= 9.0, mae
    assert "the recording starts" in out, out
    # the separator count covers only the lines that were recorded
    import re

    m = re.search(r"(\d+) of (\d+) line separators", out)
    assert m is None or int(m.group(2)) < 240, out


def _garbled_header(sig: dict, seed: int = 5) -> dict:
    """Leader intact, start bit / data bits / stop bit replaced by picture tones."""
    g = dict(sig)
    g["freq"] = sig["freq"].copy()
    rng = np.random.default_rng(seed)
    fs = sig["fs"]
    start = int(round((sig["image_start_ms"] - 300.0) * fs / 1000.0))
    bit = int(round(30.0 * fs / 1000.0))
    for k in range(10):
        g["freq"][start + k * bit : start + (k + 1) * bit] = rng.choice([1500.0, 1700.0, 2100.0, 2300.0])
    return g


def test_picture_tones_are_not_a_vis_header():
    """v3.4 read a garbled header as 'mode 0x00' and refused the file."""
    from robot36_decode.vis import detect_vis

    g = _garbled_header(_signal())
    info = detect_vis(synth.to_audio(g, snr_db=20), FS)
    assert info is None or info.code == 0x08, info


def test_fm_picture_that_fades_out_without_readable_header():
    """No readable header and the carrier is lost 12 s into the picture: the
    lines are numbered from the start of the pulse train, not from the end."""
    sig = _signal()
    img0 = sig["image_start_ms"] / 1000.0
    iq = _fm_capture(_garbled_header(sig))
    cut = int((5.0 + img0 + 12.0) * FM_FS)
    rng = np.random.default_rng(9)
    sigma = float(np.std(iq[: 4 * FM_FS]))
    noise = ((rng.standard_normal(30 * FM_FS) + 1j * rng.standard_normal(30 * FM_FS)) * sigma / np.sqrt(2)).astype(np.complex64)
    d = _tmp()
    synth.write_raw_iq(d / "fade.raw", np.concatenate((iq[:cut], noise)))
    rc, out, err = _run_cli(str(d / "fade.raw"), "--format", "fm", "--raw-sample-rate", "78125")
    assert rc == 0, err
    assert "start of the sync pulse train" in out, out
    got, rows = _picture_rows(d / "fade.png")
    assert rows[:70].all() and not rows[90:].any(), np.flatnonzero(rows)
    mae = float(np.abs(got[:70] - sig["expected_rgb"][:70]).mean())
    RESULTS.append(("fm, header unreadable, carrier lost at line 80", "ok" if mae <= 9.0 else "FAIL", mae, 9.0))
    assert mae <= 9.0, mae


def test_fm_dropouts_before_and_inside_the_signal():
    """Receiver dropouts (near-silence) neither look like signal nor split the
    transmission; the lines inside one are grey."""
    sig = _signal()
    img0 = sig["image_start_ms"] / 1000.0
    iq = _fm_capture(sig, dead_before_s=15.0)
    iq[2 * FM_FS : 10 * FM_FS] *= 1e-3  # in the dead air
    d0 = int((15.0 + img0 + 15.0) * FM_FS)
    iq[d0 : d0 + 6 * FM_FS] *= 1e-3  # lines 100-140
    d = _tmp()
    synth.write_raw_iq(d / "drop.raw", iq)
    rc, out, err = _run_cli(str(d / "drop.raw"), "--format", "fm", "--raw-sample-rate", "78125")
    assert rc == 0, err
    assert "VIS:    code 0x08" in out and "receiver dropout" in out, out
    got, rows = _picture_rows(d / "drop.png")
    grey = np.flatnonzero(~rows)
    assert 38 <= grey.size <= 43 and 98 <= grey[0] <= 101, grey
    mae = float(np.abs(got[rows] - sig["expected_rgb"][rows]).mean())
    RESULTS.append(("fm, two receiver dropouts", "ok" if mae <= 8.0 else "FAIL", mae, 8.0))
    assert mae <= 8.0, mae


def test_fm_carrier_without_robot36_is_refused():
    """An FM carrier with other audio (voice-like tones) must not become a
    'picture' of noise."""
    n = 25 * FS
    rng = np.random.default_rng(3)
    tones = np.repeat(rng.uniform(300.0, 3000.0, n // 4800 + 1), 4800)[:n]
    fake = {"freq": tones, "active": np.ones(n, dtype=bool), "fs": FS}
    d = _tmp()
    synth.write_raw_iq(d / "voice.raw", _fm_capture(fake, dead_before_s=8.0, dead_after_s=8.0))
    rc, out, err = _run_cli(str(d / "voice.raw"), "--format", "fm", "--raw-sample-rate", "78125")
    assert rc == 1 and not (d / "voice.png").exists(), (out, err)
    assert "no Robot 36 picture" in err or "image lines are in the audio" in err, err


def test_headerless_16bit_capture():
    """A WAV whose header was overwritten with 0xFF bytes: skipped with a hint,
    decoded with --raw-format s16."""
    sig = _signal()
    iq = _fm_capture(sig)
    pcm = np.empty(iq.size * 2, dtype="<i2")
    scale = 0.9 * 32767 / float(np.abs(iq).max())
    pcm[0::2] = np.round(iq.real * scale)
    pcm[1::2] = np.round(iq.imag * scale)
    raw = bytearray(pcm.tobytes())
    raw[:65536] = b"\xff" * 65536
    d = _tmp() / "caps"
    d.mkdir()
    (d / "damaged.wav").write_bytes(bytes(raw))
    rc, out, err = _run_cli(str(d), "--format", "fm")
    assert "skipped  damaged.wav" in out and "--raw-format s16" in out, out
    rc, out, err = _run_cli(str(d), "--format", "fm", "--raw-format", "s16", "--raw-sample-rate", "78125")
    assert rc == 0 and (d / "damaged.png").exists(), (out, err)
    mae = _mae_of(d / "damaged.png", sig)
    RESULTS.append(("fm, WAV header destroyed, --raw-format s16", "ok" if mae <= 8.0 else "FAIL", mae, 8.0))
    assert mae <= 8.0, mae
    assert _run_cli(str(d / "damaged.wav"), "--raw-format", "s16")[0] == 2, "needs --format iq/fm"


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {exc}")
            if not isinstance(exc, AssertionError):
                traceback.print_exc()
    print("\nmetric table")
    print(f"{'case':58s} {'result':6s} {'value':>8s} {'limit':>7s}")
    for case, res, val, lim in RESULTS:
        print(f"{case:58s} {res:6s} {val:8.2f} {lim:7.2f}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0



# ---------------------------------------------------------------------------
# FM baseband captures (raw SDR recordings)
# ---------------------------------------------------------------------------


def _fm_file(d: Path, sig: dict, name: str = "fm.wav", fs_out: int = 78125, **kw) -> Path:
    iq = synth.to_fm_baseband(sig, fs_out=fs_out, **kw)
    synth.write_iq_wav(d / name, iq, fs_out, kind="pcm16")
    return d / name


def test_fm_locate_and_doppler_accuracy():
    from robot36_decode.fmdemod import demodulate_fm, locate_fm_signal

    sig = _signal()
    curve = synth.doppler_curve(9000, -70)
    iq = synth.to_fm_baseband(sig, carrier=curve, dead_before_s=20.0, dead_after_s=25.0)
    fs = 78125
    loc = locate_fm_signal(iq.astype(np.complex64), fs)
    assert loc is not None
    true_start, true_end = 20.0, 20.0 + sig["freq"].size / FS
    assert abs(loc.start_s - true_start) < 2.0, loc.start_s
    assert abs(loc.end_s - true_end) < 2.0, loc.end_s
    fm = demodulate_fm(iq.astype(np.complex64), fs, loc)
    err = np.abs(fm.doppler_hz - curve(fm.doppler_t_s))
    assert err.max() < 120.0, f"Doppler error {err.max():.0f} Hz"
    RESULTS.append(("FM Doppler curve error (max, Hz)", "ok", float(err.max()), 120.0))
    RESULTS.append(("FM located start error (s)", "ok", abs(loc.start_s - true_start), 2.0))


def test_fm_clean_positive_doppler():
    d, sig = _tmp(), _signal()
    f = _fm_file(d, sig, carrier=synth.doppler_curve(9000, -70), inband_snr_db=20.0)
    _check("fm clean, +9 kHz, -70 Hz/s, 20 s dead air", f, sig, 8.0, "--format", "fm")


def test_fm_negative_doppler_noise_and_burst():
    d, sig = _tmp(), _signal()
    f = _fm_file(d, sig, carrier=synth.doppler_curve(-1000, -110), inband_snr_db=13.0,
                 burst=(8.0, 3.0, 25.0))
    _check("fm -1 kHz, -110 Hz/s, 13 dB, 25 dB burst", f, sig, 14.0, "--format", "fm")


def test_fm_other_sample_rate():
    d, sig = _tmp(), _signal()
    f = _fm_file(d, sig, fs_out=156250, dead_before_s=5.0, dead_after_s=5.0,
                 carrier=synth.doppler_curve(12000, -60), inband_snr_db=18.0)
    _check("fm 156.25 kHz capture", f, sig, 8.0, "--format", "fm")


def test_fm_weak_signal_still_found():
    """About 4 dB in-band SNR: far below FM threshold, so the image is noisy,
    but the header must still be found and the image must be placed correctly."""
    d, sig = _tmp(), _signal()
    f = _fm_file(d, sig, carrier=synth.doppler_curve(6000, -30), inband_snr_db=5.0)
    rc, out, err = _decode(f, "--format", "fm")
    assert rc == 0, err
    assert "VIS:    code 0x08" in out, out
    got = synth.read_png_rgb(f.with_suffix(".png")).astype(np.float64)
    mae = float(np.abs(got - sig["expected_rgb"]).mean())
    RESULTS.append(("fm weak (5 dB in-band), image MAE", "ok" if mae < 60 else "FAIL", mae, 60.0))
    assert mae < 60.0, mae


def test_vis_detector_finds_weak_header_but_not_noise():
    from robot36_decode.vis import detect_vis

    sig = _signal(lead_ms=1000.0)
    audio = synth.to_audio(sig, snr_db=-6.0, seed=3)  # per-sample SNR far below the 14 dB cases
    info = detect_vis(audio.astype(np.float64), FS)
    assert info is not None and info.code == 0x08
    assert abs(info.image_start_ms - sig["image_start_ms"]) < 40.0, info.image_start_ms
    rng = np.random.default_rng(0)
    assert detect_vis(rng.standard_normal(FS * 15), FS) is None


def test_vis_polarity_is_logic1_1100hz():
    """Real VIS: 1100 Hz = 1, 1300 Hz = 0 (the module originally had these swapped)."""
    from robot36_decode.constants import FREQ_VIS_BIT_0, FREQ_VIS_BIT_1

    assert FREQ_VIS_BIT_1 == 1100.0 and FREQ_VIS_BIT_0 == 1300.0


def test_real_captures_if_present():
    """Decode real SDR# captures when available (set R36_CAPTURE_DIR)."""
    import os

    folder = Path(os.environ.get("R36_CAPTURE_DIR", "/mnt/user-data/uploads"))
    files = sorted(folder.glob("data*Hz.wav")) if folder.is_dir() else []
    if not files:
        print("      (no real captures found; skipped)")
        return
    for f in files:
        out_png = Path(tempfile.mkdtemp(prefix="r36_real_", dir=_TMP_ROOT)) / (f.stem + ".png")
        t0 = time.time()
        rc, out, err = _decode(f, "--format", "fm", "-o", str(out_png))
        dt = time.time() - t0
        assert rc == 0, f"{f.name}: {err}"
        assert "VIS:    code 0x08" in out, f"{f.name}: header not found\n{out}"
        tracked = int(out.split("Sync:")[1].split("/")[0])
        assert tracked >= 100, f"{f.name}: only {tracked} lines tracked"
        assert dt < 20.0, f"{f.name}: too slow ({dt:.1f}s)"
        RESULTS.append((f"real {f.name[:12]}: lines tracked / seconds", "ok", float(tracked), dt))


def test_repair_rows_fixes_burst_lines_but_keeps_edges():
    from robot36_decode.robot36 import _repair_rows

    rng = np.random.default_rng(0)
    smooth = np.tile(np.linspace(40, 200, 320), (60, 1)) + rng.normal(0, 3, (60, 320))
    damaged = smooth.copy()
    damaged[20] = rng.uniform(0, 255, 320)  # one wrecked line between two good ones
    damaged[40:] += 90.0  # a genuine horizontal edge: neighbours differ, must stay
    reference = damaged.copy()
    count = _repair_rows(damaged, 1)
    assert count == 1
    assert np.abs(damaged[20] - smooth[20]).mean() < 8.0
    assert np.array_equal(damaged[39:42], reference[39:42])


# ---------------------------------------------------------------------------
# Acceptance: real captures against quality limits and a reference decode
# ---------------------------------------------------------------------------

# Limits sit ~15-25% above what v3 measures, and well below v2 (which had
# 39-213 damaged rows and luma noise 56-155). The weak capture is limited by
# SNR, so it gets its own, looser bounds.
CAPTURE_LIMITS = {
    # name prefix: (max bad rows, max luma noise, max chroma noise, min lines tracked)
    "data03-33-54": (25, 22.0, 8.0, 225),
    "data11-13-57": (40, 24.0, 9.0, 225),
    "data16-58-07": (30, 27.0, 9.0, 220),
    "data03-31-46": (200, 90.0, 70.0, 130),  # ~1 dB above the noise floor
    "1794236581785": (3, 23.0, 9.0, 225),
}


def _capture_dir() -> Path:
    import os

    return Path(os.environ.get("R36_CAPTURE_DIR", "/mnt/user-data/uploads"))


def _real_files() -> list[Path]:
    folder = _capture_dir()
    if not folder.is_dir():
        return []
    return sorted(folder.glob("data*Hz.wav")) + sorted(folder.glob("*IQData.wav"))


def test_real_capture_quality_limits():
    from imgmetrics import quality

    files = _real_files()
    if not files:
        print("      (no real captures found; skipped)")
        return
    for f in files:
        key = next((k for k in CAPTURE_LIMITS if f.name.startswith(k)), None)
        if key is None:
            continue
        bad, ny, nc, lines = CAPTURE_LIMITS[key]
        png = Path(tempfile.mkdtemp(prefix="r36_q_", dir=_TMP_ROOT)) / (f.stem + ".png")
        rc, out, err = _decode(f, "--format", "fm", "-o", str(png))
        assert rc == 0, f"{f.name}: {err}"
        q = quality(synth.read_png_rgb(png))
        tracked = int(out.split("Sync:")[1].split("/")[0])
        RESULTS.append((f"quality {key[:12]}: bad rows / luma noise", "ok", float(q["bad_rows"]), float(bad)))
        assert q["bad_rows"] <= bad, f"{f.name}: {q['bad_rows']} damaged rows > {bad}"
        assert q["noise_y"] <= ny, f"{f.name}: luma noise {q['noise_y']:.1f} > {ny}"
        assert q["noise_c"] <= nc, f"{f.name}: chroma noise {q['noise_c']:.1f} > {nc}"
        assert tracked >= lines, f"{f.name}: only {tracked} lines tracked"


def test_fm_audio_band_limits_noise():
    """Output audio must be confined to the SSTV tone range."""
    from robot36_decode.fmdemod import AUDIO_BAND_HZ, demodulate_fm, locate_fm_signal

    sig = _signal()
    iq = synth.to_fm_baseband(sig, carrier=synth.doppler_curve(5000, -40), inband_snr_db=14.0).astype(np.complex64)
    loc = locate_fm_signal(iq, 78125)
    on = demodulate_fm(iq, 78125, loc)
    off = demodulate_fm(iq, 78125, loc, audio_band_hz=None)
    n = min(on.audio.size, off.audio.size)

    def out_of_band(a):
        spec = np.abs(np.fft.rfft(a[:n])) ** 2
        f = np.fft.rfftfreq(n, 1 / 48000)
        return spec[(f > AUDIO_BAND_HZ[1] + 300) & (f < 8000)].sum() / spec.sum()

    assert out_of_band(on.audio) < 0.05 * out_of_band(off.audio)


def test_fm_annealed_schedule_beats_fixed():
    """At ~5 dB in-band SNR the annealed feedback schedule must decode with
    clearly lower error than the old fixed 4 x 2400 Hz passes."""
    from robot36_decode import fmdemod
    from robot36_decode.fmdemod import demodulate_fm, locate_fm_signal

    d, sig = _tmp(), _signal()
    iq = synth.to_fm_baseband(sig, carrier=synth.doppler_curve(6000, -30), inband_snr_db=5.0).astype(np.complex64)
    loc = locate_fm_signal(iq, 78125)
    fixed = ((2400.0, 2400.0),) * 4
    maes = {}
    for label, sched in (("annealed", fmdemod.FEEDBACK_SCHEDULE), ("fixed", fixed)):
        fm = demodulate_fm(iq, 78125, loc, feedback_schedule=sched)
        synth.write_wav_pcm16(d / f"{label}.wav", fm.audio, 48000)
        png = d / f"{label}.png"
        rc, out, err = _decode(d / f"{label}.wav")
        assert rc == 0, err
        got = synth.read_png_rgb(png).astype(np.float64)
        maes[label] = float(np.abs(got - sig["expected_rgb"]).mean())
    RESULTS.append(("FM 5 dB annealed MAE", "ok", maes["annealed"], maes["fixed"] * 0.85))
    assert maes["annealed"] < maes["fixed"] * 0.85, maes


def test_fm_feedback_schedule_disabled():
    from robot36_decode.fmdemod import demodulate_fm, locate_fm_signal

    sig = _signal()
    iq = synth.to_fm_baseband(sig, carrier=synth.doppler_curve(5000, -40), inband_snr_db=20.0).astype(np.complex64)
    loc = locate_fm_signal(iq, 78125)
    assert demodulate_fm(iq, 78125, loc, feedback_schedule=None).audio.size > 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
