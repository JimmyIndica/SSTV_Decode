"""Command-line interface for the Robot 36 decoder."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from .constants import (
    FREQ_BLACK,
    FREQ_WHITE,
    IMAGE_MS,
    LINE_MS,
    LINES,
    ROBOT36_VIS_CODE,
)
from .dsp import locate_signal, phase
from .findstart import StartEstimate, find_start, format_report
from .image import write_png
from .robot36 import decode_robot36
from .synctrack import SyncTrack, track_line_starts
from .vis import VIS_SEARCH_MS, VisInfo, detect_vis
from .wav import WavData, read_wav
from .iq import IQData, read_iq
from .fmdemod import (
    DEFAULT_EXPECTED_S,
    FmAudio,
    demodulate_fm,
    locate_fm_signal,
)

__version__ = "3.5.0"

# A sync-tone offset larger than this (Hz) triggers a second pass so VIS
# detection and band-limiting follow the shifted signal.
OFFSET_RETRY_HZ = 40.0

# Extra analysis-band margin (Hz) around the measured Doppler sweep.
BAND_MARGIN_HZ = 100.0

# FM captures: the carrier is typically still present for this long after the
# last image line (end-of-transmission tones), used only as a coarse hint.
FM_TAIL_MS = 1500.0


def write_audio_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """Write mono float audio as a 16-bit PCM WAV."""
    import wave

    pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def parse_path_argument(text: str, windows: bool | None = None) -> tuple[Path, bool]:
    """Turn a command-line path into a ``Path`` and say whether it was written
    as a directory (trailing separator).

    ``Path`` drops a trailing ``/``, so the hint must be read from the text.
    On Windows, ``"folder\\"`` in cmd.exe is split by the C runtime as
    ``folder"`` (the backslash escapes the quote); trailing quotes are removed
    and treated as a directory hint, since a quote cannot be part of a Windows
    file name.
    """
    if windows is None:
        windows = os.name == "nt"
    directory = False
    if windows and text.endswith('"'):
        text = text.rstrip('"')
        directory = True
    separators = ("/", "\\") if windows else ("/",)
    if text.endswith(separators):
        directory = True
        stripped = text.rstrip("".join(separators))
        # Keep a bare root ("/", "C:\\") intact; stripping it would change its meaning.
        if stripped and not stripped.endswith(":"):
            text = stripped
    return Path(text or "."), directory


def _input_path(text: str) -> Path:
    return parse_path_argument(text)[0]


class _PathAction(argparse.Action):
    """Store a path and remember whether it was written with a trailing separator."""

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: ANN001
        path, directory = parse_path_argument(values)
        setattr(namespace, self.dest, path)
        setattr(namespace, f"{self.dest}_dir_hint", directory)


def _resolve_single_target(
    given: Path | None, directory_hint: bool, input_path: Path, suffix: str
) -> Path | None:
    """Output path for a single input: ``given`` itself, or ``<given>/<input
    name><suffix>`` when ``given`` is (or is written as) a directory."""
    if given is None:
        return None
    if directory_hint or given.is_dir():
        given.mkdir(parents=True, exist_ok=True)
        return given / input_path.with_suffix(suffix).name
    return given


# An FM capture shorter than this cannot hold even the VIS header and a few lines.
MIN_CAPTURE_S = 2.0


def _check_capture_length(samples: np.ndarray, sample_rate: float) -> None:
    seconds = samples.size / sample_rate
    if seconds < MIN_CAPTURE_S:
        raise ValueError(
            f"the capture is only {seconds:.2f} s long ({samples.size} samples at "
            f"{sample_rate:g} Hz); a Robot 36 transmission takes about 38 s, so there "
            "is nothing to decode (was the recording stopped early?)"
        )


def _describe_error(exc: Exception) -> str:
    """Message for an error that stops one file.

    Expected failures (unreadable file, no signal, ...) are ``OSError`` or
    ``ValueError`` and are reported as they are. Anything else is a bug: it is
    reported as an internal error so that a batch run can continue with the
    next file instead of dying with a traceback. Set ``R36_DEBUG=1`` to get
    the traceback.
    """
    if isinstance(exc, (OSError, ValueError)):
        return str(exc)
    if os.environ.get("R36_DEBUG"):
        raise exc
    return (
        f"internal error ({type(exc).__name__}: {exc}); this is a bug, please report "
        "the file. Set R36_DEBUG=1 for a traceback"
    )


def _force_start_value(text: str) -> float | str:
    """argparse type for --force-start: seconds, or the word ``auto``."""
    if text.strip().lower() == "auto":
        return "auto"
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid value {text!r}: expected SECONDS or 'auto'") from None


def _estimate_start(args: argparse.Namespace, input_path: Path) -> StartEstimate:
    """Find the first-line time of a (poor) capture; see :mod:`.findstart`."""
    if args.format == "fm":
        native = read_iq(input_path, raw_sample_rate=args.raw_sample_rate, target_rate=None, raw_format=args.raw_format or "f32")
        samples = np.conj(native.samples) if args.invert_iq else native.samples
        _check_capture_length(samples, native.sample_rate)
        location = locate_fm_signal(samples, native.sample_rate, args.signal_seconds)
        if location is None:
            return StartEstimate(
                start_s=None, confidence="NONE", onset_s=0.0, notes=["no FM signal found"]
            )
        onset = location.start_s
        window = (
            onset - FORCE_LEAD_S,
            onset + 0.91 + IMAGE_MS / 1000.0 + FORCE_TAIL_S + 1.0,
        )
        located = locate_fm_signal(samples, native.sample_rate, args.signal_seconds, window=window)
        if located is None:
            return StartEstimate(
                start_s=None, confidence="NONE", onset_s=onset, notes=["no clear carrier"]
            )
        fm = demodulate_fm(
            samples,
            native.sample_rate,
            located,
            half_bw_hz=None if args.fm_bandwidth is None else args.fm_bandwidth / 2.0,
        )
        audio, rate, t0_s = fm.audio, fm.sample_rate, fm.t0_s
    else:
        data = read_iq(input_path, raw_sample_rate=args.raw_sample_rate, raw_format=args.raw_format or "f32") if args.format == "iq" else read_wav(input_path)
        if args.format == "iq" and args.invert_iq:
            data.samples = np.conj(data.samples)
        audio, rate, t0_s = data.samples, data.sample_rate, 0.0
        onset = locate_signal(audio, rate)[0] / 1000.0
    try:
        vis = detect_vis(audio, rate)
    except (ValueError, IndexError):
        vis = None
    return find_start(audio, rate, t0_s, onset, vis)


def _command_hint(args: argparse.Namespace, input_path: Path, start_s: float) -> str:
    parts = ["python3 -m robot36_decode", f'"{input_path}"']
    if args.format != "wav":
        parts.append(f"--format {args.format}")
    if args.raw_format is not None:
        parts.append(f"--raw-format {args.raw_format}")
    if args.raw_sample_rate != 48000:
        parts.append(f"--raw-sample-rate {args.raw_sample_rate}")
    if args.invert_iq:
        parts.append("--invert-iq")
    parts.append(f"--force-start {start_s:.4f}")
    return " ".join(parts)


def _run_find_start(args: argparse.Namespace, files: list[Path]) -> int:
    """``--find-start``: report the estimated start of each file, no decoding."""
    found = 0
    for path in files:
        try:
            estimate = _estimate_start(args, path)
        except Exception as exc:  # noqa: BLE001 - reported per file, see _describe_error
            print(f"error: {path}: {_describe_error(exc)}", file=sys.stderr)
            continue
        if len(files) == 1:
            if not args.quiet:
                print(f"Input:      {path}")
            hint = _command_hint(args, path, estimate.start_s) if estimate.start_s is not None else None
            print(format_report(estimate, hint))
        else:
            start = "-" if estimate.start_s is None else f"{estimate.start_s:.4f} s"
            print(f"{estimate.confidence:<7} {start:>11}  {path}")
        found += estimate.start_s is not None
    return 0 if found == len(files) and files else 1


def _default_output(input_path: Path) -> Path:
    return input_path.with_suffix(".png")


def _human_time(ms: float) -> str:
    return f"{ms / 1000.0:.3f} s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot36-decode",
        description=(
            "Decode a Robot 36 colour SSTV image from a WAV file containing "
            "FM-demodulated SSTV baseband audio."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=_input_path,
        help=(
            "input file, or a directory: every valid input file in the "
            "directory is decoded (.wav; with --format iq/fm also .iq and .raw)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        action=_PathAction,
        help=(
            "output PNG file (default: input name with .png extension). A path "
            "ending in / (or an existing directory) means a directory, which is "
            "created if needed and receives <input name>.png. With a directory "
            "input this is always the output directory (default: each PNG beside "
            "its input)"
        ),
    )
    parser.add_argument(
        "--start-ms",
        type=float,
        metavar="MS",
        help=(
            "approximate image start time in milliseconds, measured to the first "
            "line sync; disables VIS detection"
        ),
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="ignore the VIS header and locate the image with sync tracking",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help=(
            "do not track line sync pulses; assume perfectly aligned fixed "
            "150 ms lines"
        ),
    )
    parser.add_argument(
        "--force-start",
        type=_force_start_value,
        default=None,
        metavar="SECONDS|auto",
        help=(
            "last resort for captures too poor to decode normally: take the first "
            "line sync as starting exactly SECONDS into the file (for --format fm, "
            "into the capture) and decode 240 fixed 150 ms lines with no VIS "
            "detection, no sync tracking and no signal search. Lines beyond the "
            "end of the file are left grey. Implies --no-vis and --no-sync. "
            "'auto' finds the start first, as --find-start does"
        ),
    )
    parser.add_argument(
        "--find-start",
        action="store_true",
        help=(
            "do not decode; analyse the file and report the time where the first "
            "line starts (for use with --force-start), with a confidence rating. "
            "With a directory input, one line per file"
        ),
    )
    parser.add_argument(
        "--no-rescue",
        action="store_true",
        help=(
            "(directory input) do not retry files that fail to decode with an "
            "automatically found --force-start"
        ),
    )
    parser.add_argument(
        "--black-hz",
        type=float,
        default=FREQ_BLACK,
        metavar="HZ",
        help="tone frequency representing black",
    )
    parser.add_argument(
        "--white-hz",
        type=float,
        default=FREQ_WHITE,
        metavar="HZ",
        help="tone frequency representing white",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        metavar="N",
        help="integer nearest-neighbour upscale factor for the output PNG",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print errors",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="(directory input) also process files in subfolders; layout is mirrored under -o",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "(directory input) skip files whose output PNG (or a rescued "
            "<name>_forced.png) already exists"
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "(directory input) decode N files at the same time; 0 uses every "
            "CPU core. Each job needs about 0.6 GB of memory per 2 minutes of "
            "78 kHz capture"
        ),
    )
    parser.add_argument(
        "--format",
        choices=["wav", "iq", "fm"],
        default="wav",
        help=(
            "input sample format: wav (real audio, mixed down to mono) or iq "
            "(complex I/Q baseband). IQ is treated as the analytic signal of "
            "the SSTV audio (SSB-style baseband, where the instantaneous "
            "frequency is the SSTV tone); FM-modulated captures must be "
            "FM-demodulated first. A Doppler/tuning offset is tracked from the "
            "sync pulses. fm is a raw SDR baseband recording of an FM carrier "
            "(for example an SDR# capture of a satellite downlink at any "
            "sample rate): the signal is located, its Doppler-shifted carrier "
            "tracked, dead air trimmed, and the audio FM-demodulated"
        ),
    )
    parser.add_argument(
        "--signal-seconds",
        type=float,
        default=DEFAULT_EXPECTED_S,
        metavar="S",
        help="(fm) approximate transmission length, used to trim a long noisy region",
    )
    parser.add_argument(
        "--fm-bandwidth",
        type=float,
        default=None,
        metavar="HZ",
        help=(
            "(fm) total channel filter bandwidth around the tracked carrier "
            "(default: chosen from the signal strength, 10-14 kHz)"
        ),
    )
    parser.add_argument(
        "--repair-lines",
        action="store_true",
        help=(
            "interpolate image lines wrecked by interference bursts from their "
            "neighbours (only lines that differ strongly from two agreeing "
            "neighbours); modifies image data, so off by default"
        ),
    )
    parser.add_argument(
        "--save-audio",
        action=_PathAction,
        metavar="WAV",
        help=(
            "(fm) also write the demodulated, trimmed 48 kHz audio as a WAV "
            "file; a path ending in / (or an existing directory) receives "
            "<input name>.wav; with a directory input, a directory for one WAV "
            "per capture"
        ),
    )
    parser.add_argument(
        "--raw-sample-rate",
        type=int,
        default=48000,
        metavar="HZ",
        help=(
            "sample rate of raw interleaved float32 I/Q files (.iq/.raw); "
            "WAV files carry their own rate. Resampled to 48000 Hz"
        ),
    )
    parser.add_argument(
        "--raw-format",
        choices=["f32", "s16"],
        default=None,
        help=(
            "sample type of headerless I/Q data (.iq/.raw, or a WAV whose header "
            "is damaged): f32 = 32-bit float (default), s16 = 16-bit integer, as "
            "SDR# and SDR++ record. With s16, .wav files without a WAV header in a "
            "folder are decoded as raw instead of skipped; give --raw-sample-rate too"
        ),
    )
    parser.add_argument(
        "--invert-iq",
        action="store_true",
        help="conjugate the I/Q samples (inverted-spectrum captures); iq/fm formats only",
    )
    parser.set_defaults(output_dir_hint=False, save_audio_dir_hint=False)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _late_vis(
    audio: WavData | IQData,
    tone_offset_hz: float,
    energy_start_ms: float,
    phase_track: np.ndarray | None,
) -> VisInfo | None:
    """Look for a VIS header beyond the first ``VIS_SEARCH_MS`` of a long recording.

    ``detect_vis`` only searches the start of the audio. A recording of a
    whole pass can have a minute of noise before the transmission, so search
    again just before where the signal energy starts and just before where
    the sync pulse train starts. Only a clean Robot 36 header (right code,
    parity OK) is accepted here, since a wrong one would misplace the image.
    """
    duration_ms = audio.duration_ms
    if duration_ms <= VIS_SEARCH_MS:
        return None
    anchors: list[float] = []
    if energy_start_ms > 0.0:
        anchors.append(energy_start_ms - 2000.0)
    provisional = track_line_starts(
        audio.samples, audio.sample_rate, tone_offset_hz=tone_offset_hz, phase_track=phase_track
    )
    if provisional is not None:
        # The header ends at line 0; leave room for a mis-numbered track.
        anchors.append(float(provisional.line_starts_ms[0]) - 0.5 * VIS_SEARCH_MS)
    searched: list[float] = []
    for anchor in anchors:
        start_ms = max(0.0, anchor)
        if start_ms < VIS_SEARCH_MS - 2000.0:
            continue  # already covered by the first search
        if any(abs(start_ms - done) < 2000.0 for done in searched):
            continue
        searched.append(start_ms)
        first = int(start_ms * audio.sample_rate / 1000.0)
        if first >= audio.samples.size:
            continue
        found = detect_vis(
            audio.samples[first:],
            audio.sample_rate,
            tone_offset_hz=tone_offset_hz,
            phase_track=None if phase_track is None else phase_track[first:],
        )
        if found is not None and found.code == ROBOT36_VIS_CODE and found.parity_ok:
            shift = first * 1000.0 / audio.sample_rate
            return VisInfo(
                code=found.code,
                image_start_ms=found.image_start_ms + shift,
                start_bit_ms=found.start_bit_ms + shift,
                parity_ok=found.parity_ok,
            )
    return None


def _locate_image(
    audio: WavData | IQData,
    args: argparse.Namespace,
    tone_offset_hz: float = 0.0,
    energy_start_ms: float = 0.0,
    use_energy_start: bool = False,
    phase_track: np.ndarray | None = None,
    end_hint_ms: float | None = None,
    train_start=None,
) -> tuple[float, float, VisInfo | None, SyncTrack | None, str]:
    """Work out where the image starts and ends.

    ``end_hint_ms`` (FM captures) is where the image should end if the carrier
    stopped with the transmission. ``train_start`` is an optional callable
    (see :func:`_line_numbering_hint`) asked only when there is no readable
    VIS header; its answer, when it has one, numbers the lines instead.

    Returns (first_line_ms, end_ms, vis, track, start_source).
    """
    if args.start_ms is not None:
        if args.start_ms < 0:
            raise ValueError("--start-ms cannot be negative")
        nominal_start = args.start_ms
        expected_start: float | None = args.start_ms
        vis: VisInfo | None = None
        start_source = "command line"
    elif args.no_vis:
        vis = None
        if end_hint_ms is not None:
            nominal_start = end_hint_ms - IMAGE_MS
            start_source = "end of signal (no VIS)"
        else:
            nominal_start = 0.0
            start_source = "assumed (no VIS)"
        expected_start = None
    else:
        vis = detect_vis(
            audio.samples,
            audio.sample_rate,
            tone_offset_hz=tone_offset_hz,
            phase_track=phase_track,
        )
        if vis is None and end_hint_ms is None:
            vis = _late_vis(audio, tone_offset_hz, energy_start_ms, phase_track)
        parity_failed = vis is not None and vis.parity_ok is False
        if vis is not None:
            # A header that fails its parity check cannot be trusted for the
            # mode code (so it is not rejected as "not Robot 36"), but its
            # start-bit timing is still a good hint for where the image starts.
            if not parity_failed and vis.code != ROBOT36_VIS_CODE:
                raise ValueError(
                    f"VIS header reports mode code 0x{vis.code:02X}, which is "
                    "not Robot 36 (0x08); this decoder only supports Robot 36. "
                    "Use --no-vis or --start-ms to force a Robot 36 decode."
                )
            nominal_start = vis.image_start_ms
            expected_start = nominal_start
            start_source = "VIS header"
        elif end_hint_ms is not None:
            # No readable header. Usually the end of the carrier tells the
            # tracker which line is line 0 (the transmission ends with the
            # image). A signal that fades out early is numbered from where its
            # pulse train begins instead (see _decode_file).
            numbering = train_start() if train_start is not None else None
            if numbering is not None and numbering[0] == "start":
                nominal_start = numbering[1]
                expected_start = numbering[1]
                start_source = "start of the sync pulse train (no VIS found)"
            elif numbering is not None:
                end_hint_ms = numbering[1]
                nominal_start = end_hint_ms - IMAGE_MS
                expected_start = None
                start_source = "end of the sync pulse train (no VIS found)"
            else:
                nominal_start = end_hint_ms - IMAGE_MS
                expected_start = None
                start_source = "end of signal (no VIS found)"
        elif use_energy_start and energy_start_ms > 0.0:
            # No VIS header: seed the sync tracker with the energy-located
            # signal start so the tracker refines from a real signal position
            # instead of searching the whole recording for a pulse grid.
            nominal_start = energy_start_ms
            expected_start = energy_start_ms
            start_source = "energy-located signal"
        else:
            nominal_start = 0.0
            expected_start = None
            start_source = "assumed (no VIS found)"
        if parity_failed:
            start_source = "VIS timing (parity check failed)"

    track: SyncTrack | None = None
    rejected = False
    if not args.no_sync:
        track = track_line_starts(
            audio.samples,
            audio.sample_rate,
            expected_start_ms=expected_start,
            tone_offset_hz=tone_offset_hz,
            phase_track=phase_track,
            expected_end_ms=end_hint_ms if expected_start is None else None,
        )
        if (
            track is not None
            and end_hint_ms is not None
            and abs(track.tone_offset_hz) > MAX_FM_TONE_OFFSET_HZ
        ):
            # FM-demodulated audio has no Doppler: a "sync tone" this far off
            # 1200 Hz is picture content or interference, not line sync.
            track = None
            rejected = True

    if track is not None:
        first_line = float(track.line_starts_ms[0])
        end_ms = float(track.line_starts_ms[-1]) + LINE_MS * track.line_scale
        start_source = f"{start_source} + sync tracking"
    else:
        first_line = nominal_start
        end_ms = nominal_start + IMAGE_MS
        if not args.no_sync:
            start_source = (
                f"{start_source}; sync tracking rejected (locked off the sync tone)"
                if rejected
                else f"{start_source}; sync tracking failed"
            )

    return first_line, end_ms, vis, track, start_source


class _PhaseCache:
    """Compute the whole-recording phase once per analysis-band setting."""

    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        self._samples = samples
        self._rate = sample_rate
        self._items: dict[tuple[float, float, float], np.ndarray] = {}

    def get(
        self, tone_offset_hz: float, band_span_hz: tuple[float, float] = (0.0, 0.0)
    ) -> np.ndarray:
        key = (
            round(float(tone_offset_hz), 3),
            round(float(band_span_hz[0]), 3),
            round(float(band_span_hz[1]), 3),
        )
        if key not in self._items:
            self._items[key] = phase(
                self._samples,
                self._rate,
                tone_offset_hz=key[0],
                band_span_hz=(key[1], key[2]),
            )
        return self._items[key]


# Sample rates sound cards and audio programs record at. A stereo WAV at any
# other rate (SDR# writes e.g. 78125 Hz) is almost certainly an I/Q capture.
AUDIO_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000)


def _iq_hint(audio: WavData | IQData | None, args: argparse.Namespace) -> str | None:
    """A hint when a file read as plain audio looks like an SDR I/Q capture."""
    if args.format != "wav" or not isinstance(audio, WavData) or audio.channels != 2:
        return None
    rate = audio.source_rate
    if rate is None or rate in AUDIO_RATES:
        return None
    return (
        f"this WAV has 2 channels at {rate} Hz, which looks like an SDR I/Q "
        "recording rather than audio; for a satellite FM downlink (e.g. the ISS) "
        "use --format fm, for SSB-style I/Q use --format iq"
    )


# A partial picture needs at least this many lines with audio to be worth writing.
MIN_PARTIAL_LINES = 20
# The ISS carrier stays on at least this long after the last picture line.
END_TAIL_MIN_MS = 850.0
# Lines more than this far outside the located carrier are greyed (FM).
CARRIER_EDGE_MS = 300.0
# Plausibility of a decode without a VIS header (see _implausible_picture).
MAX_FM_TONE_OFFSET_HZ = 60.0
MIN_TRACKED_FRACTION = 0.35
MAX_SEPARATOR_MISMATCH_FRACTION = 0.40


def _line_numbering_hint(fm: FmAudio, phases):
    """For an FM capture without a VIS header: a callable that says how to
    number the lines, as ``("end", ms)`` (the last line ends at ``ms``),
    ``("start", ms)`` (line 0 starts at ``ms``) or None (no evidence).

    The ISS keeps its carrier on for about a second after the last picture
    line (0.9-1.8 s in real passes), with no sync pulses. When the capture
    shows that quiet tail, the transmission ended inside it and the last sync
    pulse marks the end of line 239 (more precisely than the carrier's end).
    When the pulses run right up to where the carrier fades, the end of the
    picture is missing, and line 0 is found from where the pulse train begins
    (with the header pattern before it, see :mod:`.findstart`).
    """
    cache: dict[str, tuple[str, float] | None] = {}

    def hint() -> tuple[str, float] | None:
        if "value" in cache:
            return cache["value"]
        value = None
        track = track_line_starts(fm.audio, fm.sample_rate, phase_track=phases.get(0.0))
        last = _last_dense_line(track)
        if last is not None:
            last_end_ms = float(track.line_starts_ms[last]) + LINE_MS * track.line_scale
            carrier_end_ms = (fm.signal_end_s - fm.t0_s) * 1000.0
            if carrier_end_ms - last_end_ms >= END_TAIL_MIN_MS:
                value = ("end", last_end_ms)
            else:
                estimate = find_start(fm.audio, fm.sample_rate, fm.t0_s, fm.signal_start_s, None)
                if estimate.start_s is not None and estimate.confidence in ("HIGH", "MEDIUM"):
                    value = ("start", (estimate.start_s - fm.t0_s) * 1000.0)
        cache["value"] = value
        return value

    return hint


def _last_dense_line(track: SyncTrack | None, window: int = 8, needed: int = 4) -> int | None:
    """The last line of the pulse train that is followed by no gap: a line
    whose ``window`` lines up to it hold at least ``needed`` detected pulses.
    (Noise after the carrier can produce the odd stray match on the grid.)"""
    if track is None or track.matched_idx is None or not len(track.matched_idx):
        return None
    matched = np.zeros(LINES + window, dtype=int)
    idx = np.asarray(track.matched_idx, dtype=int)
    matched[idx[(idx >= 0) & (idx < LINES)] + window] = 1
    counts = np.convolve(matched, np.ones(window, dtype=int), mode="full")[window : LINES + window]
    dense = np.flatnonzero((counts >= needed) & (matched[window:] == 1))
    return int(dense[-1]) if dense.size else None


def _implausible_picture(
    track: SyncTrack | None,
    stats,
    line_valid: np.ndarray,
    fm_audio: bool,
    header: bool = False,
) -> str | None:
    """Why a decode is not a Robot 36 picture, or None.

    Without a header nothing confirms the mode, and on a carrier that holds no
    Robot 36 (voice, another mode, noise) the sync tracker can still lock onto
    something periodic. Such a "picture" is noise, so it is refused. With a
    header but no usable line sync, only the colour separators are checked:
    the header proves the mode, not that a picture followed it.
    """
    recorded = int(np.count_nonzero(line_valid))
    partial = recorded < LINES
    if header:
        return _separator_problem(stats)
    if track is None:
        # Fixed 150 ms lines are only trusted for a complete recording.
        return "no line sync pulses found in a partial recording" if partial else None
    if fm_audio and abs(track.tone_offset_hz) > MAX_FM_TONE_OFFSET_HZ:
        # FM demodulation removes Doppler: the tones of demodulated audio are
        # where they belong, so a far-off "sync tone" is something else.
        return (
            f"the line sync is {track.tone_offset_hz:+.0f} Hz off the 1200 Hz sync tone, "
            "which demodulated FM audio cannot be"
        )
    matched = 0
    if track.matched_idx is not None:
        idx = np.asarray(track.matched_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < LINES)]
        matched = int(np.count_nonzero(line_valid[idx]))
    if recorded and matched < MIN_TRACKED_FRACTION * recorded:
        return f"only {matched} of {recorded} lines show a sync pulse"
    if fm_audio:
        # Sync pulses on the 1200 Hz tone are the stronger evidence. In heavy
        # FM noise the short separator tones read low (odd lines look even),
        # so a real but very noisy picture can fail the separator test.
        return None
    return _separator_problem(stats)


def _separator_problem(stats) -> str | None:
    if stats.checked_lines >= MIN_PARTIAL_LINES and (
        stats.separator_mismatches > MAX_SEPARATOR_MISMATCH_FRACTION * stats.checked_lines
    ):
        return (
            f"{stats.separator_mismatches} of {stats.checked_lines} colour separators are "
            "out of order (a Robot 36 picture alternates them; noise gets about half wrong)"
        )
    return None


def _decode_file(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    save_audio: Path | None,
    error_sink: list[str] | None = None,
    force_start: float | None = None,
) -> int:
    """Decode one input file to a PNG. Returns 0 on success, 1 on failure.

    ``force_start`` (seconds) overrides ``--force-start``; ``--force-start
    auto`` is resolved here with :func:`_estimate_start`.
    """
    audio: WavData | IQData | None = None
    try:
        if force_start is None:
            force_start = args.force_start
        if force_start == "auto":
            estimate = _estimate_start(args, input_path)
            if not args.quiet:
                print("Finding the start (--force-start auto):")
                print("  " + format_report(estimate, None).replace("\n", "\n  "))
            if estimate.start_s is None:
                raise ValueError(
                    "--force-start auto could not find the start of the image; "
                    "try --find-start for details, or give --force-start SECONDS"
                )
            force_start = estimate.start_s
        fm: FmAudio | None = None
        if args.format == "fm":
            native = read_iq(input_path, raw_sample_rate=args.raw_sample_rate, target_rate=None, raw_format=args.raw_format or "f32")
            samples = np.conj(native.samples) if args.invert_iq else native.samples
            _check_capture_length(samples, native.sample_rate)
            if force_start is not None:
                file_s = samples.size / native.sample_rate
                if force_start >= file_s:
                    raise ValueError(
                        f"--force-start {force_start:g} s is beyond the end of "
                        f"the file ({file_s:.1f} s)"
                    )
                # The caller names the stretch; only the carrier is estimated.
                window = (
                    force_start - FORCE_LEAD_S,
                    force_start + IMAGE_MS / 1000.0 + FORCE_TAIL_S,
                )
                location = locate_fm_signal(
                    samples, native.sample_rate, args.signal_seconds, window=window
                )
                if location is None:
                    raise ValueError(
                        "no FM carrier found in the forced window "
                        f"({window[0]:.1f} s - {window[1]:.1f} s); check --format and "
                        "--force-start"
                    )
            else:
                location = locate_fm_signal(samples, native.sample_rate, args.signal_seconds)
                if location is None:
                    raise ValueError("no FM signal found in the capture")
            fm = demodulate_fm(
                samples,
                native.sample_rate,
                location,
                half_bw_hz=None if args.fm_bandwidth is None else args.fm_bandwidth / 2.0,
            )
            audio = WavData(
                samples=fm.audio,
                sample_rate=fm.sample_rate,
                channels=1,
                bits=native.bits,
                encoding=f"FM-demodulated {native.sample_rate} Hz baseband",
            )
            if save_audio is not None:
                write_audio_wav(save_audio, fm.audio, fm.sample_rate)
        elif args.format == "iq":
            audio = read_iq(input_path, raw_sample_rate=args.raw_sample_rate, raw_format=args.raw_format or "f32")
            if args.invert_iq:
                audio.samples = np.conj(audio.samples)
        else:
            audio = read_wav(input_path)

        phases = _PhaseCache(audio.samples, audio.sample_rate)
        forced = force_start is not None
        if forced:
            # No search, no VIS, no tracking: the caller says where line 0 starts.
            t0_s = fm.t0_s if fm is not None else 0.0
            first_line = (force_start - t0_s) * 1000.0
            end_ms = first_line + IMAGE_MS
            vis = None
            track = None
            start_source = "forced by --force-start; VIS and sync tracking skipped"
            used_tone_offset_hz = 0.0
            used_band_span = (0.0, 0.0)
        else:
            # Energy location only seeds the image start when no VIS header is
            # found. The tone (Doppler) offset is NOT taken from the spectrum --
            # the dominant bin depends on picture content -- but measured from the
            # sync pulses by the tracker.
            # (FM captures are already trimmed to the located carrier, and their
            # end-of-signal hint takes precedence, so they skip this search.)
            energy_start_ms = 0.0
            if audio.samples.size and fm is None:
                energy_start_ms, _sig_end_ms, _centre_hz = locate_signal(
                    audio.samples, audio.sample_rate
                )

            end_hint_ms = None
            train_start = None
            if fm is not None:
                end_hint_ms = (fm.signal_end_s - fm.t0_s) * 1000.0 - FM_TAIL_MS
                train_start = _line_numbering_hint(fm, phases)
            min_matched_lines = max(4, int(LINES * 0.5))

            first_line, end_ms, vis, track, start_source = _locate_image(
                audio,
                args,
                tone_offset_hz=0.0,
                energy_start_ms=energy_start_ms,
                use_energy_start=False,
                phase_track=phases.get(0.0),
                end_hint_ms=end_hint_ms,
                train_start=train_start,
            )
            used_tone_offset_hz = 0.0
            used_band_span = (0.0, 0.0)

            # Second pass, only when sync tracking is enabled: either the nominal
            # lock was weak, or it found a clearly shifted sync tone (Doppler), so
            # VIS detection and the analysis band should follow the shifted signal.
            if not args.no_sync:
                # Offset at line 0: the VIS header sits just before it, so a
                # drifting Doppler shift is best matched there. The analysis band
                # is widened to cover the whole measured sweep.
                estimate = 0.0
                span = (0.0, 0.0)
                sweep = 0.0
                if track is not None and track.line_offsets_hz is not None:
                    offsets = track.line_offsets_hz
                    estimate = float(offsets[0])
                    span = (
                        min(0.0, float(offsets.min()) - estimate) - BAND_MARGIN_HZ,
                        max(0.0, float(offsets.max()) - estimate) + BAND_MARGIN_HZ,
                    )
                    sweep = float(offsets.max() - offsets.min())
                weak = track is None or track.matched_lines < min_matched_lines
                if weak or abs(estimate) > OFFSET_RETRY_HZ or sweep > 2 * BAND_MARGIN_HZ:
                    retry = _locate_image(
                        audio,
                        args,
                        tone_offset_hz=estimate,
                        energy_start_ms=energy_start_ms,
                        use_energy_start=True,
                        phase_track=phases.get(estimate, span),
                        end_hint_ms=end_hint_ms,
                        train_start=train_start,
                    )
                    retry_track = retry[3]
                    # Keep whichever pass tracked more lines.
                    better = (
                        retry_track is not None
                        and (track is None or retry_track.matched_lines >= track.matched_lines)
                    ) or (track is None and retry_track is None)
                    if better:
                        first_line, end_ms, vis, track, start_source = retry
                        used_tone_offset_hz = estimate
                        used_band_span = span

        start_sample = int(round(first_line * audio.sample_rate / 1000.0))
        required_samples = int(round((end_ms - first_line) * audio.sample_rate / 1000.0))
        line_samples = LINE_MS * audio.sample_rate / 1000.0
        # The image may have begun before the recording (or before a dropout),
        # and may end after it: those lines have no audio and are grey.
        lead_missing = max(0, -start_sample)
        if start_sample >= audio.samples.size or lead_missing >= required_samples:
            where = "beyond the end" if start_sample >= 0 else "entirely before the start"
            raise ValueError(
                f"the located image ({_human_time(first_line)} - {_human_time(end_ms)}) lies "
                f"{where} of the audio ({_human_time(audio.duration_ms)})"
            )
        first_sample = start_sample + lead_missing
        image_samples = audio.samples[first_sample : start_sample + required_samples]
        image_phase = phases.get(used_tone_offset_hz, used_band_span)[
            first_sample : start_sample + required_samples
        ]
        if lead_missing:
            image_samples = np.pad(image_samples, (lead_missing, 0))
            image_phase = np.pad(image_phase, (lead_missing, 0), mode="edge")
        # Allow a couple of milliseconds of rounding shortfall at the end.
        tolerance = int(round(0.002 * audio.sample_rate))
        tail_missing = required_samples - image_samples.size
        valid_lines = LINES
        if tail_missing > tolerance:
            valid_lines = min(LINES, int((image_samples.size) // line_samples))
        if tail_missing > 0:
            image_samples = np.pad(image_samples, (0, tail_missing))
            image_phase = np.pad(image_phase, (0, tail_missing), mode="edge")
        missing_top = min(LINES, int(np.ceil(lead_missing / line_samples))) if lead_missing else 0
        if fm is not None and not forced:
            # The demodulated audio runs a little past both ends of the
            # carrier; lines entirely outside the carrier are just noise.
            carrier_from = (fm.signal_start_s - fm.t0_s) * 1000.0 - CARRIER_EDGE_MS
            carrier_to = (fm.signal_end_s - fm.t0_s) * 1000.0 + CARRIER_EDGE_MS
            if carrier_from > first_line:
                missing_top = max(missing_top, min(LINES, int(np.ceil((carrier_from - first_line) / LINE_MS))))
            valid_lines = max(0, min(valid_lines, int((carrier_to - first_line) // LINE_MS)))
        recorded_lines = max(0, valid_lines - missing_top)
        if not forced and recorded_lines < MIN_PARTIAL_LINES:
            raise ValueError(
                f"only {recorded_lines} of the {LINES} image lines are in the audio "
                f"(the image is located at {_human_time(first_line)} - {_human_time(end_ms)}, "
                f"the audio is {_human_time(audio.duration_ms)} long)"
                + (
                    f". The signal was located at {_human_time(fm.signal_start_s * 1000)}"
                    f" - {_human_time(fm.signal_end_s * 1000)} of the capture; "
                    "--find-start reports where the image starts, and --force-start auto "
                    "decodes from there"
                    if fm is not None
                    else ""
                )
            )
        line_valid = np.zeros(LINES, dtype=bool)
        line_valid[missing_top:valid_lines] = True
        # Lines inside a receiver dropout carry no picture: grey them too.
        dropout_lines = np.zeros(LINES, dtype=bool)
        if fm is not None and fm.dropouts_s:
            if track is not None:
                line_begin = np.asarray(track.line_starts_ms, dtype=np.float64)
            else:
                line_begin = first_line + np.arange(LINES) * LINE_MS
            for a_s, b_s in fm.dropouts_s:
                a_ms, b_ms = (a_s - fm.t0_s) * 1000.0, (b_s - fm.t0_s) * 1000.0
                dropout_lines |= (line_begin < b_ms) & (line_begin + LINE_MS > a_ms)
            line_valid &= ~dropout_lines
        relative_starts = None
        line_offsets = None
        if track is not None:
            relative_starts = track.line_starts_ms - first_line
            line_offsets = track.line_offsets_hz

        rgb, stats = decode_robot36(
            image_samples,
            audio.sample_rate,
            black_hz=args.black_hz,
            white_hz=args.white_hz,
            line_starts_ms=relative_starts,
            tone_offset_hz=used_tone_offset_hz,
            line_offsets_hz=line_offsets,
            phase_track=image_phase,
            repair_lines=args.repair_lines,
            line_valid=line_valid,
        )
        if not forced and (vis is None or track is None):
            problem = _implausible_picture(
                track, stats, line_valid, fm is not None, header=vis is not None
            )
            if problem is not None:
                raise ValueError(f"no Robot 36 picture found: {problem}")
        if valid_lines < LINES:
            rgb[valid_lines:] = 128
        if missing_top:
            rgb[:missing_top] = 128
        rgb[dropout_lines] = 128
        write_png(
            output_path,
            rgb,
            scale=args.scale,
            title=f"robot36-decode v{__version__}",
        )

        hint = _iq_hint(audio, args)
        if hint is not None:
            print(f"warning: {input_path}: {hint}", file=sys.stderr)
        if not args.quiet:
            channel_text = "mono" if audio.channels == 1 else f"{audio.channels} ch"
            print(f"Input:  {input_path}")
            print(
                f"Audio:  {audio.sample_rate} Hz, {channel_text}, "
                f"{audio.bits}-bit {audio.encoding}, {_human_time(audio.duration_ms)}"
            )
            if fm is not None:
                print(
                    f"FM:     signal {_human_time(fm.signal_start_s * 1000)} - "
                    f"{_human_time(fm.signal_end_s * 1000)} of the capture "
                    f"({fm.signal_end_s - fm.signal_start_s:.1f} s); carrier "
                    f"{fm.doppler_hz[0] / 1000:+.2f} -> {fm.doppler_hz[-1] / 1000:+.2f} kHz "
                    f"from centre (Doppler {fm.doppler_rate_hz_per_s:+.1f} Hz/s)"
                )
                print(
                    f"Image:  starts at {_human_time(fm.t0_s * 1000 + first_line)} "
                    "in the capture"
                )
            if vis is not None:
                parity_text = "" if vis.parity_ok else " (bad parity)"
                print(
                    f"VIS:    code 0x{vis.code:02X} (Robot 36){parity_text}, "
                    f"start bit at {_human_time(vis.start_bit_ms)}"
                )
            if track is not None:
                print(
                    f"Sync:   {track.matched_lines}/{LINES} lines tracked, "
                    f"drift {track.drift_ms_per_line:+.3f} ms/line"
                )
                if track.line_offsets_hz is not None and (
                    abs(track.tone_offset_hz) >= 1.0
                    or abs(track.line_offsets_hz[-1] - track.line_offsets_hz[0]) >= 1.0
                ):
                    swing = float(track.line_offsets_hz[-1] - track.line_offsets_hz[0])
                    print(
                        f"Tone:   {track.tone_offset_hz:+.1f} Hz offset "
                        f"(Doppler/tuning), {swing:+.1f} Hz change over the image"
                    )
            else:
                print("Sync:   not tracked; using fixed 150 ms lines")
            print(f"Start:  {_human_time(first_line)} ({start_source})")
            if missing_top:
                print(
                    f"Note:   the recording starts {missing_top} lines into the image; "
                    "those lines are grey."
                )
            if valid_lines < LINES:
                print(
                    f"Note:   the recording ends after line {valid_lines} of {LINES}; "
                    "the remaining lines are grey."
                )
            if dropout_lines.any():
                print(
                    f"Note:   {int(np.count_nonzero(dropout_lines))} lines fall in a receiver "
                    "dropout (no signal recorded); they are grey."
                )
            if stats.repaired_lines:
                print(f"Repair: {stats.repaired_lines} line(s) interpolated from neighbours")
            if stats.separator_mismatches:
                print(
                    f"Note:   {stats.separator_mismatches} of {stats.checked_lines} line "
                    "separators did not match the expected alternating colour order."
                )
            print("Decoded: 320x240 RGB")
            print(f"Output: {output_path}")
        return 0

    except Exception as exc:  # noqa: BLE001 - reported per file, see _describe_error
        message = _describe_error(exc)
        hint = _iq_hint(audio, args)
        if hint is not None:
            message = f"{message}. Note: {hint}"
        print(f"error: {message}", file=sys.stderr)
        if error_sink is not None:
            error_sink.append(message)
        return 1



# --force-start: the carrier is tracked from this long before the forced first
# line (room for the VIS header) to this long after the last line.
FORCE_LEAD_S = 1.5
FORCE_TAIL_S = 0.5

# File extensions considered when the input is a directory.
WAV_EXTENSIONS = (".wav",)
IQ_EXTENSIONS = (".wav", ".iq", ".raw")


def _candidate_files(directory: Path, formats: str, recursive: bool) -> list[Path]:
    """Files in ``directory`` that could be decoder input, in name order."""
    allowed = WAV_EXTENSIONS if formats == "wav" else IQ_EXTENSIONS
    found: list[Path] = []
    walker = directory.rglob("*") if recursive else directory.glob("*")
    for path in walker:
        relative_parts = path.relative_to(directory).parts
        if any(part.startswith(".") for part in relative_parts):
            continue  # hidden files/folders, macOS "._" resource forks
        if path.suffix.lower() not in allowed or not path.is_file():
            continue
        found.append(path)
    return sorted(found, key=lambda item: [part.lower() for part in item.relative_to(directory).parts])


def _looks_like_wave(path: Path) -> bool:
    """True when ``path`` starts with a RIFF/RF64 WAVE header."""
    try:
        with path.open("rb") as handle:
            head = handle.read(12)
    except OSError:
        return False
    return len(head) == 12 and head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE"


def _not_wave_reason(path: Path, formats: str) -> str:
    """Why a .wav file is skipped, with a hint when it looks like a recording
    whose WAV header was overwritten (SDR software that crashed or was killed
    can leave the start of the file filled with 0xFF bytes)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(1 << 20)
    except OSError as exc:
        return f"cannot read ({exc})"
    if not head:
        return "empty file"
    fill = len(head) - len(head.lstrip(b"\xff"))
    if fill >= 64 and formats in ("iq", "fm"):
        return (
            f"no WAV header (the file starts with {fill if fill < len(head) else 'over 1 MB of'} "
            "0xFF bytes); if it is an SDR recording, decode it as headerless data: "
            "--raw-format s16 --raw-sample-rate <your SDR rate, e.g. 78125>"
        )
    return "not a valid WAV file"


def _batch_targets(
    inputs: list[Path], root: Path, out_dir: Path | None, suffix: str
) -> dict[Path, Path]:
    """Map each input to its output path.

    With no ``out_dir`` the output sits next to the input. Otherwise the
    folder layout under ``root`` is mirrored inside ``out_dir``. When several
    inputs would share an output name (``a.wav`` and ``a.iq``) the input
    extension is added to the name (``a_iq.png``) to keep them apart.
    """
    def base(path: Path) -> Path:
        if out_dir is None:
            return path.with_suffix(suffix)
        return (out_dir / path.relative_to(root)).with_suffix(suffix)

    counts: dict[Path, int] = {}
    for path in inputs:
        counts[base(path)] = counts.get(base(path), 0) + 1
    targets: dict[Path, Path] = {}
    for path in inputs:
        target = base(path)
        if counts[target] > 1:
            target = target.with_name(f"{target.stem}_{path.suffix.lstrip('.').lower()}{suffix}")
        targets[path] = target
    return targets


def _rescue(
    args: argparse.Namespace, path: Path, output: Path, save_audio: Path | None
) -> str | None:
    """Retry a file that failed normal decoding with an automatically found
    ``--force-start``. Returns a description of the result, or None."""
    try:
        estimate = _estimate_start(args, path)
    except Exception as exc:  # noqa: BLE001 - a failed rescue must not stop the batch
        if not args.quiet:
            print(f"Rescue: could not analyse the file ({_describe_error(exc)})")
        return None
    if estimate.start_s is None or estimate.confidence not in ("HIGH", "MEDIUM"):
        if not args.quiet:
            print(f"Rescue: no reliable start found ({estimate.confidence}); not forcing a decode")
        return None
    target = output.with_name(f"{output.stem}_forced.png")
    audio_target = None if save_audio is None else save_audio.with_name(f"{save_audio.stem}_forced.wav")
    if not args.quiet:
        print(
            f"Rescue: retrying with --force-start {estimate.start_s:.4f} "
            f"({estimate.confidence} confidence)"
        )
    if _decode_file(args, path, target, audio_target, None, force_start=estimate.start_s) != 0:
        return None
    return f"{target}  (forced start {estimate.start_s:.4f} s, {estimate.confidence} confidence)"


def _process_file(
    args: argparse.Namespace,
    path: Path,
    output: Path,
    save_audio: Path | None,
    capture: bool = False,
) -> tuple[str, str, str]:
    """Decode one file of a directory run, rescuing it if that fails.

    Returns ``(status, detail, text)``. With ``capture`` (parallel jobs) the
    messages are collected and returned as ``text`` instead of printed, so the
    output of files decoded at the same time does not interleave; in quiet
    mode only the error messages are kept.
    """
    import contextlib
    import io

    buffer = io.StringIO()
    out = contextlib.redirect_stdout(buffer) if capture and not args.quiet else contextlib.nullcontext()
    err = contextlib.redirect_stderr(buffer) if capture else contextlib.nullcontext()
    with out, err:
        output.parent.mkdir(parents=True, exist_ok=True)
        if save_audio is not None:
            save_audio.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        if _decode_file(args, path, output, save_audio, errors) == 0:
            status, detail = "ok", str(output)
        else:
            reason = errors[0] if errors else "decode failed"
            rescued = None if args.no_rescue else _rescue(args, path, output, save_audio)
            status, detail = ("rescued", rescued) if rescued is not None else ("FAILED", reason)
    return status, detail, buffer.getvalue()


def _run_directory(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Decode every valid input file in ``args.input`` (a directory)."""
    root: Path = args.input
    out_dir: Path | None = args.output
    audio_dir: Path | None = args.save_audio

    if out_dir is not None and out_dir.exists() and not out_dir.is_dir():
        parser.error("with a directory input, -o/--output must be a directory")
    if audio_dir is not None and audio_dir.exists() and not audio_dir.is_dir():
        parser.error("with a directory input, --save-audio must be a directory")
    if args.save_audio is not None and args.format != "fm":
        parser.error("--save-audio requires --format fm")

    candidates = _candidate_files(root, args.format, args.recursive)
    if not candidates:
        wanted = ", ".join(WAV_EXTENSIONS if args.format == "wav" else IQ_EXTENSIONS)
        print(f"error: no input files ({wanted}) found in {root}", file=sys.stderr)
        return 1

    skipped: list[tuple[Path, str]] = []
    inputs: list[Path] = []
    for path in candidates:
        if path.suffix.lower() == ".wav" and not _looks_like_wave(path) and args.raw_format is None:
            skipped.append((path, _not_wave_reason(path, args.format)))
        else:
            inputs.append(path)

    outputs = _batch_targets(inputs, root, out_dir, ".png")
    audio_outputs = _batch_targets(inputs, root, audio_dir, ".wav") if audio_dir is not None else {}
    resolved_inputs = {path.resolve() for path in inputs}
    for target in audio_outputs.values():
        if target.resolve() in resolved_inputs:
            parser.error(
                f"--save-audio would overwrite the input file {target}; "
                "choose a different folder"
            )

    results: list[tuple[Path, str, str]] = []  # (path, status, detail)
    already_done = 0
    todo: list[Path] = []
    for path in inputs:
        forced = outputs[path].with_name(f"{outputs[path].stem}_forced.png")
        if args.skip_existing and (outputs[path].exists() or forced.exists()):
            already_done += 1
            results.append((path, "skipped", "output exists"))
        else:
            todo.append(path)

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(todo)))
    work = [(args, path, outputs[path], audio_outputs.get(path)) for path in todo]
    numbered = {path: len(skipped) + already_done + n + 1 for n, path in enumerate(todo)}
    if jobs == 1:
        for item in work:
            path = item[1]
            if not args.quiet:
                print(f"[{numbered[path]}/{len(candidates)}] {path}")
            status, detail, _ = _process_file(*item)
            results.append((path, status, detail))
            if not args.quiet:
                print()
    else:
        import concurrent.futures
        import multiprocessing

        if not args.quiet:
            print(f"Decoding {len(todo)} file(s), {jobs} at a time")
            print()
        context = multiprocessing.get_context("spawn")  # same behaviour on every OS
        with concurrent.futures.ProcessPoolExecutor(jobs, mp_context=context) as pool:
            futures = {pool.submit(_process_file, *item, True): item[1] for item in work}
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    status, detail, text = future.result()
                except concurrent.futures.process.BrokenProcessPool:
                    status, text = "FAILED", ""
                    detail = (
                        "a decoding process stopped unexpectedly (out of memory?); "
                        "try fewer --jobs"
                    )
                except Exception as exc:  # noqa: BLE001 - a crashed worker must not stop the batch
                    status, detail, text = "FAILED", _describe_error(exc), ""
                results.append((path, status, detail))
                if not args.quiet:
                    print(f"[{numbered[path]}/{len(candidates)}] {path}")
                    if text:
                        print(text.rstrip("\n"))
                    print()
                elif text:
                    print(text.rstrip("\n"), file=sys.stderr)

    for path, reason in skipped:
        results.append((path, "skipped", reason))

    decoded = sum(1 for _, status, _ in results if status == "ok")
    rescued_count = sum(1 for _, status, _ in results if status == "rescued")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    not_run = sum(1 for _, status, _ in results if status == "skipped")
    if not args.quiet:
        rescue_text = f" ({rescued_count} rescued with --force-start)" if rescued_count else ""
        print(
            f"Summary: {decoded + rescued_count} decoded{rescue_text}, {failed} failed, "
            f"{not_run} skipped of {len(candidates)} file(s) in {root}"
        )
        width = max(len(str(path.relative_to(root))) for path, _, _ in results)
        for path, status, detail in sorted(results, key=lambda item: str(item[0]).lower()):
            print(f"  {status:<8} {str(path.relative_to(root)):<{width}}  {detail}")
    elif failed:
        print(f"{failed} of {len(candidates)} file(s) failed", file=sys.stderr)
    return 0 if failed == 0 and (decoded + rescued_count > 0 or already_done > 0) else 1

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.black_hz == args.white_hz:
        parser.error("--black-hz and --white-hz must differ")
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    if args.invert_iq and args.format == "wav":
        parser.error("--invert-iq requires --format iq or fm")
    if args.raw_format is not None and args.format == "wav":
        parser.error("--raw-format requires --format iq or fm")
    if (args.fm_bandwidth is not None and args.fm_bandwidth <= 0) or args.signal_seconds <= 0:
        parser.error("--fm-bandwidth and --signal-seconds must be positive")
    if args.raw_sample_rate < 1:
        parser.error("--raw-sample-rate must be positive")
    if args.jobs < 0:
        parser.error("--jobs cannot be negative (0 means one job per CPU core)")
    if args.find_start and args.force_start is not None:
        parser.error("--find-start and --force-start cannot be combined")
    if args.find_start and (args.start_ms is not None):
        parser.error("--find-start and --start-ms cannot be combined")
    if args.force_start == "auto" and args.input.is_dir():
        parser.error("--force-start auto applies to a single file; in directory mode failed files are retried automatically (see --no-rescue)")
    if args.force_start is not None:
        if args.force_start != "auto" and args.force_start < 0:
            parser.error("--force-start cannot be negative")
        if args.start_ms is not None:
            parser.error("--force-start and --start-ms cannot be combined")
        if args.input.is_dir():
            parser.error("--force-start applies to a single file, not a directory")

    if args.find_start:
        if args.input.is_dir():
            files = [
                path
                for path in _candidate_files(args.input, args.format, args.recursive)
                if path.suffix.lower() != ".wav" or _looks_like_wave(path) or args.raw_format is not None
            ]
        else:
            files = [args.input]
        return _run_find_start(args, files)

    if args.input.is_dir():
        return _run_directory(args, parser)

    if args.recursive or args.skip_existing or args.jobs != 1:
        parser.error(
            "--recursive, --skip-existing and --jobs only apply when the input is a directory"
        )
    if args.save_audio is not None and args.format != "fm":
        parser.error("--save-audio requires --format fm")
    try:
        output_path = _resolve_single_target(
            args.output, args.output_dir_hint, args.input, ".png"
        ) or _default_output(args.input)
        save_audio = _resolve_single_target(
            args.save_audio, args.save_audio_dir_hint, args.input, ".wav"
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return _decode_file(args, args.input, output_path, save_audio)


if __name__ == "__main__":
    raise SystemExit(main())
