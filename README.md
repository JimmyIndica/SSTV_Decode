# robot36-decode v3.5.0

A Robot 36 colour SSTV decoder for Python. It decodes:

* ordinary **audio WAV** files (FM-demodulated SSTV audio),
* **I/Q baseband** files (SSB-style analytic signals, WAV or raw float32),
* **raw SDR FM captures** such as SDR# / SDR++ baseband recordings of a satellite downlink (for example the ISS SSTV downlink), at any sample rate, with dead air before and after the signal, a Doppler-shifted and drifting carrier, and noisy or fading signal.

It writes a 320x240 PNG, for one file or for every valid file in a folder. Typical decode time for a two-minute SDR capture is about 2.5 seconds on a laptop-class CPU, and a folder of captures can be decoded several files at a time (`-j`).

Robot 36 is the only SSTV mode supported (VIS code `0x08`: 320x240, 150 ms lines, 4:2:0 chroma).

---

## Contents

1. [Requirements and installation](#1-requirements-and-installation)
2. [Quick start](#2-quick-start)
3. [Input formats](#3-input-formats)
4. [Command-line reference](#4-command-line-reference)
5. [Reading the decoder output](#5-reading-the-decoder-output)
6. [Examples](#6-examples)
7. [Choosing the right options](#7-choosing-the-right-options)
8. [Troubleshooting](#8-troubleshooting)
9. [Performance and expected quality](#9-performance-and-expected-quality)
10. [How it works](#10-how-it-works)
11. [Tests](#11-tests)
12. [Package layout and using it as a library](#12-package-layout-and-using-it-as-a-library)
13. [Limitations](#13-limitations)
14. [Version history](#14-version-history)

---

## 1. Requirements and installation

**Python 3.10 or newer** (developed and tested on 3.12 and 3.13), plus two packages:

| Package | Used for |
|---|---|
| `numpy` | all signal processing |
| `scipy` | fast FFTs (`scipy.fft`); `scipy.ndimage` only for `--find-start` and rescue |

PNG output uses a built-in writer, so no imaging library (such as Pillow) is needed, for decoding or for the tests.

There is nothing to build. Unzip the package, install the two dependencies, and run the decoder from the unzipped folder (the one that contains the `robot36_decode` directory and this README).

### macOS / Linux (bash, zsh)

```bash
unzip robot36_decode_v3.5.0.zip -d robot36
cd robot36

python3 -m venv .venv                # optional but recommended
source .venv/bin/activate
python3 -m pip install numpy scipy

python3 -m robot36_decode --version
```

### Windows (Command Prompt)

```bat
:: unzip with Explorer ("Extract All...") or:
tar -xf robot36_decode_v3.5.0.zip -C robot36
cd robot36

py -3 -m venv .venv
.venv\Scripts\activate.bat
py -m pip install numpy scipy

py -m robot36_decode --version
```

### Windows (PowerShell)

```powershell
Expand-Archive robot36_decode_v3.5.0.zip -DestinationPath robot36
cd robot36

py -3 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
py -m pip install numpy scipy

py -m robot36_decode --version
```

On Windows the launcher is `py` (or `python` if it is on your PATH). On macOS/Linux use `python3`. Everywhere below, `python3 -m robot36_decode` and `py -m robot36_decode` are interchangeable.

> **Run it from the folder that contains `robot36_decode/`.** To run from anywhere, either set `PYTHONPATH` to that folder (macOS/Linux: `export PYTHONPATH=/path/to/robot36`; cmd: `set PYTHONPATH=C:\path\to\robot36`; PowerShell: `$env:PYTHONPATH="C:\path\to\robot36"`), or `pip install` the folder after adding your own `pyproject.toml`.

### Optional: a short command name

macOS/Linux, add to `~/.bashrc` or `~/.zshrc`:

```bash
alias robot36-decode='PYTHONPATH=$HOME/robot36 python3 -m robot36_decode'
```

Windows PowerShell, add to your `$PROFILE`:

```powershell
function robot36-decode { $env:PYTHONPATH="$HOME\robot36"; py -m robot36_decode @args }
```

Windows cmd: create `robot36-decode.bat` somewhere on your PATH containing
`@set PYTHONPATH=C:\path\to\robot36 && py -m robot36_decode %*`.

---

## 2. Quick start

| You have | Command |
|---|---|
| An SDR# / SDR++ baseband recording of the FM downlink (stereo I/Q WAV, any rate) | `python3 -m robot36_decode capture.wav --format fm` |
| FM-demodulated SSTV audio (e.g. from a receiver or soundcard) | `python3 -m robot36_decode audio.wav` |
| An I/Q file of SSTV audio (SSB-style, not FM) | `python3 -m robot36_decode iq.wav --format iq` |
| A raw `.iq` / `.raw` float32 file | `python3 -m robot36_decode file.iq --format iq --raw-sample-rate 96000` |
| A **folder** of captures | `python3 -m robot36_decode captures/ --format fm -o decoded/` |

The result is written next to the input as `<input name>.png` unless you pass `-o`.

> **Important:** a `.wav` extension tells you nothing about what is inside. SDR# baseband recordings are `.wav` files that contain *I/Q samples of an FM carrier*. Those need `--format fm`. Using the default (`--format wav`) on such a file will not produce an image.

---

## 3. Input formats

### `--format wav` (default): audio

A WAV file of FM-demodulated SSTV audio (the tones at 1200-2300 Hz). Any sample rate; 8/16/24/32-bit integer or 32/64-bit float; stereo is mixed down to mono. The audio is resampled to 48 kHz internally.

### `--format iq`: complex I/Q baseband of SSTV audio

For I/Q data where the instantaneous frequency of the complex signal *is* the SSTV tone (an SSB-style analytic signal), not an FM carrier. Accepted files:

* RIFF or RF64 WAV with **two channels** (channel 0 = I, channel 1 = Q), integer or float samples, any sample rate;
* WAV with **one channel**: converted to an analytic signal internally;
* **raw** interleaved little-endian **float32** I/Q (`.iq`, `.raw`, anything that is not a WAV), at the rate given by `--raw-sample-rate`.

A constant tuning offset and a drifting Doppler shift (a few hundred Hz, hundreds of Hz of sweep) are tracked automatically from the sync pulses.

If the picture comes out mirrored in colour or never syncs, the capture may have inverted spectrum: add `--invert-iq`.

### `--format fm`: raw FM baseband capture (SDR#/SDR++ recordings)

For recordings of an FM carrier. Works with stereo I/Q WAV (including large `RF64` files) and raw float32 I/Q, at any sample rate (tested at 78,125 Hz, 156,250 Hz). The decoder:

1. finds when the signal is present and where the carrier is (immune to fixed spurs and wideband bursts),
2. tracks the Doppler-shifted, drifting carrier,
3. trims the dead air before and after,
4. FM-demodulates with threshold extension (feedback) and click removal,
5. band-limits the audio to 1000-2500 Hz, resamples it to 48 kHz,
6. runs the normal VIS + sync-tracking decode.

Real passes are often incomplete, and the decoder copes with that:

* **Partial pictures.** A recording that starts after the picture began, or a signal that fades out or stops before it ends, gives the lines that were received in their right place; the rest are grey. With a VIS header, the header places the picture. Without one, the ISS's habit of keeping its carrier on for about a second after the last line tells which end is missing: if that quiet tail is in the capture, the lines are counted back from the last sync pulse; if the pulses run until the carrier fades, they are counted from where the pulse train begins.
* **Receiver dropouts.** Stretches where the SDR delivered near-silence (seconds at 20-30 dB below the noise) are neither mistaken for the signal nor allowed to split it; the lines that fall in one are grey.
* **Not every carrier is a picture.** Without a VIS header, a decode must look like Robot 36 (sync pulses on most lines, at the 1200 Hz sync tone, colour separators alternating) or the file is reported as `no Robot 36 picture found` instead of being written as a picture of noise.
* **Damaged headers.** A recording whose WAV header was overwritten (the file starts with 0xFF bytes) can be decoded as headerless data with `--raw-format s16 --raw-sample-rate <rate>`.

---

## 4. Command-line reference

```
robot36-decode [-h] [-o OUTPUT] [--start-ms MS] [--no-vis] [--no-sync]
               [--force-start SECONDS|auto] [--find-start]
               [--black-hz HZ] [--white-hz HZ] [--scale N] [--quiet]
               [-r] [--skip-existing] [--no-rescue] [-j N]
               [--format {wav,iq,fm}] [--signal-seconds S]
               [--fm-bandwidth HZ] [--repair-lines] [--save-audio WAV]
               [--raw-sample-rate HZ] [--raw-format {f32,s16}] [--invert-iq] [--version]
               input
```

Invoke it as `python3 -m robot36_decode ...` (macOS/Linux) or `py -m robot36_decode ...` (Windows).

### Positional argument

| Argument | Description |
|---|---|
| `input` | Input file (WAV, or raw float32 I/Q with `--format iq`/`fm`), **or a directory**: every valid input file in it is decoded (see [Directory input](#directory-input)). |

### General options

| Option | Default | Description |
|---|---|---|
| `-h`, `--help` | | Show the help text and exit. |
| `--version` | | Print the version (`robot36-decode 3.5.0`) and exit. |
| `-o OUTPUT`, `--output OUTPUT` | input name with `.png` | Output PNG path. An existing file is overwritten. The parent folder must exist. **A path ending in a slash (`out/`), or an existing directory, means a directory**: it is created if needed (including parents) and receives `<input name>.png`. With a directory input this is always the output *directory*. |
| `--scale N` | `1` | Integer nearest-neighbour upscale of the PNG. `--scale 2` writes 640x480, `--scale 3` 960x720. Does not change the decode, only the saved file. |
| `--quiet` | off | Print only errors. The exit status still reports success or failure. |

### Image location and timing

| Option | Default | Description |
|---|---|---|
| `--no-vis` | off | Ignore the VIS header and locate the image by tracking line sync pulses instead. Use when the header is missing, clipped or too noisy. |
| `--start-ms MS` | auto | Approximate time, in milliseconds from the start of the (trimmed) audio, of the **first line sync** pulse. Disables VIS detection. Sync tracking still refines the exact line positions. Cannot be negative. |
| `--force-start SECONDS` or `auto` | off | **Last resort for captures too poor to decode normally.** Take the first line sync as starting exactly `SECONDS` into the file (for `--format fm`, into the capture) and decode 240 fixed 150 ms lines with no VIS detection, no sync tracking and no signal search. Implies `--no-vis` and `--no-sync`; cannot be combined with `--start-ms` or a directory input. Lines beyond the end of the file are left grey. `auto` first finds the start, exactly as `--find-start` does. See [Forcing a decode](#forcing-a-decode-of-a-very-poor-capture). |
| `--find-start` | off | **Do not decode: report where the image starts.** Analyses the file and prints the time of the first line sync with a confidence rating and a ready-to-run `--force-start` command. With a directory input, prints one line per file. Cannot be combined with `--force-start` or `--start-ms`. See [Finding the start automatically](#finding-the-start-automatically). |
| `--no-sync` | off | Do not track sync pulses; assume perfectly aligned, fixed 150 ms lines. Useful for clean synthetic or studio-quality audio; do not use on real captures, where sound-card clocks and Doppler make line timing drift. |

### Finding the start automatically

`--find-start` answers "where does the image start in this file?" so you don't have to work it out by eye:

```bash
python3 -m robot36_decode poor_capture.wav --format fm --find-start
```

```
Input:      poor_capture.wav
Signal:     onset about 8.9 s
Line sync:  comb found, lines start at 0.1143 s + n x 0.150 s (strength 15x background)
Parity:     separator contrast +764 Hz (even lines 1500 Hz, odd lines 2300 Hz)
Best start: 10.1643 s  (from the VIS header)
Confidence: HIGH
Note:       start taken from the VIS header and refined with the sync comb

Try:  python3 -m robot36_decode "poor_capture.wav" --format fm --force-start 10.1643
Adjust by +/-0.150 s only if the picture looks wrong.
```

Nothing is decoded and no image is written. `--force-start auto` does the same search and then decodes from the result in one step. In a directory, `--find-start` prints one line per file:

```
HIGH      64.5954 s  captures/data03-33-54_437548418Hz.wav
HIGH      10.1643 s  captures/e0d7c36a-18-43-03_437544919Hz.wav
NONE              -  captures/pure_noise.raw
```

How it works, in the order the clues are used:

1. **Signal onset** from the carrier search (FM captures) or the signal energy (audio and I/Q).
2. **Sync comb.** Every line starts with a 9 ms, 1200 Hz sync pulse. Folding the "is this 1200 Hz?" indicator at the 150 ms line period piles every pulse on top of the others, so the line phase shows up as a sharp peak even when single pulses are lost in noise. This fixes where lines start to about a millisecond, but not which line is line 0. The strength is reported as a ratio to the background; below 5x nothing is found.
3. **Parity.** The 4.5 ms separator after each luma line is 1500 Hz on even lines and 2300 Hz on odd lines. Measuring it on each candidate shows whether the start is an even or an odd number of lines out, which is what decides the correct colours.
4. **Header pattern.** Line 0 is preceded by the VIS header: 600 ms of 1900 Hz leader, a 1200 Hz start bit two lines before and a 1200 Hz stop bit just before it. A readable VIS header decides directly. If it is unreadable, the same pattern (plus the step up in sync-pulse strength where the picture begins) chooses among the candidates the comb and parity allow.

Confidence ratings:

| Rating | Meaning |
|---|---|
| `HIGH` | A readable VIS header agrees with the sync comb, or the comb is at least 10x background, the parity contrast at least 400 Hz and one candidate clearly beats the rest. Start times were within 1 ms of the normal decoder's on every real capture used in development. |
| `MEDIUM` | A usable start, but the header or pulse-train start is not clear-cut. Neighbours two lines away score similarly; they are listed as `Next best`. |
| `LOW` | The separator parity or the start of the pulse train is unclear; the start may be a line out. Not used for automatic rescue. |
| `NONE` | No periodic sync pulses were found (pure noise, wrong `--format`, or no Robot 36 signal). |

### Forcing a decode of a very poor capture

Some captures are too broken for the automatic steps: the signal fades out for seconds at a time (antenna mispointing), a spur or burst competes with it, or the header is lost. The automatic search then finds only part of the transmission and the decode fails with `audio is too short...`, or the sync tracker gives up. `--force-start` skips every automatic step and decodes whatever is there, from a start time you give:

```bash
python3 -m robot36_decode poor_capture.wav --format fm --force-start 10.1646
```

What it does: for `--format fm` it estimates the carrier (and its Doppler drift) only from the stretch from 1.5 s before to 0.5 s after the 36 s image, using the clear parts of the signal and a straight line (or a gentle curve when the clear parts span most of the window) to carry it across fades. It then demodulates that stretch and decodes 240 lines of exactly 150 ms each from your start time, with the spec timing and no tracking. Plain audio and I/Q files are decoded the same way from the given time. If the file ends before the 240th line, the missing lines are grey.

What it cannot do: it cannot make noise into a picture. Where the signal was gone you get colour noise. It also gives up the corrections tracking normally makes, so a start time that is off shifts the picture instead of being corrected.

**Finding the start time.** Use `--force-start auto`, or run `--find-start` first and copy the command it prints. To work it out by hand instead: the value is the time of the first line sync pulse (the start of line 0), in seconds from the start of the file:

1. *Estimate from the signal onset.* The VIS header lasts about 0.91 s, so the first line starts about **0.91 s after the signal begins**. The onset is visible in an SDR# or SDR++ waterfall, or in the located window that a failed normal decode prints (`The signal was located at ... of the capture`).
2. *Refine by looking at the result.* Decode, then adjust:
   * Picture shifted **sideways**: move the start by the shift. One pixel is 0.275 ms; later start moves the picture left. Accuracy of about 1 ms (about 4 pixels) is good; the image error grows in proportion to the miss.
   * **Colours wrong** (skin tones blue, red looks blue): the start is an odd number of lines off, which swaps the even/odd chroma lines. Move it by 0.150 s, try one step later, and compare.
   * Picture shifted **vertically** by whole rows: move by multiples of 0.150 s.
3. Lines repeat every 150 ms, so once one start gives a clean sideways alignment you only need to find the right line by stepping 0.150 s at a time.

Example from the poor captures used in development (`e0d7c36a-18-43-03_437544919Hz.wav`, 51 s, 78,125 Hz, a fixed spur at +32.5 kHz, the signal fading between 26 and 43 s): versions before 3.5.0 refused it with `audio is too short` (since 3.5.0 the normal decode keeps the received part of the picture by itself), but

```bash
python3 -m robot36_decode e0d7c36a-18-43-03_437544919Hz.wav --format fm --force-start 10.1646 -o e0d7c36a.png
```

recovers roughly the top 40% of the picture (the signal was lost below that); `--force-start 10.0146` gives the same picture one line early, with the chroma lines swapped.

### Tone-to-brightness mapping

| Option | Default | Description |
|---|---|---|
| `--black-hz HZ` | `1500` | Tone frequency mapped to black (level 0). |
| `--white-hz HZ` | `2300` | Tone frequency mapped to white (level 255). |

The defaults are the Robot 36 specification. Change them only if your transmitter is off-spec; the luminance and chroma then re-scale linearly. (A constant frequency *offset* from tuning or Doppler is corrected automatically from the sync pulses, so this is not how you fix that.)

### Input format selection

| Option | Default | Description |
|---|---|---|
| `--format {wav,iq,fm}` | `wav` | `wav`: real audio. `iq`: complex I/Q whose frequency is the SSTV tone. `fm`: raw FM baseband recording (see [section 3](#3-input-formats)). |
| `--raw-sample-rate HZ` | `48000` | Sample rate of **raw** (headerless) I/Q data (`.iq`, `.raw`, or a WAV whose header is damaged). WAV files carry their own rate, so this is ignored for them. |
| `--raw-format {f32,s16}` | `f32` | Sample type of raw (headerless) I/Q data: `f32` interleaved 32-bit float, `s16` interleaved 16-bit integer, as SDR# and SDR++ record. With `s16`, `.wav` files in a folder that have no WAV header are decoded as raw data instead of being skipped. `iq` and `fm` formats only. |
| `--invert-iq` | off | Conjugate the I/Q samples (swap the sign of Q), for captures with inverted spectrum. `iq` and `fm` formats only. |

### FM capture options (`--format fm` only)

| Option | Default | Description |
|---|---|---|
| `--signal-seconds S` | `38.0` | Approximate length of the transmission. Used to trim to the right stretch if a long noisy region surrounds the signal. A Robot 36 transmission with its VIS header and tail is about 38 s. Must be positive. |
| `--fm-bandwidth HZ` | auto | **Total** channel filter width around the tracked carrier. Auto-selected between about 10 kHz (weak signals) and 14 kHz (strong signals) from the measured signal strength. Set it manually only to experiment; must be positive. |
| `--save-audio WAV` | none | Also write the demodulated, trimmed **48 kHz audio** as a 16-bit WAV. Handy for listening to the capture, checking it in another SSTV program, or reprocessing with `--format wav`. A path ending in a slash (or an existing directory) receives `<input name>.wav`; **with a directory input this is an output directory**, receiving one WAV per capture. |

### Image repair

| Option | Default | Description |
|---|---|---|
| `--repair-lines` | off | Replace image lines wrecked by interference or deep fades with the average of their neighbours, but only lines that differ strongly from two agreeing neighbours. **This modifies image data**, so it is off by default. |

### Directory input

Give a folder instead of a file and every valid input file in it is decoded, one after another, with the same options.

| Option | Default | Description |
|---|---|---|
| `-r`, `--recursive` | off | Also process files in subfolders. With `-o`, the subfolder layout is mirrored under the output folder. |
| `--skip-existing` | off | Skip files whose output PNG (or rescued `<name>_forced.png`) already exists, so an interrupted batch can be resumed or new captures added to a folder cheaply. |
| `-j N`, `--jobs N` | 1 | Decode N files at the same time, one per CPU core; `0` uses every core. Each job needs about 0.6 GB of memory per two minutes of 78 kHz capture, so on a laptop with 8 GB keep N at 4 or less. The pictures are identical to a one-at-a-time run. |
| `--no-rescue` | off | Do not retry files that fail to decode with an automatically found start (see below). |

How a folder is processed:

* **Which files:** with `--format wav` (default) only `*.wav`; with `--format iq` or `fm` also `*.iq` and `*.raw`. Extensions are case-insensitive. Hidden files and folders (names starting with `.`), other file types and subfolders (unless `-r`) are ignored silently.
* **Valid WAVs only:** a `.wav` that does not start with a RIFF/RF64 `WAVE` header is reported as `skipped` and does not count as a failure.
* **One format per run:** `--format` and all other options apply to every file in the folder. Do not mix SDR baseband captures and plain audio in one folder; the ones that do not match will fail (and be listed as failed) while the rest still decode.
* **Where output goes:** by default each PNG is written next to its input. With `-o DIR`, all PNGs go under `DIR` (created if needed); with `--save-audio DIR`, the audio WAVs likewise. Existing PNGs are overwritten.
* **Name collisions:** if `pass.wav` and `pass.iq` would both produce `pass.png`, they are written as `pass_wav.png` and `pass_iq.png`.
* **Failures don't stop the run:** a file that cannot be decoded prints its `error:` message and the next file is processed.
* **Automatic rescue:** a file that fails normal decoding is not given up on straight away. The decoder looks for where its image starts ([`--find-start`](#finding-the-start-automatically)) and, if it finds a start with HIGH or MEDIUM confidence, retries with `--force-start` at that time. The result is written as `<name>_forced.png` (so it can be told from a normal decode) and listed as `rescued`, with the start time and confidence, in the summary. (Recordings that merely start late or stop early no longer need this: since 3.5.0 they decode normally, with grey for the missing lines. Rescue is for what is left, such as a signal present for only a handful of lines, or one without a readable header whose line numbering is unclear.) Files with no reliable start (pure noise, wrong format) stay `FAILED`. Use `--no-rescue` to turn this off. A rescued picture may still be mostly noise where the signal was lost; check it.
* **Order:** files are processed in case-insensitive name order, a numbered `[3/12] path` header before each. With `-j`, each file's messages are printed together when it finishes, so the order on screen follows completion; the summary table is always in name order.
* **Summary:** a final table lists every file as `ok`, `rescued`, `FAILED` or `skipped` with its output path or reason. With `--quiet` the table is omitted and only errors are printed.
* **Partial pictures and refusals:** recordings that start late or stop early are decoded normally (grey where nothing was received), so they no longer need a rescue. Carriers that hold no Robot 36 are reported `FAILED` with `no Robot 36 picture found`. Header-damaged `.wav` files are listed as `skipped` with a hint; add `--raw-format s16 --raw-sample-rate <rate>` to decode them.
* **Exit status:** `0` when no file failed and at least one was decoded (rescued files count as decoded) (or already existed with `--skip-existing`); `1` if any file failed, if nothing could be decoded, or if the folder holds no candidate files.
* **Safety checks:** `-o` and `--save-audio` must be directories (or not exist yet) in directory mode, `--save-audio` still needs `--format fm`, and `--save-audio` refuses to run if any audio file would overwrite one of your inputs. `-r`, `--skip-existing` and `-j` are errors with a single-file input (and `--save-audio` without `--format fm` is an error there too).

### Exit status and messages

| Code | Meaning |
|---|---|
| `0` | An image was decoded and written. |
| `1` | Error (unreadable file, no signal found, not Robot 36, invalid option value). The message goes to standard error as `error: ...`. In directory mode: any file failed, or nothing could be decoded. |

Argument-syntax errors (unknown option, missing input, invalid combinations) exit with argparse's status `2`.

---

## 5. Reading the decoder output

Normal (non-quiet) output of a successful decode of an SDR capture:

```
Input:  1794236581785_26OCT_IQData.wav
Audio:  48000 Hz, mono, 16-bit FM-demodulated 78125 Hz baseband, 41.588 s
FM:     signal 24.117 s - 62.705 s of the capture (38.6 s); carrier +14.55 -> +14.19 kHz from centre (Doppler -9.3 Hz/s)
Image:  starts at 25.792 s in the capture
VIS:    code 0x08 (Robot 36), start bit at 2.880 s
Sync:   235/240 lines tracked, drift -0.006 ms/line
Start:  3.175 s (VIS header + sync tracking)
Note:   2 of 240 line separators did not match the expected alternating colour order.
Decoded: 320x240 RGB
Output: 1794236581785_26OCT_IQData.png
```

| Line | Meaning |
|---|---|
| `Audio` | Description of what was read and the trimmed 48 kHz audio length. |
| `FM` | (fm only) Where the signal was found in the capture, and the carrier offset from the tuned centre at the start and end, with the average Doppler rate. |
| `Image` | (fm only) Where the picture starts in the original capture. |
| `VIS` | The detected header: `0x08` is Robot 36. Other codes (for example Robot 72) are rejected. If no header is found the decoder falls back to sync tracking and says so on the `Start` line. |
| `Sync` | Lines whose sync pulse was found, out of 240. A good capture tracks 225-240. Lines that are not found are placed by interpolation. `drift` is clock/Doppler time-scaling per line. |
| `Start` | How the first line was located: VIS header + sync tracking, or `--start-ms` / sync only. |
| `Note` | Warnings such as separator mismatches (a symptom of noise or a few damaged lines; counted over the lines that were recorded) or a tone offset that was corrected. For a partial picture: `the recording starts N lines into the image`, `the recording ends after line N of 240`, `N lines fall in a receiver dropout`; those lines are grey. |

Weak signals print fewer tracked lines (about 150 for a barely-above-noise capture) and a noisy picture; this is the limit of the signal, not a failure of the tool.

---

## 6. Examples

The examples below mirror the captures used to develop and test this version. File names are from those captures; substitute your own.

### 6.1 A normal SDR# FM baseband capture

The 26 October capture (RF64 stereo WAV, 78,125 Hz sample rate, about 80 s long, ISS downlink with Doppler from +14.5 kHz down):

macOS / Linux:

```bash
python3 -m robot36_decode 1794236581785_26OCT_IQData.wav --format fm
```

Windows (cmd or PowerShell):

```bat
py -m robot36_decode 1794236581785_26OCT_IQData.wav --format fm
```

Writes `1794236581785_26OCT_IQData.png` next to the input; takes about 6 seconds.

### 6.2 Choose the output name and folder

macOS / Linux:

```bash
mkdir -p decoded
python3 -m robot36_decode captures/data11-13-57_437548418Hz.wav --format fm -o decoded/pass_1113.png
```

Windows cmd:

```bat
mkdir decoded
py -m robot36_decode captures\data11-13-57_437548418Hz.wav --format fm -o decoded\pass_1113.png
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force decoded | Out-Null
py -m robot36_decode captures\data11-13-57_437548418Hz.wav --format fm -o decoded\pass_1113.png
```

Paths containing spaces must be quoted on every platform: `-o "decoded/ISS pass 1.png"`.

Trailing separators are fine on every platform: `captures/`, `captures//` and `-o decoded/` all work, and on Windows so does `captures\`. One Windows quirk is handled for you: cmd.exe and PowerShell 5 read the `\"` in `"my captures\"` as an escaped quote and pass `my captures"`; the decoder strips that stray quote and treats the path as a directory. (Using a forward slash, `"my captures/"`, avoids the quirk entirely.)

### 6.3 Decode a whole folder of captures

Point the decoder at the folder; every valid capture in it is decoded and a summary is printed.

macOS / Linux:

```bash
python3 -m robot36_decode captures/ --format fm -o decoded/
```

Windows (cmd or PowerShell):

```bat
py -m robot36_decode captures\ --format fm -o decoded\
```

Example summary for the five captures used while developing this tool (the ISS passes plus the 26OCT file):

```
[1/5] captures/1794236581785_26OCT_IQData.wav
...
Summary: 5 decoded, 0 failed, 0 skipped of 5 file(s) in captures
  ok       1794236581785_26OCT_IQData.wav  decoded/1794236581785_26OCT_IQData.png
  ok       data03-31-46_437548418Hz.wav    decoded/data03-31-46_437548418Hz.png
  ok       data03-33-54_437548418Hz.wav    decoded/data03-33-54_437548418Hz.png
  ok       data11-13-57_437548418Hz.wav    decoded/data11-13-57_437548418Hz.png
  ok       data16-58-07_437548418Hz.wav    decoded/data16-58-07_437548418Hz.png
```

Useful variations:

```bash
python3 -m robot36_decode captures/ --format fm                       # PNGs beside the captures
python3 -m robot36_decode captures/ --format fm -r -o decoded/        # include subfolders, mirror layout
python3 -m robot36_decode captures/ --format fm -o decoded/ --skip-existing   # resume / add new files only
python3 -m robot36_decode captures/ --format fm -o decoded/ --save-audio audio/   # also keep the demodulated audio
python3 -m robot36_decode captures/ --format fm --quiet               # errors only; check the exit status
python3 -m robot36_decode captures/ --format fm -o decoded/ -j 0      # decode several files at once (all cores)
python3 -m robot36_decode recordings/ -o decoded/ --scale 2           # a folder of plain audio WAVs
python3 -m robot36_decode raw_iq/ --format iq --raw-sample-rate 96000 -o decoded/   # .wav, .iq and .raw files
```

Folder processing needs no shell loop, so it behaves the same in bash, zsh, cmd and PowerShell. If you do want one command per file (for example to give each file different options), a loop still works:

macOS / Linux (bash/zsh):

```bash
mkdir -p decoded
for f in captures/*.wav; do
  python3 -m robot36_decode "$f" --format fm --quiet -o "decoded/$(basename "${f%.wav}").png" \
    && echo "ok   $f" || echo "FAIL $f"
done
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force decoded | Out-Null
Get-ChildItem captures\*.wav | ForEach-Object {
  py -m robot36_decode $_.FullName --format fm --quiet -o "decoded\$($_.BaseName).png"
  if ($LASTEXITCODE -eq 0) { "ok   $($_.Name)" } else { "FAIL $($_.Name)" }
}
```

Windows cmd:

```bat
mkdir decoded
for %f in (captures\*.wav) do @(py -m robot36_decode "%f" --format fm --quiet -o "decoded\%~nf.png" && echo ok   %f || echo FAIL %f)
```

(In a `.bat` file, double the percent signs: `%%f`, `%%~nf`.)

### 6.4 The four ISS passes (`data03-33-54`, `data11-13-57`, `data16-58-07`, `data03-31-46`)

These are all 437.548 MHz SDR# captures at various gains and Doppler offsets; the same command works for all of them:

```bash
python3 -m robot36_decode data03-33-54_437548418Hz.wav --format fm
python3 -m robot36_decode data11-13-57_437548418Hz.wav --format fm
python3 -m robot36_decode data16-58-07_437548418Hz.wav --format fm
python3 -m robot36_decode data03-31-46_437548418Hz.wav --format fm
```

Expected results from v3.1 (lines tracked of 240 / damaged rows):

| Capture | Lines tracked | Damaged rows | Notes |
|---|---|---|---|
| `1794236581785_26OCT_IQData` | 235 | 2 | Strong, clean; matches an independent decoder's image (luma error about 6.5 grey levels) |
| `data03-33-54` | 235 | 11 | Strong |
| `data11-13-57` | 237 | 27 | Two fades of about 7 dB (antenna pointing) |
| `data16-58-07` | 229 | 22 | Strong |
| `data03-31-46` | 147 | 150 | Very weak, about 1 dB above noise; picture is noisy but correctly placed |

### 6.5 A weak or fading capture

Weak pictures are noise-limited, but a few options help diagnose or tidy them.

Listen to what the decoder heard (and open it in another program if you like):

```bash
python3 -m robot36_decode data03-31-46_437548418Hz.wav --format fm --save-audio weak_audio.wav
```

Decode that audio again with sync-only line location if the header was lost:

```bash
python3 -m robot36_decode weak_audio.wav --no-vis
```

Replace lines destroyed by a fade or burst with the average of their neighbours (this changes the image, so it is optional):

```bash
python3 -m robot36_decode data11-13-57_437548418Hz.wav --format fm --repair-lines -o data11_repaired.png
```

Force a decode of a capture the automatic search gives up on (see [Forcing a decode](#forcing-a-decode-of-a-very-poor-capture) for how to find the start time):

```bash
python3 -m robot36_decode poor_capture.wav --format fm --force-start 10.1646
```

Try neighbouring starts (one line is 0.150 s) if the colours come out wrong:

```bash
python3 -m robot36_decode poor_capture.wav --format fm --force-start 10.0146 -o try_a.png
python3 -m robot36_decode poor_capture.wav --format fm --force-start 10.1646 -o try_b.png
```

Let the decoder find the start itself, then decode from it:

```bash
python3 -m robot36_decode poor_capture.wav --format fm --find-start          # report only
python3 -m robot36_decode poor_capture.wav --format fm --force-start auto    # find, then decode
```

A folder with a mix of good and poor captures: poor ones are rescued automatically and written as `*_forced.png`:

```bash
python3 -m robot36_decode captures/ --format fm -o decoded/
python3 -m robot36_decode captures/ --format fm --find-start                 # just list the start of every file
```

A recording whose WAV header was destroyed (the folder summary lists it as `skipped ... no WAV header (the file starts with ... 0xFF bytes)`): the samples are usually intact, so read them as headerless 16-bit I/Q at your SDR's sample rate:

```bash
python3 -m robot36_decode damaged.wav --format fm --raw-format s16 --raw-sample-rate 78125
python3 -m robot36_decode captures/ --format fm -r --raw-format s16 --raw-sample-rate 78125   # whole folder
```

### 6.6 A long file with a lot of noise around the signal

If the capture contains several minutes of noisy data and the detector picks the wrong stretch, say how long the transmission really is:

```bash
python3 -m robot36_decode long_capture.wav --format fm --signal-seconds 36
```

### 6.7 Plain audio WAV (FM-demodulated SSTV)

```bash
python3 -m robot36_decode sstv_audio.wav
python3 -m robot36_decode sstv_audio_44k1.wav -o picture.png --scale 2
```

Any sample rate works (44.1 kHz, 22.05 kHz, 48 kHz...); the audio is resampled internally.

### 6.8 Audio that is missing its header

If the VIS header was cut off, or the recording started in the middle of a transmission:

```bash
python3 -m robot36_decode partial.wav --no-vis
```

If you know roughly where the first line sync is (here at 3.2 s):

```bash
python3 -m robot36_decode partial.wav --start-ms 3200
```

### 6.9 I/Q files (SSB-style baseband)

Stereo I/Q WAV, any sample rate:

```bash
python3 -m robot36_decode sstv_iq.wav --format iq
```

Inverted spectrum:

```bash
python3 -m robot36_decode sstv_iq.wav --format iq --invert-iq
```

Raw float32 I/Q at 96 kHz:

```bash
python3 -m robot36_decode capture.iq --format iq --raw-sample-rate 96000
```

Mono WAV treated as I/Q (converted to an analytic signal internally):

```bash
python3 -m robot36_decode mono_sstv.wav --format iq
```

Raw FM baseband (not WAV) at 156.25 kHz:

```bash
python3 -m robot36_decode fm_capture.raw --format fm --raw-sample-rate 156250
```

### 6.10 Clean synthetic audio

For a perfectly clocked, noise-free file (for instance one generated by the test encoder), skip sync tracking:

```bash
python3 -m robot36_decode test_tone.wav --no-sync
```

### 6.11 Enlarge the output for viewing

```bash
python3 -m robot36_decode capture.wav --format fm --scale 3     # 960x720 PNG
```

### 6.12 Scripts and batch jobs

Quiet mode prints nothing on success; use the exit status:

macOS/Linux:

```bash
if python3 -m robot36_decode capture.wav --format fm --quiet; then echo decoded; else echo failed; fi
```

PowerShell:

```powershell
py -m robot36_decode capture.wav --format fm --quiet
if ($LASTEXITCODE -ne 0) { "decode failed" }
```

cmd:

```bat
py -m robot36_decode capture.wav --format fm --quiet || echo decode failed
```

### 6.13 Check the version

```bash
python3 -m robot36_decode --version      # robot36-decode 3.5.0
```

---

## 7. Choosing the right options

```
Is it an SDR# / SDR++ baseband recording of the FM carrier?  ----> --format fm
Is it a stereo/complex file where frequency = SSTV tone?     ----> --format iq   (+ --invert-iq if needed)
Is it ordinary audio of the SSTV tones?                      ----> (default)

Header missing or unreadable?                                ----> --no-vis   (or --start-ms)
Capture too poor for the automatic search to decode at all? ----> --find-start, then --force-start SECONDS (or --force-start auto)
Studio-clean synthetic audio?                                ----> --no-sync
Picture speckled by fades or bursts, and you accept
modifying the image?                                         ----> --repair-lines
Want to hear or reuse the demodulated audio?                 ----> --save-audio out.wav
```

Many files in one folder?  ----------------------------------> pass the folder (add -r for subfolders)

Leave `--fm-bandwidth`, `--black-hz`, `--white-hz` alone unless you have a specific reason.

---

## 8. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `error: the capture is only ... s long ... nothing to decode` | The recording is under 2 seconds, usually because it was stopped early or truncated (for example a 200 kB stereo file at 78,125 Hz is only 0.64 s). A Robot 36 transmission takes about 38 s. |
| `error: internal error (...)` | A bug, not a problem with your command. In a folder run the file is marked `FAILED` and the batch carries on. Please keep the file; setting the environment variable `R36_DEBUG=1` (`export R36_DEBUG=1`, `set R36_DEBUG=1` in cmd, `$env:R36_DEBUG=1` in PowerShell) prints the full traceback instead. |
| `error: no FM signal found in the capture` | `--format fm` could not find a carrier. Check that the file really is an FM baseband recording and that it contains the signal. |
| `error: VIS header reports mode code 0x.., which is not Robot 36` | A different SSTV mode (for example Robot 72). Only Robot 36 is supported. Use `--no-vis` or `--start-ms` only if you are sure it is Robot 36. (A "header" that is really picture content, which earlier versions misread as code 0x00, is no longer reported: a header must have real 1200 Hz start/stop bits and 1100/1300 Hz data bits.) |
| `error: only N of the 240 image lines are in the audio` | Less than 20 lines of the picture were received (the signal was only briefly there). For an FM capture the message names the located stretch; `--force-start` can still decode what there is. |
| `error: no Robot 36 picture found: ...` | There is a carrier but it carries something else (voice, another mode, interference) or nothing readable: without a VIS header the decode did not look like Robot 36 (too few sync pulses, a "sync tone" far from 1200 Hz, or colour separators in random order, as noise gives). In directory mode the file is still offered to the automatic rescue. |
| `skipped ... no WAV header (the file starts with ... 0xFF bytes)` | The recording's header was overwritten, typically when the SDR program crashed or was closed while recording. The samples after it are usually intact: decode with `--format fm --raw-format s16 --raw-sample-rate 78125` (your SDR's sample rate). |
| A file shows as `FAILED` in directory mode although you expect a picture | Rescue found no reliable start (HIGH or MEDIUM). Run `--find-start` on that file to see why; LOW means the start may be a line out and you can try the listed alternatives with `--force-start`. |
| `error: no FM carrier found in the forced window` | `--force-start` was given but there is no clear carrier within 1.5 s before to 0.5 s after the 36 s image. Check `--format fm` and the start time. |
| `error: --force-start ... is beyond the end of the file` | The start time is after the end of the recording. |
| `error: the located image (...) lies beyond the end of the audio` | The located start is past the end of the data, usually from a wrong `--format` or `--start-ms`. |
| `warning: ... this WAV has 2 channels at 78125 Hz, which looks like an SDR I/Q recording` | The file was read as plain audio (the default `--format wav`), but a stereo WAV at a rate no sound card uses is almost certainly an SDR baseband recording. Add `--format fm` (satellite FM downlink, e.g. the ISS) or `--format iq`. The same note is added to any error for such a file. |
| Decodes to a garbage or empty picture, no error | Wrong `--format` for the file. SDR# `.wav` baseband recordings need `--format fm`; FM-demodulated audio needs the default. If the header is missing or unreadable, try `--no-vis`. |
| `No module named robot36_decode` | You are not in the folder that contains `robot36_decode/`, or `PYTHONPATH` is not set. |
| `No module named numpy` / `scipy` | `python3 -m pip install numpy scipy` (Windows: `py -m pip install numpy scipy`). Activate your venv first if you use one. |
| `python3` not found on Windows | Use `py -m robot36_decode`. |
| PowerShell refuses to activate the venv | `Set-ExecutionPolicy -Scope Process RemoteSigned`, then activate again. |
| Image is noisy but the right way round | Signal near or below the FM threshold (weak or fading). Nothing in the demodulator can recover it; improve the antenna pointing or gain. `--repair-lines` can hide a few broken lines. |
| Horizontal bands of speckle for 1-3 seconds | A fade (antenna off-pointing). See above. |
| Colours wrong or image never syncs on an I/Q file | Try `--invert-iq`. |
| Wrong part of a long recording decoded | Set `--signal-seconds` close to the real transmission length. |
| Out-of-memory on a very large capture | The decoder holds the file in memory (complex64). Trim the recording first, or use a machine with more RAM; a 2-minute 78 kHz stereo capture needs about 0.6 GB. With `-j`, every job needs that much at the same time: lower N. A job that runs out of memory is reported as `a decoding process stopped unexpectedly`. |
| Output PNG not where expected | It defaults to the input's folder and name with `.png`. Use `-o`. |

---

## 9. Performance and expected quality

On the seven real captures used during development (2-core machine, 78 kHz, about two minutes each):

| Stage | Time |
|---|---|
| Start Python and load the decoder | about 0.3 s |
| Read and locate the signal | about 0.3 s |
| Demodulation (carrier tracking, 6 feedback passes) | about 1.5 s |
| VIS, sync tracking, image decode | about 0.4 s |
| **Total per two-minute capture** | **about 2.5 s** (3.4.2: 4.5-5.5 s) |

With `-j N` a folder decodes up to N times faster on an N-core machine, memory permitting. FFTs use all cores already, so the gain from `-j` is largest on machines with many cores.

Picture quality is limited by signal-to-noise ratio. FM demodulation has a threshold near 10 dB in-band SNR below which noise grows rapidly. v3.1's feedback demodulator extends that threshold: on a synthetic 3 dB signal it cuts image error from 62 to 35 grey levels, and on a 5 dB signal from 40 to 18, compared with a conventional demodulator. Fades of 7 dB or more in a signal that is already near threshold cannot be recovered by any demodulator and show up as speckled bands in the image.

---

## 10. How it works

1. **Locate** (`fmdemod.locate_fm_signal`): a time/frequency power map and a sliding 12 kHz band score find when the FM signal is present and where its carrier is, ignoring spurs and wideband bursts. The noise level comes from the quiet part of the capture or, for a capture trimmed close to the transmission, from the bands beside the signal. Receiver dropouts (frames 20x below the typical power) are set aside: they neither count as signal nor split a transmission.
2. **Track the carrier**: a robust cubic polynomial through the carrier centroid track, refined twice from the discriminator's own residual mean per half second. The final curve is the Doppler shift.
3. **Demodulate with feedback**: the carrier is mixed to 0 Hz and channel-filtered, then FM-discriminated. An annealed series of six feedback passes re-mixes the signal with its own audio estimate so that progressively narrower filters (half-bandwidth 4200 Hz down to 2400 Hz) can be used, which pushes the FM threshold lower. Spikes from phase slips ("clicks") beyond what the filter can pass are interpolated out.
4. **Audio clean-up**: moving-average DC removal, a 1000-2500 Hz band-pass (covers VIS 1100/1300 Hz, sync 1200 Hz and picture 1500-2300 Hz), resample to 48 kHz.
5. **VIS detection** (`vis`): a phase-based detector, with a spectral matched-template fallback for weak signals, verifies the 300/10/300 ms leader, the start bit and the 8 data bits (logic 1 = 1100 Hz, logic 0 = 1300 Hz, LSB first). Robot 36 is code `0x08`. The first 15 s are searched; in a longer audio recording with no header there, it is searched for again just before the signal energy and the sync pulse train begin.
6. **Sync tracking** (`synctrack`): finds the 9 ms 1200 Hz line syncs with porch verification (3 ms at 1500 Hz) and pulse-grid agreement, fits the line positions robustly, and tolerates clock error and Doppler drift. Per-line tone offsets handle a drifting carrier.
7. **Line numbering without a header** (FM): from the end of the sync pulse train when the carrier's quiet tail after the last line is in the capture, otherwise from where the pulse train begins (`findstart`). A picture that began before the recording, or ends after it, keeps the lines that exist; the rest are grey.
8. **Plausibility** (no VIS header): sync pulses on at least 35% of the recorded lines and, for FM audio (which has no Doppler left), a sync tone within 60 Hz of 1200 Hz; for other input, and whenever there is a header but no usable line sync, at most 40% of the colour separators out of order (noise gets about half wrong). Otherwise `no Robot 36 picture found`.
9. **Image decode** (`robot36`): Y at 320 pixels over 88 ms, alternating Cr/Cb at 160 pixels over 44 ms with 4.5 ms separators (1500 Hz even / 2300 Hz odd lines), 4:2:0 chroma upsampled and converted to RGB over the full 0-255 range.

---

## 11. Tests

The `tests` folder holds a synthetic Robot 36 encoder (`synth.py`) and a test runner with 85 tests: round trips for audio, I/Q (raw, WAV, mono, inverted) and FM baseband at various sample rates; Doppler tracking (offset, rate, acceleration, clock error); noise and burst robustness; VIS detection; and, when the real captures are present, quality limits on each capture.

Run from the package folder:

macOS / Linux:

```bash
python3 tests/test_roundtrip.py
```

Windows:

```bat
py tests\test_roundtrip.py
```

The suite takes about two minutes. There is no pytest requirement; the file runs itself and prints a table of results and `85/85 tests passed`. Test files go in one temporary folder that is deleted at the end of the run (set `R36_KEEP_TMP=1` to keep it for inspection).

The real-capture tests look for the capture files in the folder named by the `R36_CAPTURE_DIR` environment variable (default `/mnt/user-data/uploads`) and are skipped if it does not exist. Set it to your own folder:

```bash
R36_CAPTURE_DIR=$HOME/captures python3 tests/test_roundtrip.py          # macOS / Linux
```

```bat
set R36_CAPTURE_DIR=C:\captures && py tests\test_roundtrip.py           :: Windows cmd
```

```powershell
$env:R36_CAPTURE_DIR="C:\captures"; py tests\test_roundtrip.py          # PowerShell
```


---

## 12. Package layout and using it as a library

```
robot36_decode/
  __main__.py    python -m robot36_decode entry point
  cli.py         argument parsing, single-file decode, directory batch mode
  constants.py   Robot 36 timing and frequencies, VIS code
  wav.py         audio WAV reader
  iq.py          I/Q WAV (RIFF/RF64) and raw float32 reader
  fmdemod.py     FM signal location, Doppler tracking, feedback demodulation
  resample.py    rational resampling
  filters.py     FIR design, FFT convolution, polyphase resampling (numpy + scipy.fft)
  dsp.py         analytic-signal phase, band limiting, signal location
  vis.py         VIS header detection (phase detector + spectral fallback)
  synctrack.py   line-sync tracking and per-line offsets
  findstart.py   automatic start-of-image search (--find-start, rescue)
  robot36.py     image decode, optional line repair
  image.py       PNG writer (no external dependency)
tests/
  synth.py, test_roundtrip.py, imgmetrics.py
```

Minimal use from Python:

```python
from robot36_decode.cli import main

rc = main(["capture.wav", "--format", "fm", "-o", "pass.png", "--quiet"])
```

The tuning constants for the FM stage live at the top of `fmdemod.py`; `FEEDBACK_SCHEDULE` is the list of `(half-bandwidth Hz, audio low-pass Hz)` pairs used by the feedback passes, and `demodulate_fm(..., feedback_schedule=None)` disables the feedback stage.

---

## 13. Limitations

* **Robot 36 only.** Other modes (Robot 72, Martin, Scottie, PD) are not decoded.
* **Fades below the FM threshold are not recoverable.** Antenna mispointing that drops the signal to about 3 dB SNR or lower produces speckled lines. Narrowing filters or tweaking the schedule makes that worse, not better; the fix is on the capture side.
* The signal location assumes one transmission per file (about 38 s). Recordings containing several transmissions decode the strongest one.
* Whole files are read into memory (in directory mode one file at a time).
* Directory mode applies one `--format` and one option set to every file.
* `--find-start` needs at least a few seconds of periodic sync pulses; with a signal that is only present for the VIS header or a handful of lines it reports NONE. Its start can be a whole line out (LOW/MEDIUM) when the header and pulse train are both unclear.
* `--force-start` decodes what is there but cannot recover a picture from stretches where the signal is gone, and it needs a start time accurate to about 1 ms.
* A partial picture without a VIS header is placed from the sync pulse train. When neither the header nor the carrier's tail after the last line is in the capture (it starts late *and* fades early), there is no way to tell which line came first, and the picture is placed as if its first received line were near the top.
* `--repair-lines` is the only option that edits image data; everything else is signal processing before the picture is formed.

---

## 14. Version history

* **3.4.1** Fixed a crash (`IndexError` in the FM signal locator) on captures shorter than about one second, such as a 200 kB stub; such files now report `the capture is only ... s long`. Unexpected errors on one file in a folder run are reported as `internal error (...)` for that file and no longer stop the batch (`R36_DEBUG=1` shows the traceback). Decoding of normal files is unchanged. 67 tests.
* **3.4.2** Fixed the green stripe on the right edge of the picture (and a milder one on the left). The last chroma pixel of every line ends exactly where the next 1200 Hz sync pulse begins, and the demodulator and 1-2.5 kHz band-pass smear that step over about half a millisecond, so the last pixel(s) were averaged with the sync tone; sync decodes as chroma zero, which is green. The last luma pixels likewise picked up the 2300 Hz separator on odd lines. Edge pixels are now measured over the nearest clean stretch of the same signal (no interpolation, no filtering, timing unchanged); the rest of the picture is bit-identical. 70 tests.
* **3.4.3** Speed and loose ends. Decoding is about 1.8x faster (a two-minute FM capture from the command line: 4.5-5.5 s down to about 2.5 s; measured in one process, 3.1-4.8 s down to 2.1-2.9 s per capture): the decoder no longer imports `scipy.signal` (about 1 s at start-up) but uses its own FIR design, FFT convolution and polyphase resampler (`filters.py`, checked against SciPy), uses `scipy.fft` instead of `numpy.fft` (6-15x faster on the odd transform lengths of real recordings), computes the VIS tone energies with one matrix product, skips an energy search the FM path never used, and reads I/Q files without intermediate copies. Pictures differ from 3.4.2 by at most 1-2 levels in a few dozen pixels (rounding). New `-j/--jobs` decodes several files of a folder in parallel. Fixes: a VIS header more than 15 s into an audio recording is now found (the image start was 6-11 lines off); FM captures trimmed to within a few seconds of the transmission now decode (only half the signal was located); a stereo WAV at an SDR sample rate read without `--format` gets a hint to use `--format fm`; `--save-audio` without `--format fm` is an error for a single file too (was ignored); `--skip-existing` also skips files rescued earlier (`<name>_forced.png`); RF64 files take the data size from `ds64`. The tests now delete their temporary files (a full run used to leave several GB behind). The reference-image benchmark test and its JPEG (`tests/data/`) are gone, so nothing needs Pillow any more; the real-capture quality limits still check the 26OCT capture. 79 tests.
* **3.5.0** Tested against a folder of 65 real ISS captures full of traps (empty recordings, header-only files, dropouts, damaged headers, partial passes); a folder run now gives 21 pictures (19 before, 10 of them only by rescue), all in the right place and none of them a picture of pure noise, plus a 22nd from a header-damaged file with `--raw-format s16`. Changes: a VIS "header" must have real header tones (picture content was read as "mode 0x00" and stopped seven files); recordings that start late or stop early decode as partial pictures with grey for the missing lines, numbered from the VIS header, or without one from the sync pulse train (the carrier's ~1 s quiet tail after the last line tells which end is missing); without a VIS header a decode must look like Robot 36 (sync pulses, 1200 Hz sync tone, alternating colour separators) or it is refused as `no Robot 36 picture found`; a sync track far off the 1200 Hz tone is not trusted for FM audio; receiver dropouts no longer pose as the signal or split it, their lines are grey and the carrier fit skips them; lines outside the carrier are grey; `--raw-format s16` reads headerless 16-bit I/Q, and header-damaged WAVs in a folder are skipped with that hint; separator mismatches are counted over the recorded lines only (and grey lines no longer vote on the colour order). 85 tests.
* **3.4.0** `--find-start` (report where the image starts, with a confidence rating), `--force-start auto`, and automatic rescue of failed files in directory mode (`*_forced.png`, `--no-rescue` to disable). Start search uses the sync comb, separator parity, a VIS-header template and the VIS decoder when readable; within 1 ms of the normal decoder's start on all five real captures, with or without the VIS reader. Normal decoding is unchanged. 65 tests.
* **3.3.1** Path arguments: trailing `/` (and `\`) are handled everywhere. `-o dir/` and `--save-audio dir/` with a single input now write `<dir>/<input name>.png`/`.wav` (creating the directory) instead of a file named `dir`; the Windows `"folder\"` quoting quirk is handled. 58 tests.
* **3.3.0** `--force-start SECONDS`: forced decode of very poor captures from a given start time, with no VIS detection, no sync tracking and no signal search; windowed carrier estimate for FM captures; truncated files padded with grey lines; the "audio too short" error for FM captures now names the located stretch and suggests the option. Normal decoding is unchanged. 55 tests.
* **3.2.0** Directory input: pass a folder to decode every valid file in it (`-r`, `--skip-existing`, per-file error handling, summary table, mirrored output folders). Decoding itself is unchanged from 3.1.0. 50 tests.
* **3.1.0** Annealed six-pass feedback schedule (half-bandwidth 4200 to 2400 Hz) replaces four fixed 2400 Hz passes; 3 dB synthetic error 62 to 37 and 5 dB 40 to 18; weak real capture chroma noise down about 28%. `demodulate_fm` takes `feedback_schedule` in place of `feedback_passes`. 44 tests.
* **3.0.0** Feedback (threshold-extension) FM demodulation, click removal, output band-pass; real-capture quality equal to an independent decoder on the benchmark capture.
* **2.x** Raw SDR FM baseband support: signal location, Doppler tracking, spectral VIS fallback, RF64 reading.
* **1.x** Fixed bugs in the original module (VIS polarity, sync argument, level clipping, hard-coded 48 kHz); added IQ input and Doppler tracking.
