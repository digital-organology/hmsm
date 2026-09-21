# Local browser roll player

The browser player digitizes piano roll scans in a background Python worker and
plays the recovered notes while the rest of the scan is still being decoded.
The scanned paper travels down past a fixed tracker bar. Detected note spans
are overlaid on their tracks; sounding notes and piano keys light up in gold.

## Run

Requires Python 3.11+, Node.js 20.19+ or 22.12+, and pnpm.
From the repository root, install the optional server dependencies:

```sh
pip install -e ".[web]"
```

For development, start these in two terminals:

```sh
hmsm-web
```

```sh
cd web
pnpm install
pnpm dev
```

Open the URL Vite prints (usually http://127.0.0.1:5173). Vite proxies `/api`
to the Python server at `127.0.0.1:8000`.

To serve a built frontend with just Python:

```sh
cd web
pnpm install
pnpm build
cd ..
hmsm-web --static-dir web/dist
```

Then visit http://127.0.0.1:8000. `--port` changes the Python port; when using
Vite, update its proxy target as well. The frontend is built from this checkout;
it is not included in the Python wheel.

## Using the player

1. Choose or drop an image, select its roll format, and set the marked roll
   tempo (60 means six feet per minute). Scan settings optionally override DPI
   or skip rows containing a problematic head.
2. Select **Digitize & load roll**. The upload progress is followed by decoding
   progress. Once enough rows are buffered, **Play** becomes available.
3. Play, pause, rewind, seek within the decoded region, or change playback speed
   with the brass tempo dial in feet per minute (ft/min), from one quarter to twice
   the marked tempo. Drag vertically or use the arrow keys for fine adjustment. Speed changes affect both sound and paper travel, preserving
   pitch. If playback catches up, it buffers and resumes when more rows arrive.
4. Download the MIDI once decoding finishes. **Close roll** cancels a running
   decoder and removes the session before another image can be opened. **Load different roll** opens a file
   picker and replaces the current session once a new image is selected.

Audio starts on an explicit Play click. Switching away from the browser tab
pauses playback. The browser uses an offline, additive piano-like synthesizer;
there are no external fonts, samples, or network services. The live preview
plays pitches and merged durations at fixed velocity. Printed dynamics and
pedal interpretation still need whole-sequence analysis and are included in the
final MIDI export, not in the browser synth. The tempo dial changes preview
speed only; the MIDI uses the marked roll tempo entered before digitization.

## Large scans and lifecycle

The browser sends the selected `File` directly, without converting it to a
base64 string or an in-memory array. The server streams the request to a
session file, with a 4 GiB upload limit. Digitization starts after the upload
finishes. Allow disk space for the upload plus JPEG previews in the OS temporary
directory (`TMPDIR` can select a different disk).

A single background worker processes one session at a time. Supported 8-bit RGB
strip/tile TIFFs retain the existing band-based decoding path, with a default
4000-row band and context margins. Other image formats, grayscale TIFFs and
unusual TIFF layouts still use the existing full-image fallback, which can
require substantially more RAM. The browser never decodes the full scan.

Each decoded band produces a JPEG tile at at most 1000 pixels wide. Tiles stay
on disk; the browser retains only tiles in its viewport. Paper edges sampled
every 32 scan rows align note overlays with lateral paper drift. The uploaded
scan is removed when processing finishes or fails; previews and MIDI last until
**Close roll** or server shutdown. Reloading the page restores the current local session; audio remains paused
until Play is clicked again.

The server binds to loopback, rejects non-local Host headers and cross-origin
requests, and does not accept arbitrary local file paths. It is intended for
one local user, not deployment on a shared host.

## Streaming contract

`RollDigitizer.run(..., on_update=callback)` calls back synchronously after each
band. A `RollUpdate` contains band pixels, absolute start/stop rows, paper edges,
a cumulative merged note snapshot and `safe_row`. Notes crossing bands may
extend in subsequent snapshots. Consumers must replace the previous snapshot,
keep absolute scan coordinates, and never play beyond `safe_row`. The watermark
withholds one band plus the merge gap. This avoids playing ahead of hole merging
and boundary-fragment detection. Profiles without a nominal hole width keep
the watermark at the start until completion because their merge threshold
depends on the whole scan. A callback may raise to cancel the run.

Pixels belong to the current processing band. Resize or persist them during the
callback; retaining them defeats bounded scan memory. The CLI path does not
create snapshots or previews. The final transcription retains `origin_row`,
which maps its rebased note table back to the original scan for visualization.

The server keeps only its latest snapshot. Versioned polling every 500 ms avoids
an event backlog if a browser is slow or disconnected. Notes and tile metadata
grow with musical content, not decoded pixel count. Snapshot merging scans all
notes recovered so far; it does not re-run segmentation.

## Checks

```sh
pip install -e ".[web,dev]" httpx
pytest
cd web
pnpm test
pnpm build
# With hmsm-web --static-dir web/dist running in another terminal:
pnpm exec playwright install chromium
node tests/browser.mjs
# Firefox compatibility and switching rolls:
pnpm exec playwright install firefox
HMSM_BROWSER=firefox node tests/browser.mjs
# Also asserts that playback becomes available before decoding completes:
node tests/browser.mjs ../assets/phonola.tif
```

The browser check writes desktop and mobile screenshots under `/tmp/hmsm-*.png`.
