"""One local digitization session with streamed uploads and disk-backed tiles.

Only the latest note snapshot is retained. The browser polls it, so a slow or
absent client cannot build an unbounded event queue. Full scan pixels never
cross the API: previews are produced from the decoder's existing bands.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from hmsm.io import open_source  # sets the OpenCV large-image limit before cv2
import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from hmsm.midi import MidiGenerator
from hmsm.profiles import available_presets, load_profile
from hmsm.rolls.digitizer import RollDigitizer

logger = logging.getLogger(__name__)
MAX_UPLOAD = 4 * 1024**3


class JobOptions(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    profile: str = "phonola"
    tempo: int = Field(default=60, ge=1, le=200)
    skip_rows: int = Field(default=0, ge=0)
    dpi: float | None = Field(default=None, gt=0, le=10000)


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, root: Path, options: JobOptions):
        self.id = uuid.uuid4().hex
        self.directory = root / self.id
        self.directory.mkdir()
        self.options = options
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.future = None
        self.closing = False
        self.state = dict(
            id=self.id,
            options=options.model_dump(),
            status="waiting",
            version=0,
            notes=[],
            tiles=[],
            safe_row=0,
            origin_row=None,
            rows_processed=0,
        )

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state["version"] += 1

    def snapshot(self):
        with self.lock:
            return self.state.copy()

    def process(self):
        try:
            self.update(status="processing")
            profile = load_profile(self.options.profile, "roll")
            with open_source(str(self.directory / "scan")) as source:
                dpi = self.options.dpi or source.dpi or 300
                self.update(
                    width=source.width,
                    height=source.height,
                    dpi=dpi,
                    tracks=profile.alignment_grid().tolist(),
                    seconds_per_row=MidiGenerator(
                        self.options.tempo, dpi
                    ).seconds_per_row,
                )
                if self.options.skip_rows >= source.height:
                    raise ValueError(
                        "Rows to skip must be smaller than the scan height."
                    )
                tiles = []

                def publish(update):
                    if self.cancel.is_set():
                        raise Cancelled()
                    scale = min(1, 1000 / source.width)
                    preview = cv2.resize(
                        update.pixels,
                        (
                            max(1, round(source.width * scale)),
                            max(1, round(len(update.pixels) * scale)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                    index = len(tiles)
                    ok = cv2.imwrite(
                        str(self.directory / f"{index}.jpg"),
                        cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 82],
                    )
                    if not ok:
                        raise OSError("Could not write preview tile")
                    edges = []
                    if update.edges is not None:
                        for i in range(0, len(update.edges), 32):
                            edges.append(
                                [
                                    update.start + i,
                                    float(update.edges.left[i]),
                                    float(update.edges.right[i]),
                                ]
                            )
                    tiles.append(
                        dict(
                            index=index,
                            start=update.start,
                            stop=update.stop,
                            edges=edges,
                        )
                    )
                    self.update(
                        notes=update.notes.tolist(),
                        tiles=list(tiles),
                        safe_row=update.safe_row,
                        rows_processed=update.stop,
                        origin_row=(
                            int(update.notes[:, 0].min()) if len(update.notes) else None
                        ),
                    )

                result = RollDigitizer(profile, dpi=dpi).run(
                    source,
                    skip_rows=self.options.skip_rows,
                    on_update=publish,
                )
                if self.cancel.is_set():
                    raise Cancelled()
                result.to_midi(self.options.tempo).write(
                    str(self.directory / "roll.mid")
                )
                notes = result.notes.copy()
                notes[:, :2] += result.origin_row
                self.update(
                    status="complete",
                    notes=notes.tolist(),
                    origin_row=result.origin_row,
                    safe_row=int(notes[:, 1].max()),
                    rows_processed=result.rows_processed,
                )
        except Cancelled:
            self.update(status="cancelled")
        except Exception as exc:
            logger.exception("Roll digitization failed")
            self.update(status="error", error=str(exc))
        finally:
            # Previews and MIDI suffice after decoding. Release the large upload.
            (self.directory / "scan").unlink(missing_ok=True)


def create_app(static_dir: Path | None = None):
    @asynccontextmanager
    async def lifespan(app):
        with tempfile.TemporaryDirectory(prefix="hmsm-web-") as directory:
            app.state.root = Path(directory)
            app.state.job = None
            app.state.pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="roll"
            )
            yield
            if app.state.job:
                app.state.job.cancel.set()
            await asyncio.to_thread(app.state.pool.shutdown, wait=True)

    app = FastAPI(title="HMSM roll player", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse(
                {"detail": "Cross-origin requests are not allowed"}, status_code=403
            )
        return await call_next(request)

    def get_job(identifier):
        job = app.state.job
        if job is None or job.id != identifier:
            raise HTTPException(404, "Unknown session")
        return job

    @app.get("/api/profiles")
    def profiles():
        return sorted(available_presets("roll"))

    @app.get("/api/session")
    def session():
        job = app.state.job
        return job.snapshot() if job else None

    @app.post("/api/jobs", status_code=201)
    async def create(options: JobOptions):
        if options.profile not in available_presets("roll"):
            raise HTTPException(422, "Choose a bundled roll profile")
        if app.state.job is not None:
            raise HTTPException(409, "Close the current roll before opening another")
        job = Job(app.state.root, options)
        app.state.job = job
        return job.snapshot()

    @app.put("/api/jobs/{identifier}/scan")
    async def upload(identifier: str, request: Request):
        job = get_job(identifier)
        if job.snapshot()["status"] != "waiting":
            raise HTTPException(409, "This session already has a scan")
        job.update(status="uploading")
        size = 0
        try:
            with (job.directory / "scan").open("wb") as handle:
                async for chunk in request.stream():
                    if job.cancel.is_set():
                        raise HTTPException(409, "Upload cancelled")
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "The local upload limit is 4 GiB")
                    await asyncio.to_thread(handle.write, chunk)
            if size == 0:
                raise HTTPException(400, "The selected file is empty")
            job.update(status="queued")
            job.future = app.state.pool.submit(job.process)
        except BaseException:
            job.update(
                status="error",
                error="Upload interrupted or rejected. Close this roll and retry.",
            )
            (job.directory / "scan").unlink(missing_ok=True)
            raise
        return {"id": job.id}

    @app.get("/api/jobs/{identifier}")
    def status(identifier: str, version: int = -1):
        snapshot = get_job(identifier).snapshot()
        return snapshot if snapshot["version"] != version else {"version": version}

    @app.get("/api/jobs/{identifier}/tiles/{index}.jpg")
    def tile(identifier: str, index: int):
        job = get_job(identifier)
        path = job.directory / f"{index}.jpg"
        if index < 0 or not path.is_file():
            raise HTTPException(404, "Tile is not available")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/jobs/{identifier}/midi")
    def midi(identifier: str):
        job = get_job(identifier)
        if job.snapshot()["status"] != "complete":
            raise HTTPException(409, "Digitization has not finished")
        return FileResponse(
            job.directory / "roll.mid", media_type="audio/midi", filename="roll.mid"
        )

    @app.delete("/api/jobs/{identifier}")
    async def close(identifier: str):
        job = get_job(identifier)
        if job.closing:
            raise HTTPException(409, "Session is already closing")
        if job.snapshot()["status"] == "uploading":
            job.cancel.set()
            raise HTTPException(
                409, "Stopping upload; retry closing when it has stopped"
            )
        job.closing = True
        job.cancel.set()
        if job.future:
            await asyncio.wrap_future(job.future)
        await asyncio.to_thread(shutil.rmtree, job.directory)
        if app.state.job is job:
            app.state.job = None
        return {"status": "closed"}

    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="player")
    return app


app = create_app()
