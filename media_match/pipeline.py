"""Parallel scan pipeline: prepare frames on CPU, embed them on the GPU.

The Phase-1 scan prepared one file and then embedded one batch, so the GPU sat
idle while Python decoded images and hashed frames (the "spikes and valleys").
Here the CPU-heavy work (open, decode, pHash, CLIP preprocessing, ffmpeg) runs
on a thread pool that keeps the embedding queue full, while the main thread
drains that queue in GPU batches. Images and video keyframes share the same
queue, so both kinds of media fill the same pipeline.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from .clip_processor import process_image
from .config import is_ignored
from .db import Database
from .exif import read_exif
from .hashing import hash_to_hex, phash
from .video import VideoMeta, extract_frames, probe, sample_count

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
    ".tif", ".tiff", ".heic", ".heif",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv",
}


@dataclass
class ScanJob:
    kind: str  # "image" | "video"
    path: str
    size: int
    mtime: float
    prev_description: str = ""
    prev_keywords: str = ""


@dataclass
class ScanResult:
    job: ScanJob
    sha1: str
    width: int
    height: int
    phashes: list[str]
    arrays: list[np.ndarray]
    timestamps: list[float] | None = None
    video_meta: VideoMeta | None = None
    signature: str = ""
    vectors: list[np.ndarray] = field(default_factory=list)
    captured_at: str = ""
    camera: str = ""
    gps_lat: float | None = None
    gps_lon: float | None = None


@dataclass
class ScanStats:
    """Measured cost of one scan run, accumulated into the index for `stats`.

    ``gpu`` is the time spent inside CLIP embedding forward passes (a good
    proxy for GPU-busy time); ``cpu`` is process CPU time (includes decoding,
    hashing and the DB writes done from the result callback).
    """

    items: int = 0
    frames: int = 0
    bytes_read: int = 0
    wall: float = 0.0
    cpu: float = 0.0
    gpu: float = 0.0


def _sha1_of(path: str) -> str:
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clear_mlx_cache() -> None:
    """Return MLX's cached (unused) memory to the OS; best effort."""
    try:
        import mlx.core as mx

        (mx.clear_cache if hasattr(mx, "clear_cache") else mx.metal.clear_cache)()
    except (ImportError, AttributeError):
        pass


def iter_media(root: str, include_video: bool, *,
                ignore: list[str] | None = None) -> list[tuple[str, str]]:
    """(kind, path) for every image (and, optionally, video) under root.

    ``ignore`` is a list of .media-matchignore globs; matching paths are left
    out of the index entirely.
    """
    ignore = ignore or []
    paths: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            # skip dotfiles: macOS .trashed-*, AppleDouble ._*, the index itself
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            path = os.path.join(dirpath, name)
            if ignore and is_ignored(os.path.relpath(path, root), ignore):
                continue
            if ext in IMAGE_EXTENSIONS:
                paths.append(("image", path))
            elif include_video and ext in VIDEO_EXTENSIONS:
                paths.append(("video", path))
    return paths


def build_jobs(
    root: str,
    db: Database,
    *,
    include_video: bool,
    reanalyze: bool,
    limit: int | None = None,
    ignore: list[str] | None = None,
) -> tuple[list[ScanJob], int, int]:
    """Return (jobs, total, up_to_date) after applying the mtime/size check."""
    media = iter_media(root, include_video, ignore=ignore)
    if limit:
        media = media[:limit]
    jobs: list[ScanJob] = []
    up_to_date = 0
    for kind, path in media:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        existing = db.get_file(path) if kind == "image" else db.get_video(path)
        unchanged = (
            existing is not None
            and existing["size"] == stat.st_size
            and abs(existing["mtime"] - stat.st_mtime) < 1e-6
        )
        if unchanged and not reanalyze:
            up_to_date += 1
            continue
        jobs.append(
            ScanJob(
                kind=kind,
                path=path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                prev_description=existing["description"] if unchanged else "",
                prev_keywords=existing["keywords"] if unchanged else "",
            )
        )
    return jobs, len(media), up_to_date


def _under(path: str, root: str, prefix: str) -> bool:
    return path == root or path.startswith(prefix)


def prune_missing(root: str, db: Database, *, include_video: bool,
                  ignore: list[str] | None = None) -> int:
    """Drop index rows for media under ``root`` no longer present on disk.

    Scoped to ``root`` so scanning a subtree never touches the rest of the
    index, and skipped entirely when ``root`` itself is gone, so an unmounted
    or renamed drive cannot wipe the index. Paths matched by
    ``.media-matchignore`` are treated as absent. Returns rows removed.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return 0
    prefix = root + os.sep
    ignore = ignore or []
    file_rows, video_rows = db.all_paths()

    def _stale(rows: list) -> list[int]:
        out = []
        for r in rows:
            if not _under(r["path"], root, prefix):
                continue
            if not os.path.exists(r["path"]):
                out.append(r["id"])
            elif ignore and is_ignored(os.path.relpath(r["path"], root), ignore):
                out.append(r["id"])
        return out

    stale_files = _stale(file_rows)
    stale_videos: list[int] = []
    if include_video:
        stale_videos = _stale(video_rows)

    if stale_files:
        db.delete_files(stale_files)
    if stale_videos:
        db.delete_videos(stale_videos)
    return len(stale_files) + len(stale_videos)


def load_image_tolerant(source):
    """Open and fully decode an image, tolerating truncated files.

    ``source`` is anything ``Image.open`` accepts — a path or a binary stream
    (the staged pipeline reads bytes into its ring buffer first). Returns a
    loaded PIL.Image; the caller owns it and must ``close()`` it. Raises
    OSError/ValueError if the file cannot be identified or decoded at all,
    which means corrupt rather than merely truncated.

    Pillow's warnings (truncated reads, malformed MPO, decompression bombs) are
    captured rather than printed: they carry a site-packages path and no
    filename, which is noise in a scan of thousands of files. ``diagnose.py``
    turns the same warnings into a sentence the CLI can print per file.
    """
    import warnings

    from PIL import Image, ImageFile

    def _open():
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            # The flag is global, but racing the other worker threads is
            # benign: tolerance only affects files that are already truncated.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = Image.open(source)
                img.load()
            return img
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous

    try:
        return _open()
    except (OSError, IOError, ValueError):
        pass
    # the file may be truncated; retry with PIL's truncation tolerance
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return _open()
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def _open_image_tolerant(path: str):
    """``load_image_tolerant`` for a path on disk (kept for callers/tests)."""
    return load_image_tolerant(path)


def prepare(job: ScanJob, interval: float, max_frames: int) -> ScanResult:
    """CPU-only work for one file: hash, decode, perceptual-hash, preprocess."""
    from PIL import Image

    sha1 = _sha1_of(job.path)
    if job.kind == "image":
        img = _open_image_tolerant(job.path)
        try:
            width, height = img.size
            meta = read_exif(img)
            frame_hash = hash_to_hex(phash(img))
            array = process_image(img.convert("RGB"))
        finally:
            img.close()
        return ScanResult(
            job=job, sha1=sha1, width=width, height=height,
            phashes=[frame_hash], arrays=[array], signature=frame_hash,
            captured_at=meta["captured_at"], camera=meta["camera"],
            gps_lat=meta["gps_lat"], gps_lon=meta["gps_lon"],
        )

    meta = probe(job.path)
    count = sample_count(meta.duration, interval=interval, max_frames=max_frames)
    frames = extract_frames(job.path, meta, count=count)
    phashes: list[str] = []
    arrays: list[np.ndarray] = []
    timestamps: list[float] = []
    for t, arr in frames:
        img = Image.fromarray(arr)
        phashes.append(hash_to_hex(phash(img)))
        arrays.append(process_image(img))
        timestamps.append(t)
    return ScanResult(
        job=job, sha1=sha1, width=meta.width, height=meta.height,
        phashes=phashes, arrays=arrays, timestamps=timestamps,
        video_meta=meta, signature=min(phashes) if phashes else "",
    )


# Sentinel a CPU worker puts on the batch queue when it has finished, so the
# main loop can tell "nothing left" from "nothing has arrived yet".
_DONE = object()


@dataclass
class RingBuffer:
    """A bounded, back-pressured queue of raw file bytes.

    Reader threads push ``(path, bytes)`` chunks; CPU workers pull them. The
    total buffered bytes is capped at ``max_bytes`` so the buffer never grows
    beyond ~1 GB. Producers block when full, consumers block when empty.
    """

    max_bytes: int = 1 << 30  # 1 GB

    def __post_init__(self) -> None:
        import queue

        self._q: queue.Queue = queue.Queue()
        self._bytes = 0
        self._lock = __import__("threading").Lock()
        self._not_full = __import__("threading").Condition(self._lock)
        self._not_empty = __import__("threading").Condition(self._lock)
        self._done = False

    def put(self, item: tuple[str, bytes]) -> None:
        data = item[1]
        with self._not_full:
            while self._bytes + len(data) > self.max_bytes and not self._done:
                self._not_full.wait()
            self._q.put(item)
            self._bytes += len(data)
            self._not_empty.notify()

    def get(self) -> tuple[str, bytes]:
        with self._not_empty:
            while self._q.empty() and not self._done:
                self._not_empty.wait()
            if self._q.empty():
                raise EOFError
            item = self._q.get_nowait()
            self._bytes -= len(item[1])
            self._not_full.notify()
            return item

    def close(self) -> None:
        with self._lock:
            self._done = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    @property
    def closed(self) -> bool:
        return self._done

    def __len__(self) -> int:
        return self._q.qsize()

    def fill(self) -> tuple[int, int]:
        """(bytes currently buffered, capacity) — for the progress display.

        Read under the lock so the renderer never sees a half-updated pair.
        A full buffer means the readers are ahead of the CPU workers and disk
        throughput is not the bottleneck; an empty one means they are not.
        """
        with self._lock:
            return self._bytes, self.max_bytes


def run_staged_scan(
    jobs: list[ScanJob],
    encoder,
    *,
    interval: float,
    max_frames: int,
    batch_size: int = 16,
    workers: int = 4,
    io_workers: int = 2,
    ring_bytes: int = 1 << 30,
    on_result=None,
    progress=None,
    on_error=None,
    stats: ScanStats | None = None,
    telemetry=None,
) -> int:
    """Four-stage pipeline: Read → Preprocess → Embed → Store.

    A ring buffer of raw file bytes (``ring_bytes`` cap) is always backfilled
    by ``io_workers`` reader threads, so CPU workers never wait on disk. CPU
    workers decode, pHash and preprocess into numpy arrays; the main thread
    batches arrays and runs the GPU embedding. Results stream to ``on_result``.

    Returns the number of results emitted.
    """
    t_wall = time.perf_counter()
    t_cpu = time.process_time()
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    reader_count = max(1, io_workers)
    emitted = 0
    clear_every = 200
    ring = RingBuffer(max_bytes=ring_bytes)
    batch_queue: queue.Queue = queue.Queue(maxsize=batch_size * 2)
    results_queue: queue.Queue = queue.Queue()

    # -- stage 1: reader threads fill the ring buffer from disk
    #
    # Only the *last* reader to finish closes the ring. Closing it from each
    # reader's own finally block was a silent data race: with io_workers > 1
    # the first thread to finish set _done, so a CPU worker that had not yet
    # pulled its item saw an empty-but-closed buffer and raised EOFError,
    # dropping the file without an error. The scan just under-counted.
    readers_left = reader_count
    readers_lock = threading.Lock()
    # One shared cursor, not one copy of the list per reader: with more than
    # one reader thread every reader used to walk the whole list, so each file
    # was read and decoded reader_count times.
    cursor = iter(jobs)
    cursor_lock = threading.Lock()

    def next_job():
        with cursor_lock:
            return next(cursor, None)

    def reader() -> None:
        nonlocal readers_left
        try:
            while True:
                job = next_job()
                if job is None:
                    return
                if telemetry is not None:
                    telemetry.note_reading(job.path)
                try:
                    with open(job.path, "rb") as fh:
                        data = fh.read()
                    ring.put((job.path, data))
                    if telemetry is not None:
                        telemetry.note_read(len(data))
                except OSError:
                    continue
                finally:
                    if telemetry is not None:
                        telemetry.note_finished(job.path)
        finally:
            with readers_lock:
                readers_left -= 1
                last = readers_left == 0
            if last:
                ring.close()

    # -- stage 2: CPU workers decode/preprocess from the ring buffer
    def cpu_worker() -> None:
        while True:
            try:
                path, data = ring.get()
            except EOFError:
                # Signal that this worker is finished. It goes *after* every
                # result this worker produced, so the main loop can rely on
                # FIFO order to know that no result is still in flight.
                batch_queue.put(_DONE)
                return
            # reconstruct a ScanJob from the path
            job = next((j for j in jobs if j.path == path), None)
            if job is None:
                continue
            try:
                if telemetry is not None:
                    telemetry.note_decoding(path)
                    telemetry.note_cpu(time.process_time() - t_cpu)
                result = _prepare_bytes(job, data, interval=interval,
                                        max_frames=max_frames)
                batch_queue.put(result)
            except Exception as exc:  # noqa: BLE001
                if on_error:
                    on_error(job.path, exc)
                if telemetry is not None:
                    telemetry.note_error()
            finally:
                if telemetry is not None:
                    telemetry.note_finished(path)

    # -- stage 3: main thread batches and embeds on the GPU
    readers = [threading.Thread(target=reader) for _ in range(reader_count)]
    worker_count = max(1, workers)
    cpus = ThreadPoolExecutor(max_workers=worker_count)
    cpu_futures = [cpus.submit(cpu_worker) for _ in range(worker_count)]

    for t in readers:
        t.start()

    pending: dict[object, ScanResult] = {}
    owners: dict[object, list] = {}
    remaining: dict[object, int] = {}
    batch: list[tuple[int, int, np.ndarray]] = []
    next_owner = 0
    jobs_iter = iter(jobs)
    job_by_path = {j.path: j for j in jobs}

    def flush() -> None:
        nonlocal emitted, batch
        if not batch:
            return
        arrays = [arr for _, _, arr in batch]
        vectors = None
        if stats is not None:
            stats.gpu += 0.0  # measured below
        t_gpu = time.perf_counter()
        try:
            vectors = encoder.embed_images(arrays)
        except Exception as exc:  # noqa: BLE001
            if on_error:
                on_error("<batch>", exc)
        if stats is not None:
            stats.gpu += time.perf_counter() - t_gpu
        if telemetry is not None:
            telemetry.note_gpu(time.perf_counter() - t_gpu)
        completed: list[int] = []
        for k, (owner, frame, _) in enumerate(batch):
            if vectors is not None:
                owners[owner][frame] = vectors[k]
            remaining[owner] -= 1
            if remaining[owner] == 0:
                completed.append(owner)
        batch = []
        for owner in completed:
            emit(owner)

    def emit(owner: int) -> None:
        nonlocal emitted
        result = pending.pop(owner)
        result.vectors = owners.pop(owner)
        remaining.pop(owner, None)
        emitted += 1
        if stats is not None:
            # run_scan has always accounted bytes here; the staged pipeline did
            # not, so 'stats' reported 0B read for every scan since it shipped.
            stats.frames += len(result.arrays)
            stats.bytes_read += result.job.size
        if emitted % clear_every == 0:
            _clear_mlx_cache()
        if on_result is not None:
            on_result(result)
        if telemetry is not None:
            telemetry.note_completed()
            telemetry.note_cpu(time.process_time() - t_cpu)
        if progress is not None:
            progress(emitted, len(jobs), result.job.path)

    # Feed batch_queue into the GPU loop.
    #
    # Termination keys off the CPU workers' _DONE sentinels, not off the reader
    # threads being finished. A reader finishing says nothing about whether a
    # worker is still mid-decode, and the old `any(t.is_alive() for t in
    # readers)` check exited early and silently dropped results in flight. Each
    # worker queues its sentinel after its last result, so FIFO order means all
    # worker_done sentinels seen => every result has already arrived.
    workers_done = 0
    while True:
        try:
            if telemetry is not None:
                # Sample the ring here rather than from a worker: this is the
                # one place that is never blocked, so the fill level we show
                # is current instead of a stale snapshot.
                used, cap = ring.fill()
                telemetry.set_ring(used, cap)
            item = batch_queue.get(timeout=0.5)
        except queue.Empty:
            if workers_done >= worker_count:
                break  # every worker is finished and the queue is empty
            continue
        if item is _DONE:
            workers_done += 1
            # The last worker is done, so nothing more can arrive: embed
            # whatever partial batch is left over rather than waiting for it
            # to reach batch_size, which would hang forever on a short tail.
            if workers_done >= worker_count and batch:
                flush()
            continue
        owner = next_owner
        next_owner += 1
        pending[owner] = item
        owners[owner] = [None] * len(item.arrays)
        remaining[owner] = len(item.arrays)
        for frame, array in enumerate(item.arrays):
            batch.append((owner, frame, array))
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()

    cpus.shutdown(wait=True)
    for t in readers:
        t.join()

    if stats is not None:
        stats.items = emitted
        stats.wall = time.perf_counter() - t_wall
        stats.cpu = time.process_time() - t_cpu
    return emitted


def _prepare_bytes(job: ScanJob, data: bytes, *, interval: float,
                   max_frames: int) -> ScanResult:
    """Prepare a job from raw bytes (no disk read)."""
    import io

    sha1 = hashlib.sha1(data).hexdigest()
    if job.kind == "image":
        # Tolerate truncation here too: the staged pipeline replaced this with
        # a raw Image.open(io.BytesIO(...)) and silently dropped every
        # truncated file, which the sequential path had always indexed.
        img = load_image_tolerant(io.BytesIO(data))
        with img:
            img.load()
            width, height = img.size
            meta = read_exif(img)
            frame_hash = hash_to_hex(phash(img))
            array = process_image(img.convert("RGB"))
        return ScanResult(
            job=job, sha1=sha1, width=width, height=height,
            phashes=[frame_hash], arrays=[array], signature=frame_hash,
            captured_at=meta["captured_at"], camera=meta["camera"],
            gps_lat=meta["gps_lat"], gps_lon=meta["gps_lon"],
        )
    # video: write bytes to a temp file for ffmpeg
    import tempfile

    # This branch needs PIL in its own scope: the image branch above goes
    # through load_image_tolerant and no longer imports it. Dropping this
    # import made every video fail with NameError on Image.fromarray.
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(job.path)[1],
                                     delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        meta = probe(tmp_path)
        count = sample_count(meta.duration, interval=interval, max_frames=max_frames)
        frames = extract_frames(tmp_path, meta, count=count)
    finally:
        os.unlink(tmp_path)
    phashes: list[str] = []
    arrays: list[np.ndarray] = []
    timestamps: list[float] = []
    for t, arr in frames:
        img = Image.fromarray(arr)
        phashes.append(hash_to_hex(phash(img)))
        arrays.append(process_image(img))
        timestamps.append(t)
    return ScanResult(
        job=job, sha1=sha1, width=meta.width, height=meta.height,
        phashes=phashes, arrays=arrays, timestamps=timestamps,
        video_meta=meta, signature=min(phashes) if phashes else "",
    )


def run_scan(
    jobs: list[ScanJob],
    encoder,
    *,
    interval: float,
    max_frames: int,
    batch_size: int = 16,
    workers: int = 4,
    on_result=None,
    progress=None,
    on_error=None,
    stats: ScanStats | None = None,
) -> int:
    """Prepare jobs concurrently and embed their frames in GPU batches.

    Results are streamed to ``on_result`` as soon as all of a file's frames are
    embedded, so memory stays bounded on large trees (a video's 120 keyframe
    vectors are dropped right after it is stored). Returns the number of results
    emitted.

    A failed embedding batch leaves the affected frames' vectors as ``None`` but
    keeps *positional* alignment: ``result.vectors[i]`` always belongs to
    ``result.phashes[i]`` / ``result.timestamps[i]``.

    When ``stats`` is given it is filled with the measured cost of this run
    (items, frames, bytes read, and wall/CPU/GPU seconds).
    """
    t_wall = time.perf_counter()
    t_cpu = time.process_time()
    results: dict[int, ScanResult] = {}
    owners: dict[int, list[np.ndarray | None]] = {}
    remaining: dict[int, int] = {}
    batch: list[tuple[int, int, np.ndarray]] = []
    next_owner = 0
    emitted = 0
    clear_every = 200  # return the MLX cache to the OS periodically

    def emit(owner: int) -> None:
        nonlocal emitted
        result = results.pop(owner)
        result.vectors = owners.pop(owner)
        remaining.pop(owner, None)
        emitted += 1
        if emitted % clear_every == 0:
            _clear_mlx_cache()
        if on_result is not None:
            on_result(result)

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        arrays = [arr for _, _, arr in batch]
        vectors = None
        t_gpu = time.perf_counter()
        try:
            vectors = encoder.embed_images(arrays)
        except Exception as exc:  # noqa: BLE001 - report, drop this batch
            if on_error:
                on_error("<batch>", exc)
        if stats is not None:
            stats.gpu += time.perf_counter() - t_gpu
        completed: list[int] = []
        for k, (owner, frame, _) in enumerate(batch):
            if vectors is not None:
                owners[owner][frame] = vectors[k]
            remaining[owner] -= 1
            if remaining[owner] == 0:
                completed.append(owner)
        batch = []
        for owner in completed:
            emit(owner)

    prepare_one = partial(prepare, interval=interval, max_frames=max_frames)
    job_of: dict[object, ScanJob] = {}
    pending: set = set()
    # keep only a bounded number of prep jobs in flight so a huge tree does not
    # spawn one Future per file up front. It must hold at least a couple of
    # batches or the GPU stalls waiting for the queue to refill.
    window = max(max(1, workers) * 4, batch_size * 2)

    def fill(iterator) -> None:
        while len(pending) < window:
            try:
                job = next(iterator)
            except StopIteration:
                return
            future = pool.submit(prepare_one, job)
            job_of[future] = job
            pending.add(future)

    def handle(result: ScanResult) -> None:
        nonlocal next_owner
        if stats is not None:
            stats.frames += len(result.arrays)
            stats.bytes_read += result.job.size
        owner = next_owner
        next_owner += 1
        results[owner] = result
        owners[owner] = [None] * len(result.arrays)
        remaining[owner] = len(result.arrays)
        if not result.arrays:
            emit(owner)
            return
        for frame, array in enumerate(result.arrays):
            batch.append((owner, frame, array))
            if len(batch) >= batch_size:
                flush()
        # the processed arrays are only needed once they are in the batch
        # queue; drop them so memory stays bounded on large trees
        result.arrays = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        iterator = iter(jobs)
        fill(iterator)
        done = 0
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                job = job_of.pop(future)
                done += 1
                if progress:
                    progress(done, len(jobs), job.path)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - keep scanning past bad files
                    if on_error:
                        on_error(job.path, exc)
                    continue
                handle(result)
            fill(iterator)
        flush()

    if stats is not None:
        stats.items = emitted
        stats.wall = time.perf_counter() - t_wall
        stats.cpu = time.process_time() - t_cpu
    return emitted


def store_result(db: Database, result: ScanResult) -> None:
    """Write one prepared+embedded result to the index.

    ``result.vectors`` is positionally aligned with the frames; entries may be
    ``None`` when a batch failed, in which case the frame is stored without an
    embedding rather than shifting the rest.
    """
    job = result.job
    if job.kind == "image":
        embedding = None
        if result.vectors and result.vectors[0] is not None:
            embedding = result.vectors[0].tobytes()
        db.upsert_file(
            path=job.path, size=job.size, mtime=job.mtime, sha1=result.sha1,
            width=result.width, height=result.height, phash=result.phashes[0],
            description=job.prev_description, keywords=job.prev_keywords,
            embedding=embedding,
            captured_at=result.captured_at, camera=result.camera,
            gps_lat=result.gps_lat, gps_lon=result.gps_lon,
        )
        return

    video_id = db.upsert_video(
        path=job.path, size=job.size, mtime=job.mtime, sha1=result.sha1,
        duration=result.video_meta.duration, width=result.width,
        height=result.height, fps=result.video_meta.fps,
        codec=result.video_meta.codec, bitrate=result.video_meta.bitrate,
        signature=result.signature, description=job.prev_description,
        keywords=job.prev_keywords,
    )
    frames = []
    for i, t in enumerate(result.timestamps or []):
        phash_hex = result.phashes[i] if i < len(result.phashes) else ""
        vector = result.vectors[i] if i < len(result.vectors) else None
        frames.append((i, t, result.width, result.height, phash_hex,
                       vector.tobytes() if vector is not None else None))
    db.set_video_frames(video_id, frames)
