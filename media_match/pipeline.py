"""Parallel scan pipeline: prepare frames on CPU, embed them on the GPU.

The Phase-1 scan prepared one file and then embedded one batch, so the GPU sat
idle while Python decoded images and hashed frames (the "spikes and valleys").
Here the CPU-heavy work (open, decode, pHash, CLIP preprocessing, ffmpeg) runs
on a thread pool that keeps the embedding queue full, while the main thread
drains that queue in GPU batches. Images and video keyframes share the same
queue, so both kinds of media fill the same pipeline.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from .clip_processor import process_image
from .db import Database
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


def iter_media(root: str, include_video: bool) -> list[tuple[str, str]]:
    """(kind, path) for every image (and, optionally, video) under root."""
    paths: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            path = os.path.join(dirpath, name)
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
) -> tuple[list[ScanJob], int, int]:
    """Return (jobs, total, up_to_date) after applying the mtime/size check."""
    media = iter_media(root, include_video)
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


def prune_missing(root: str, db: Database, *, include_video: bool) -> int:
    """Drop index rows for media under ``root`` no longer present on disk.

    Scoped to ``root`` so scanning a subtree never touches the rest of the
    index, and skipped entirely when ``root`` itself is gone, so an unmounted
    or renamed drive cannot wipe the index. Returns the number of rows removed.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return 0
    prefix = root + os.sep
    file_rows, video_rows = db.all_paths()

    stale_files = [
        r["id"] for r in file_rows
        if _under(r["path"], root, prefix) and not os.path.exists(r["path"])
    ]
    stale_videos: list[int] = []
    if include_video:
        stale_videos = [
            r["id"] for r in video_rows
            if _under(r["path"], root, prefix) and not os.path.exists(r["path"])
        ]

    if stale_files:
        db.delete_files(stale_files)
    if stale_videos:
        db.delete_videos(stale_videos)
    return len(stale_files) + len(stale_videos)


def prepare(job: ScanJob, interval: float, max_frames: int) -> ScanResult:
    """CPU-only work for one file: hash, decode, perceptual-hash, preprocess."""
    from PIL import Image

    sha1 = _sha1_of(job.path)
    if job.kind == "image":
        with Image.open(job.path) as img:
            width, height = img.size
            frame_hash = hash_to_hex(phash(img))
            array = process_image(img.convert("RGB"))
        return ScanResult(
            job=job, sha1=sha1, width=width, height=height,
            phashes=[frame_hash], arrays=[array], signature=frame_hash,
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

    def emit(owner: int) -> None:
        nonlocal emitted
        result = results.pop(owner)
        result.vectors = owners.pop(owner)
        remaining.pop(owner, None)
        emitted += 1
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
