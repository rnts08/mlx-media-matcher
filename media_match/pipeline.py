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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    progress=None,
    on_error=None,
) -> list[ScanResult]:
    """Prepare jobs concurrently and embed their frames in GPU batches.

    Returns results with ``.vectors`` filled in (one vector per prepared array).
    """
    prepared: list[ScanResult] = []
    owners: dict[int, list[np.ndarray | None]] = {}
    batch: list[tuple[int, int, np.ndarray]] = []

    def flush() -> None:
        if not batch:
            return
        arrays = [arr for _, _, arr in batch]
        try:
            vectors = encoder.embed_images(arrays)
        except Exception as exc:  # noqa: BLE001 - report, drop this batch
            if on_error:
                on_error("<batch>", exc)
            batch.clear()
            return
        for (owner, frame, _), vec in zip(batch, vectors):
            owners[owner][frame] = vec
        batch.clear()

    prepare_one = partial(prepare, interval=interval, max_frames=max_frames)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(prepare_one, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            job = futures[future]
            done += 1
            if progress:
                progress(done, len(jobs), job.path)
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep scanning past bad files
                if on_error:
                    on_error(job.path, exc)
                continue
            owner = len(prepared)
            prepared.append(result)
            owners[owner] = [None] * len(result.arrays)
            for frame, array in enumerate(result.arrays):
                batch.append((owner, frame, array))
                if len(batch) >= batch_size:
                    flush()
            # the processed arrays are only needed once they are in the batch
            # queue; drop them so memory stays bounded on large trees
            result.arrays = []
        flush()

    for owner, result in enumerate(prepared):
        result.vectors = [v for v in owners[owner] if v is not None]
    return prepared


def store_result(db: Database, result: ScanResult) -> None:
    """Write one prepared+embedded result to the index."""
    job = result.job
    if job.kind == "image":
        embedding = result.vectors[0].tobytes() if result.vectors else None
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
