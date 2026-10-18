"""Integrated passes: one describe stage for images AND videos.

The scan pipeline (see `pipeline.py`) already overlaps CPU decoding with GPU
CLIP batches. Describing used to be the slow, serial tail: for every video the
main thread ran ffmpeg, wrote a temp PNG, generated the caption, deleted the
temp file — and the GPU sat idle through all the CPU work.

This module mirrors the scan pipeline's shape for the VLM pass:

  * one unified job list for images and videos, best-of-duplicate-group only
  * CPU workers prefetch the representative frame (decode the image, or pull a
    mid-frame still out of the video with ffmpeg) into a bounded queue
  * the main thread runs the VLM generation back to back, never waiting on a
    decode that has not already finished
"""

from __future__ import annotations

import io
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

from .db import Database

# how many prefetched frames to keep ready; each one is a decoded 448x448 RGB
# image (~600 KB), so even a generous window costs little memory
DEFAULT_WINDOW = 8


@dataclass
class DescribeJob:
    """One media item waiting for a VLM caption."""

    kind: str  # "image" | "video"
    media_id: int
    path: str  # absolute, for ffmpeg / PIL / error reports
    label: str  # relative to root, for progress lines
    duration: float = 0.0  # videos: still is taken at duration / 2


@dataclass
class DescribePlan:
    """Who needs a caption, and what was skipped before the run started."""

    jobs: list[DescribeJob]
    missing: int = 0  # indexed but no longer on disk (stale rows)
    shadowed: int = 0  # duplicates whose best version is described instead
    pending: dict[str, int] = None  # type: ignore[assignment]  # per-kind counts

    def __post_init__(self) -> None:
        if self.pending is None:
            self.pending = {}


def load_frame(job: DescribeJob):
    """CPU work: return the job's representative frame as a PIL image.

    Images are opened in place (HEIC included, once the opener is registered);
    videos get a single mid-clip still piped out of ffmpeg as PNG bytes, so no
    temporary file is ever written.
    """
    from PIL import Image

    if job.kind == "image":
        with Image.open(job.path) as img:
            return img.convert("RGB")

    from .video import extract_still_png

    png = extract_still_png(job.path, at=(job.duration / 2 if job.duration else 0.0))
    with Image.open(io.BytesIO(png)) as img:
        return img.convert("RGB")


def describe_targets(db: Database, root: str, *, include_videos: bool = True,
                     missing_ok: bool = False) -> DescribePlan:
    """Build the unified describe queue: undescribed images + videos.

    The best-resolution version of each duplicate group is queued; its copies
    are left undescribed (shadowed) so captions are not spent twice. Indexed
    items that vanished from disk are counted and skipped, exactly like the
    old per-kind loops, unless ``missing_ok`` keeps them queued.
    """
    from .groups import find_duplicate_groups
    from .video_groups import find_video_duplicate_groups

    jobs: list[DescribeJob] = []
    missing = 0
    shadowed = 0
    pending_counts: dict[str, int] = {}

    def _select(records, kind: str, best_of: dict[int, int]) -> list:
        nonlocal missing, shadowed
        pending = [r for r in records if not r.description]
        pending_counts[kind] = len(pending)
        gone = [r for r in pending if not os.path.exists(r.path)]
        missing += len(gone)
        if gone:
            pending = [r for r in pending if os.path.exists(r.path)]
        picked = [r for r in pending if best_of.get(r.id, r.id) == r.id]
        shadowed += len(pending) - len(picked)
        return picked

    records = db.all_files()
    best_of: dict[int, int] = {}
    for group in find_duplicate_groups(records):
        for member in [group.best] + [rec for rec, _ in group.others]:
            best_of[member.id] = group.best.id
    for r in _select(records, "image", best_of):
        jobs.append(DescribeJob("image", r.id, r.path, os.path.relpath(r.path, root)))

    if include_videos:
        vrecords = db.all_videos_with_frames()
        vbest: dict[int, int] = {}
        for group in find_video_duplicate_groups(vrecords):
            for member in [group.best] + [rec for rec, _ in group.others]:
                vbest[member.id] = group.best.id
        for r in _select(vrecords, "video", vbest):
            jobs.append(DescribeJob(
                "video", r.id, r.path, os.path.relpath(r.path, root),
                duration=r.duration or 0.0,
            ))

    return DescribePlan(jobs=jobs, missing=missing, shadowed=shadowed,
                        pending=pending_counts)


def run_describe_stage(
    jobs: list[DescribeJob],
    describer,
    db: Database,
    *,
    workers: int = 3,
    window: int = DEFAULT_WINDOW,
    batch_size: int = 1,
    on_result=None,
    on_error=None,
    progress=None,
) -> int:
    """Caption every job, keeping the GPU fed with prefetched frames.

    CPU workers decode frames ``window`` items ahead; the calling thread runs
    the VLM and stores each result as soon as it is ready. Returns the number of
    items described.

    When ``batch_size`` > 1 the VLM captions that many frames per forward pass
    (``Describer.describe_batch``), which is significantly faster than one
    call per image. Batch mode requires the describer to expose
    ``describe_batch``; otherwise it falls back to one-at-a-time.

    Every 100 items the MLX cache is returned to the OS, so a 100k-item
    library cannot grow the process into a jetsam target.
    """
    if not jobs:
        return 0
    window = max(2, min(window, len(jobs)))
    emitted = 0
    clear_every = 100
    use_batch = batch_size > 1 and hasattr(describer, "describe_batch")

    def store(job: DescribeJob, description: str, keywords: list[str]) -> None:
        joined = ", ".join(keywords)
        if job.kind == "image":
            db.set_description(job.media_id, description, joined)
        else:
            db.set_video_description(job.media_id, description, joined)
        if on_result is not None:
            on_result(job, description, keywords)

    def store_batch(batch: list[tuple[DescribeJob, str, list[str]]]) -> None:
        nonlocal emitted
        for job, description, keywords in batch:
            store(job, description, keywords)
            emitted += 1
        if emitted % clear_every < len(batch):
            _clear_mlx_cache()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        iterator = iter(jobs)
        pending: set = set()
        job_of: dict[object, DescribeJob] = {}

        def fill() -> None:
            while len(pending) < window:
                try:
                    job = next(iterator)
                except StopIteration:
                    return
                future = pool.submit(load_frame, job)
                job_of[future] = job
                pending.add(future)

        fill()
        ready: list[tuple[DescribeJob, object]] = []

        def drain_ready() -> None:
            """Move completed futures into the ready list."""
            done, pending_set = wait(pending, return_when=FIRST_COMPLETED)
            pending.clear()
            pending.update(pending_set)
            for future in done:
                ready.append((job_of.pop(future), future))

        drain_ready()
        while ready or pending:
            # collect a batch of decoded frames
            batch_jobs: list[DescribeJob] = []
            batch_frames: list = []
            while len(batch_jobs) < batch_size and ready:
                job, future = ready.pop(0)
                if progress is not None:
                    progress(job)
                try:
                    batch_frames.append(future.result())
                    batch_jobs.append(job)
                except Exception as exc:  # noqa: BLE001 - keep describing
                    if on_error is not None:
                        on_error(job.path, exc)
            if not batch_jobs:
                if pending:
                    drain_ready()
                    fill()
                    continue
                break
            if use_batch and len(batch_jobs) > 1:
                try:
                    results = describer.describe_batch(batch_frames)
                    store_batch(list(zip(batch_jobs, [r[0] for r in results],
                                          [r[1] for r in results])))
                except Exception as exc:  # noqa: BLE001 - fall back to one-by-one
                    for job, frame in zip(batch_jobs, batch_frames):
                        if on_error is not None:
                            on_error(job.path, exc)
                        try:
                            d, k = describer.describe(frame)
                            store(job, d, k)
                            emitted += 1
                        except Exception:  # noqa: BLE001
                            pass
                    if emitted % clear_every == 0:
                        _clear_mlx_cache()
            else:
                job, frame = batch_jobs[0], batch_frames[0]
                try:
                    description, keywords = describer.describe(frame)
                    store(job, description, keywords)
                    emitted += 1
                    if emitted % clear_every == 0:
                        _clear_mlx_cache()
                except Exception as exc:  # noqa: BLE001 - keep describing
                    if on_error is not None:
                        on_error(job.path, exc)
            fill()
            drain_ready()

    return emitted


def _clear_mlx_cache() -> None:
    """Return MLX's cached (unused) memory to the OS; best effort."""
    try:
        import mlx.core as mx

        (mx.clear_cache if hasattr(mx, "clear_cache") else mx.metal.clear_cache)()
    except (ImportError, AttributeError):
        pass
