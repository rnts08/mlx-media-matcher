"""Measure this machine, so the scan flags can be set from data.

The obvious way to pick ``--batch``/``--io-workers``/``--workers`` is to guess,
run a scan over 20k files, and find out an hour later that the GPU sat idle
most of the time. This module instead answers three narrow questions quickly,
on a small synthetic corpus, before committing to a full scan:

  * how fast does the disk deliver bytes (sets how many readers are worth it)?
  * how fast can the CPU decode and preprocess them (sets ``--workers``)?
  * how fast is one CLIP forward pass (sets ``--batch``)?

Then a handful of real end-to-end configurations go through
``run_staged_scan`` and are compared, so the recommendation reflects the
pipeline as it actually behaves rather than three independent microbenchmarks:

    batch=16  io-workers=2  workers=4  ->  214 img/s

The corpus is synthetic by default on purpose: it needs no photos on disk, it
is identical every run so numbers stay comparable between runs and machines,
and it never touches the real index. Pass ``--root`` to benchmark against real
files when the synthetic shapes do not resemble your library.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

# Image shapes chosen to span what a photo library actually holds: phone
# photos, resized web copies and thumbnails, so decode cost is realistic.
CORPUS_SHAPES: tuple[tuple[int, int], ...] = (
    (4032, 3024),   # 12MP phone photo
    (2048, 1536),   # 3MP
    (1024, 768),    # 0.8MP web copy
    (800, 600),     # 0.5MP thumbnail
)

# Configurations tried end to end. Kept small: each one costs real GPU time,
# and the goal is to choose settings, not to publish a benchmark suite.
CANDIDATES: tuple[tuple[int, int, int], ...] = (
    # (batch, io_workers, workers)
    (8, 1, 2),
    (8, 2, 4),
    (16, 2, 4),
    (16, 4, 8),
    (32, 4, 8),
)

# GPU batch sizes probed individually.
GPU_BATCHES: tuple[int, ...] = (1, 8, 16, 32)


@dataclass
class StageResult:
    """One isolated measurement."""

    name: str
    seconds: float
    detail: str = ""


@dataclass
class ConfigResult:
    """One end-to-end scan configuration, timed."""

    batch: int
    io_workers: int
    workers: int
    items: int
    wall: float
    gpu: float
    cpu: float
    note: str = ""

    @property
    def per_second(self) -> float:
        return self.items / self.wall if self.wall > 0 else 0.0

    @property
    def gpu_share(self) -> float:
        """Fraction of wall time spent inside CLIP forward passes."""
        return self.gpu / self.wall if self.wall > 0 else 0.0

    @property
    def ok(self) -> bool:
        return self.items > 0 and self.wall > 0 and not self.note


@dataclass
class BenchReport:
    stages: list[StageResult] = field(default_factory=list)
    configs: list[ConfigResult] = field(default_factory=list)
    best: ConfigResult | None = None
    # Configurations that measured the same as the winner, and the one we
    # suggest when several tie (the cheapest of them).
    ties: list[ConfigResult] = field(default_factory=list)
    recommended: ConfigResult | None = None
    corpus_items: int = 0
    corpus_bytes: int = 0
    source: str = ""
    note: str = ""


# -- corpus ---------------------------------------------------------------------


def make_corpus(directory: str, count: int) -> list[str]:
    """Write ``count`` synthetic JPEGs into ``directory``; return their paths.

    Content is coarse noise upscaled to full size. A flat-colour image decodes
    far faster than a photograph, which would make every CPU-bound number look
    better than it is; noise keeps the decoder doing real DCT work while
    staying cheap to generate.
    """
    import numpy as np
    from PIL import Image

    os.makedirs(directory, exist_ok=True)
    paths: list[str] = []
    rng = np.random.default_rng(1234)
    for i in range(count):
        w, h = CORPUS_SHAPES[i % len(CORPUS_SHAPES)]
        small = rng.integers(0, 255, size=(h // 16, w // 16, 3), dtype=np.uint8)
        img = Image.fromarray(small).resize((w, h), Image.Resampling.BILINEAR)
        path = os.path.join(directory, f"bench_{i:04d}.jpg")
        img.save(path, quality=88)
        paths.append(path)
    return paths


def pick_corpus(root: str | None, count: int, *, ignore=None,
                temporary=None) -> tuple[list[str], str]:
    """Return (paths, description) for the corpus to benchmark.

    Real files under ``root`` when given, otherwise a synthetic corpus written
    into ``temporary`` (a TemporaryDirectory the caller owns).
    """
    if root:
        from .pipeline import iter_media

        paths = [p for _kind, p in iter_media(root, False, ignore=ignore)][:count]
        return paths, f"real files under {root}"
    if temporary is None:
        raise ValueError("a temporary directory is needed for the synthetic corpus")
    return make_corpus(temporary, count), f"{count} synthetic JPEGs"


# -- isolated stages ------------------------------------------------------------


def measure_disk(paths: list[str], readers: int) -> StageResult:
    """Raw read throughput with ``readers`` threads. No decoding."""
    import threading

    chunks: list[bytes] = [b""] * len(paths)
    start = time.perf_counter()

    def work(indexes) -> None:
        for i in indexes:
            try:
                with open(paths[i], "rb") as fh:
                    chunks[i] = fh.read()
            except OSError:
                pass

    threads = [
        threading.Thread(target=work, args=(range(i, len(paths), readers),))
        for i in range(readers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seconds = time.perf_counter() - start
    total = sum(len(c) for c in chunks)
    mb_s = total / 1_000_000 / seconds if seconds > 0 else 0.0
    return StageResult(
        "disk read", seconds,
        f"{readers} thread(s): {mb_s:,.0f} MB/s over {len(paths)} file(s)",
    )


def measure_cpu(paths: list[str], workers: int) -> StageResult:
    """Decode + pHash + CLIP-preprocess cost with ``workers`` threads."""
    from concurrent.futures import ThreadPoolExecutor

    from .clip_processor import process_image
    from .hashing import hash_to_hex, phash
    from .pipeline import load_image_tolerant

    def one(path: str) -> None:
        img = load_image_tolerant(path)
        try:
            hash_to_hex(phash(img))
            process_image(img.convert("RGB"))
        finally:
            img.close()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, paths))
    seconds = time.perf_counter() - start
    rate = len(paths) / seconds if seconds > 0 else 0.0
    return StageResult("cpu decode", seconds, f"{workers} thread(s): {rate:,.0f} img/s")


def measure_gpu(encoder, sample_arrays, batch: int) -> StageResult:
    """Time one CLIP forward pass at this batch size."""
    if not sample_arrays:
        return StageResult("gpu embed", 0.0, "no sample arrays available")
    arrays = [sample_arrays[i % len(sample_arrays)] for i in range(batch)]
    encoder.embed_images(arrays[:1])  # warm up: the first pass compiles kernels
    start = time.perf_counter()
    encoder.embed_images(arrays)
    seconds = time.perf_counter() - start
    return StageResult(
        "gpu embed", seconds,
        f"batch {batch}: {batch / seconds:,.0f} img/s ({seconds * 1000 / batch:.1f} ms/img)",
    )


# -- end to end -----------------------------------------------------------------


class _TimedEncoder:
    """Wraps a ClipEncoder so forward-pass time lands in ScanStats.gpu."""

    def __init__(self, encoder, stats) -> None:
        self._encoder = encoder
        self._stats = stats

    def embed_images(self, arrays):
        start = time.perf_counter()
        out = self._encoder.embed_images(arrays)
        self._stats.gpu += time.perf_counter() - start
        return out

    def unload(self) -> None:
        pass


def load_encoder(clip_model: str):
    """Load CLIP and force kernel compilation before any timing starts."""
    from .clip import ClipEncoder

    encoder = ClipEncoder(model_id=clip_model, batch_size=1)
    encoder.embed_texts(["warm up"])
    return encoder


def bench_config(jobs, encoder, batch: int, io_workers: int, workers: int,
                 *, repeat: int = 1) -> ConfigResult:
    """Time one configuration end to end, keeping its best of ``repeat``."""
    from .pipeline import ScanStats, run_staged_scan

    best: ConfigResult | None = None
    for _ in range(max(1, repeat)):
        stats = ScanStats()
        try:
            run_staged_scan(
                jobs, _TimedEncoder(encoder, stats), interval=1.0, max_frames=1,
                batch_size=batch, workers=workers, io_workers=io_workers,
                stats=stats,
            )
        except Exception as exc:  # noqa: BLE001 - a failed config is a data point
            return ConfigResult(batch, io_workers, workers, 0, 0.0, 0.0, 0.0,
                                note=f"{type(exc).__name__}: {exc}")
        got = ConfigResult(batch, io_workers, workers, stats.items, stats.wall,
                           stats.gpu, stats.cpu)
        if best is None or got.wall < best.wall:
            best = got
    return best or ConfigResult(batch, io_workers, workers, 0, 0.0, 0.0, 0.0,
                                note="no run completed")


def recommend(configs: list[ConfigResult]) -> ConfigResult | None:
    """The fastest configuration that actually processed the corpus."""
    usable = [c for c in configs if c.ok]
    return min(usable, key=lambda c: c.wall) if usable else None


#: Fraction by which a configuration may trail the best and still count as a
#: tie. Measured on a 24-image corpus with seven repeats per configuration:
#: run-to-run spread reached 10.4% (coefficient of variation 0.9-3.4%). Two
#: configurations closer than that cannot be separated by this benchmark, so
#: anything tighter would report noise as a result.
TIE_THRESHOLD = 0.10


def indistinguishable(best: ConfigResult, configs: list[ConfigResult],
                      noise: float = TIE_THRESHOLD) -> list[ConfigResult]:
    """Configurations whose wall time is within ``noise`` of the best.

    At a few dozen files, neighbouring configurations differ by less than the
    run-to-run variation, so singling one out as "the winner" would dress up a
    coin flip as a measurement. These are all equally good on this machine.
    """
    if best.wall <= 0:
        return [best]
    limit = best.wall * (1.0 + noise)
    return [c for c in configs if c.ok and c.wall <= limit]


def cheapest(candidates: list[ConfigResult]) -> ConfigResult:
    """Of several equally fast configurations, the one using least resources.

    Fewer threads is the better default when throughput is the same: it leaves
    the machine responsive and scales better as the tree grows.
    """
    return min(candidates, key=lambda c: (c.workers, c.io_workers, c.batch))


def human_seconds(seconds: float) -> str:
    """0.4s / 12s / 3.4m / 2h05m / 2 days 6h — never a bare '0.0 h'."""
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    hours, rest = divmod(int(round(seconds)), 3600)
    if hours < 48:
        return f"{hours}h{rest // 60:02d}m"
    return f"{hours // 24} days {hours % 24}h"


# -- top level ------------------------------------------------------------------


def run_benchmark(*, root: str | None, count: int, repeat: int,
                  clip_model: str, video: bool = False,
                  ignore=None) -> BenchReport:
    """Benchmark the scan pipeline on this machine and recommend settings."""
    from .pipeline import ScanJob, _prepare_bytes

    report = BenchReport()
    with tempfile.TemporaryDirectory(prefix="media-match-bench-") as tmp:
        if root:
            paths, source = pick_corpus(root, count, ignore=ignore)
        else:
            paths, source = pick_corpus(None, count, temporary=tmp)
        report.source = source
        if not paths:
            report.note = "no media found to benchmark"
            return report

        report.corpus_items = len(paths)
        report.corpus_bytes = sum(os.path.getsize(p) for p in paths)

        # Stages first: they explain the end-to-end numbers that follow.
        probe = paths[:32]
        report.stages.append(measure_disk(probe, readers=1))
        report.stages.append(measure_disk(probe, readers=4))
        report.stages.append(measure_cpu(probe, workers=1))
        report.stages.append(measure_cpu(probe, workers=4))

        # Jobs are built once so every candidate does identical work.
        jobs = [
            ScanJob(kind="image", path=p, size=os.path.getsize(p),
                    mtime=os.path.getmtime(p))
            for p in paths
        ]

        encoder = load_encoder(clip_model)

        sample_arrays = None
        try:
            with open(jobs[0].path, "rb") as fh:
                sample_arrays = _prepare_bytes(jobs[0], fh.read(), interval=1.0,
                                               max_frames=1).arrays
        except Exception:  # noqa: BLE001 - the GPU probe is best-effort
            sample_arrays = None
        for batch in GPU_BATCHES:
            report.stages.append(measure_gpu(encoder, sample_arrays, batch))

        for batch, io_workers, workers in CANDIDATES:
            report.configs.append(
                bench_config(jobs, encoder, batch, io_workers, workers, repeat=repeat)
            )
        report.best = recommend(report.configs)
        if report.best is not None:
            report.ties = indistinguishable(report.best, report.configs)
            # When several configurations measure the same, recommend the one
            # that spends the least: same speed, less load on the machine.
            report.recommended = cheapest(report.ties)
    return report