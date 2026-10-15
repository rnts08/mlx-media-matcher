"""Throwaway benchmark: old serial prepare-then-embed vs new pipelined scan."""
import os
import sys
import time

from media_match.clip import ClipEncoder
from media_match.db import Database
from media_match.pipeline import build_jobs, prepare, run_scan

T = "/private/var/folders/ct/1_9ls48x5_1d9yslc6j9t4vw0000gn/T/opencode/mm-small"
BATCH = 16


class Timed:
    def __init__(self, enc):
        self.enc = enc
        self.busy = 0.0
        self.calls = 0

    def embed_images(self, arrays):
        t = time.perf_counter()
        out = self.enc.embed_images(arrays)
        self.busy += time.perf_counter() - t
        self.calls += 1
        return out


def old_style(jobs, enc):
    _db = Database(":memory:")
    t0 = time.perf_counter()
    results = [prepare(j, 1.0, 120) for j in jobs]          # serial CPU
    arrays = [a for r in results for a in r.arrays]
    for s in range(0, len(arrays), BATCH):                   # then GPU bursts
        enc.embed_images(arrays[s:s + BATCH])
    return time.perf_counter() - t0


def new_style(jobs, enc, workers):
    t0 = time.perf_counter()
    run_scan(jobs, enc, interval=1.0, max_frames=120,
             batch_size=BATCH, workers=workers)
    return time.perf_counter() - t0


def main():
    db = Database(":memory:")
    jobs, total, _ = build_jobs(T, db, include_video=False, reanalyze=False)
    print(f"{len(jobs)} jobs")

    raw = ClipEncoder()
    # warm up the model / Metal
    _ = raw.embed_texts(["warmup"])

    for label, fn in [
        ("old (serial prep, then embed)", lambda e: old_style(jobs, e)),
        ("new workers=1", lambda e: new_style(jobs, e, 1)),
        ("new workers=4", lambda e: new_style(jobs, e, 4)),
        ("new workers=8", lambda e: new_style(jobs, e, 8)),
    ]:
        best = None
        for _ in range(3):
            timed = Timed(raw)
            wall = fn(timed)
            if best is None or wall < best[0]:
                best = (wall, timed.busy)
        wall, busy = best
        print(f"{label:32s} wall={wall:6.2f}s  gpu_busy={busy:6.2f}s  "
              f"({100 * busy / wall:4.1f}% of wall in embeddings)")


if __name__ == "__main__":
    main()
