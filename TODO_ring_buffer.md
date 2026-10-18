# Ring-buffer pipeline

**Status: shipped in 0.2.21** (branch `perf/ring-buffer-pipeline`, merged to
`main`). Kept as the design record — the numbers and the reasoning behind the
stage split.

**Branch:** `perf/ring-buffer-pipeline`
**Goal:** 20k files in 1-4h instead of ~1 day. Overlap disk I/O, CPU analytics,
and GPU embedding so no stage waits on another.

## Bottleneck

The describe stage generates captions one at a time (20k × ~1-2 s = 11+ h).
The scan already batches CLIP with 4 workers. The fix is two-fold:

1. **VLM batching** — `mlx_vlm.generate` accepts a list of images, so one
   forward pass captions 4-8 frames at once. Biggest win for describe.
2. **Ring buffer** — a dedicated I/O stage keeps ~1 GB of raw file bytes in
   flight from the NVMe, so CPU workers never wait on disk.

## Design — four stages, three bounded queues

```
NVMe
  → [ Read pool: 1-2 ]    read raw bytes → ring buffer (~1 GB, always full)
  → [ CPU pool: 4-8 ]     decode, pHash, CLIP preprocess → arrays
  → [ GPU: 1 ]            batch arrays → CLIP embed (streaming)
  → [ Store: 1 ]          upsert rows, FTS
```

### Ring buffer

- Bounded `queue.Queue` of raw `(path, bytes)` chunks, ~1 GB total.
- Reader thread(s) pull paths from the job iterator, `read()` bytes, push.
  Back-pressure when full.
- CPU workers pull `(path, bytes)`, decode, hash, preprocess, push arrays to
  the batch queue. Disk latency is decoupled from decode cost.

### VLM batching

- `Describer.describe_batch(frames: list[PIL.Image]) -> list[(desc, kw)]`
  sends N frames per `generate` call (both the description pass and the
  keywords pass are batched separately).
- Batch size is a flag (`--describe-batch`). It ships defaulting to **1**, not
  the 6 originally planned here: unified memory is tight and the safe default
  has to work on a 16 GB machine. Raise it once you have measured.

### Stage wiring

| Stage | Pool | Queue in | Queue out |
|---|---|---|---|
| Read | 1-2 | file paths | raw bytes (~1 GB ring) |
| Preprocess | 4-8 | raw bytes | numpy arrays |
| Embed | 1 (GPU) | arrays | CLIP vectors |
| Store | 1 | vectors | SQLite |

## Files to change

- `media_match/pipeline.py` — add `RingBuffer`; split `run_scan` into the
  staged pipeline; keep the existing API so tests pass.
- `media_match/analyze.py` — add `Describer.describe_batch`; feed
  `run_describe_stage` from the ring buffer; batch the VLM calls.
- `media_match/cli.py` — add `--describe-batch`, `--io-workers`; wire the new
  pipeline into `cmd_analyze` / `cmd_scan`.

## Success criteria

- 20k images: wall time drops from ~1 day to 1-4h.
- GPU utilisation stays high (no idle gaps between batches).
- NVMe read throughput stays high (ring buffer always full).
- All existing tests pass; new tests for the ring buffer and VLM batching.

## Outcome

Shipped; the suite went 137 → 150 tests. Two things the criteria did not cover
turned up during the work and are worth recording:

- The staged pipeline was wired into `cmd_scan`/`cmd_analyze` but never
  imported from `.pipeline`, so `run_staged_scan` raised `NameError` on every
  run. Two tests caught it; keep an eye on new call sites of this shape.
- `run_scan` keeps its old signature and behaviour — `run_staged_scan` sits
  beside it, so the staged path can be A/B'd rather than swapped in blind.

The 20k-file target has **not** been measured end to end; the pipeline
structure is in place, the timing claim is still open.
