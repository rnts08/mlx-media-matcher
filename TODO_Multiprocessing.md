# TODO — Multiprocessing & Library Cleanup

## 1. Ring-buffer pipeline (branch `perf/ring-buffer-pipeline`)

**Goal:** 20k files in 1-4h instead of ~1 day. Overlap disk I/O, CPU analytics,
and GPU embedding so no stage waits on another.

**Bottleneck:** the describe stage generates captions one at a time
(20k × ~1-2 s = 11+ h). The scan already batches CLIP with 4 workers.

**Design — four stages, three bounded queues:**

```
NVMe
  → [ Read pool: 1-2 ]    read raw bytes → ring buffer (~1 GB, always full)
  → [ CPU pool: 4-8 ]     decode, pHash, CLIP preprocess → arrays
  → [ GPU: 1 ]            batch arrays → CLIP embed (streaming)
  → [ Store: 1 ]          upsert rows, FTS
```

- **Ring buffer:** bounded queue of `(path, bytes)` chunks, ~1 GB total.
  Reader thread(s) pull paths, `read()` bytes, push. Back-pressure when full.
  CPU workers pull bytes, decode, hash, preprocess, push arrays. Disk latency
  is decoupled from decode cost.
- **VLM batching:** `mlx_vlm.generate` accepts a list of images, so one forward
  pass captions 4-8 frames at once. Add `Describer.describe_batch()`. Biggest
  win for describe. Batch size flag `--describe-batch` (default 6).
- **Periodic `mx.clear_cache()`** every N items (already added) so huge
  libraries do not grow into a jetsam target.

**Files:** `pipeline.py` (RingBuffer + staged pipeline), `analyze.py`
(`describe_batch` + ring feed), `cli.py` (`--describe-batch`, `--io-workers`).

**Tests:** new tests for ring buffer stages and VLM batching; all 121 existing
tests must still pass.

---

## 2. `bootstrap` command — first pass on a tree

**Goal:** the first thing a user calls on a tree. Walks the whole tree,
bootstraps the index with filenames, types and sizes, and (with flags) prunes
duplicates and non-media files — without the expensive CLIP embedding.

**Default (dry-run):** build the database only; do not touch any files.

**Flags:**
- `--prune` — find pHash exact duplicates, keep the newest version, trash the rest
- `--media-only` — keep only media-type files (by extension), prune `.zip`/`.doc`/etc.,
  and consolidate extensions to canonical form (`.jpeg`→`.jpg`, `.JPG`→`.jpg`, …)
- `--fix-extensions` (with `--media-only`) — detect the actual type of files with a
  missing/unknown extension and add/update it (e.g. `IMG_1234` → `IMG_1234.jpg`)
- `--video` — include videos (probed with ffprobe)
- `--yes` — execute without asking; without it, file operations are dry-run

**Works for images and videos.**

**Files:** new `media_match/bootstrap.py`, `db.py` (bootstrap table), `cli.py`
(`cmd_bootstrap`), wired into help summaries.

---

## 3. Extension consolidation (absorbed into `bootstrap --media-only`)

The former `clean-library` command is rolled into bootstrap. Walking the tree,
each extension is matched to a canonical form and the file is renamed:

| Canonical | Aliases |
|---|---|
| `.jpg` | .jpeg .jpe .jfif .JPG .JPEG … |
| `.png` | .PNG |
| `.gif` | .GIF |
| `.bmp` | .dib .BMP |
| `.tif` | .tiff .TIF .TIFF |
| `.webp` | .WEBP |
| `.heic` | .heif .HEIC .HEIF |
| `.mp4` | .m4v .mp4v .MP4 .M4V |
| `.mov` | .qt .MOV |
| `.avi` | .AVI |
| `.mkv` | .MKV |
| `.webm` | .WEBM |
| `.mpg` | .mpeg .mp2 .mpe .mpv .MPG .MPEG |
| `.wmv` | .WMV |
| `.mp3` | .MP3 |
| `.wav` | .WAV |
| `.flac` | .FLAC |
| `.aac` | .AAC |
| `.ogg` | .oga .OGG |
| `.m4a` | .M4A |

**DB sync:** after renames, update the affected `files`/`videos.path` rows.

---

## Summary of all requested features

| # | Feature | Status |
|---|---|---|
| 1 | Fix `--describe` without `--model` (`Path(None)`) | ✅ done |
| 2 | Integrated `analyze` command (scan+describe+refine+classify) | ✅ done |
| 3 | `describe` command (caption missing only) | ✅ done |
| 4 | `clear-metadata` / `import-metadata` / `export-metadata` | ✅ done |
| 5 | `reset` command (delete index DB) | ✅ done |
| 6 | `check` command (ok/truncated/corrupt + interactive repair) | ✅ done |
| 7 | Scan tolerates truncated images | ✅ done |
| 8 | Periodic `mx.clear_cache()` on long runs | ✅ done |
| 9 | Ring-buffer pipeline (I/O + CPU + GPU overlap) | 📋 planned |
| 10 | VLM batching (`describe_batch`) | 📋 planned |
| 11 | `bootstrap` command (dry-run index; `--prune`, `--media-only`, `--fix-extensions`) | ✅ done |
| 12 | Extension consolidation (rolled into `bootstrap --media-only`) | ✅ done |
