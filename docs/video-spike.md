# Video spike — keyframe sampling, aligned matching, and the parallel pipeline

> Status: implemented on branch `feature/phase-2-video`. This records the two
> design decisions that shape Phase 2 (sampling density and how frames are
> compared) plus the measurement behind the scan pipeline.

## 1. Sampling: ~1 frame per second, capped

A video is not indexed frame by frame. It is reduced to a small set of
keyframes spaced evenly in time:

- target spacing **1 second** (`--interval`),
- floored at **3 frames** (so a 2-second clip still gets samples),
- capped at **120 frames** (so a 1-hour clip samples *sparser* than 1 fps
  instead of exploding into thousands of CLIP passes).

```python
def sample_count(duration, interval=1.0, min_frames=3, max_frames=120):
    return max(min_frames, min(max_frames, ceil(duration / interval)))
```

Timestamps are the fixed grid `k * duration / n`. This matters for the next
point: two copies of the same clip produce frames at the same times.

Extraction is **one ffmpeg pass** per video (`fps=<n/duration>,scale=224:224`,
raw RGB over a pipe), so there is no per-frame subprocess spawn and no PNG
container to parse. The frames then go through the *existing* image pipeline
(`process_image` → CLIP, `phash`).

## 2. Matching: aligned, not all-pairs

Because both videos are on a time grid, duplicate detection does **not** compare
every frame to every other frame (O(n²)). It compares each frame with its
counterpart by relative position, with a small ±2 window for timing jitter:

```python
pos = round(linspace(0, n_b - 1, n_a))     # frame i -> position in the other
best = min over offset in [-2, 2] of hamming(a[i], b[pos[i] + offset])
ratio = matched / shorter_length
```

Cost drops from O(n²) to O(n). A same-duration re-encode aligns exactly; a
trimmed clip aligns proportionally and still scores ~1.0 against its parent.
CLIP grouping uses one mean-pooled vector per video, so it stays O(videos²)
even with a hundred keyframes each.

Videos with `lcs`-style ordering were considered and deliberately dropped: the
time grid plus proportional alignment already covers re-encodes, resolution
changes and end-trims, and it is far cheaper.

## 3. The scan pipeline: keep the GPU fed

The Phase-1 scan prepared **all** images first (CPU-only: open, decode, pHash,
preprocess) and only then ran CLIP in batches. The GPU therefore idled during
prep and burst during embedding — the "spikes and valleys".

The new pipeline runs the CPU work on a thread pool that feeds a queue the main
thread drains in GPU batches. Images and video keyframes share the same queue,
so both kinds of media fill the same pipeline. Processed arrays are released as
soon as they are queued, so memory stays bounded on large trees.

### Measured (Apple M5, CLIP ViT-B/32, 300× 224px JPEGs, best of 3)

| mode | wall | GPU busy | GPU busy / wall |
|---|---|---|---|
| old: serial prep, then embed | 0.86 s | 0.62 s | **73 %** |
| new: pipelined, `--workers 1` | 0.61 s | 0.59 s | **98 %** |
| new: pipelined, `--workers 4` | 0.60 s | 0.59 s | **99 %** |
| new: pipelined, `--workers 8` | 0.61 s | 0.60 s | **99 %** |

The gaps close (~99 % of wall clock is now spent inside `embed_images`) and wall
time drops ~30 % on the same work. Reproduce with
`scripts/bench_pipeline.py`.

## 4. End-to-end check

Test tree: a 640×480 clip, a 320×240 H.264 re-encode of it, an unrelated clip,
and a still photo.

```
media-match scan TREE --video
# 4 media found, 0 up to date, 4 to analyze
# Done: analyzed 4, errors 0. Index: 1 images (0 described), 3 videos (0 described)

media-match duplicates TREE --video
# Group 1 (2 files)
#   * orig.mp4   (640x480, 0:06) [best]
#     reenc.mp4  (320x240, 0:06) hamming_min=0

media-match search TREE "color bars" --video
# orig.mp4   (640x480, 0:06)  similarity=0.311  @ 0:02
# reenc.mp4  (320x240, 0:06)  similarity=0.311  @ 0:02
# other.mp4  (640x480, 0:06)  similarity=0.264  @ 0:05
```

The re-encode groups with its source at hamming 0 across a 4× resolution
change; the higher-resolution copy wins; search returns the matching moment.
