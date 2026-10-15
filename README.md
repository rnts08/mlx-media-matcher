# Media Match

Local photo-library analyzer built on Apple's MLX. Point it at a folder and it
builds an index that lets you:

- **find photos by person, object or free text** — CLIP vector search by
  default, caption/keyword search optionally
- **identify duplicates across resolutions** — accidentally saved thumbnails,
  WebP conversions and recompressed copies are grouped with their
  full-resolution primary file
- **group semantic near-duplicates** — e.g. several different shots of the
  same subject
- **mark the best version** — the largest image in every group is the primary

Everything runs locally: no cloud services, no PyTorch. Requires Apple Silicon.

## Status

- **Phase 1 (images): implemented** — described below.
- **Phase 2 (video): implemented** — the same capabilities for video files:
  keyframes sampled at ~1 fps, embedded with CLIP for search, and matched by
  keyframe pHash for duplicate/re-encode grouping (see `docs/video-spike.md`).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Video support also needs `ffmpeg`/`ffprobe` on `PATH` (e.g. `brew install ffmpeg`).

Models are downloaded from Hugging Face on first use: CLIP ViT-B/32
(~600 MB) for `scan`, SmolVLM-256M (~150 MB) for `scan --describe`.

## How it works

Each image is indexed by three complementary layers (see `docs/clip-spike.md`
for the measurements that justify the design):

| Layer | Engine | Purpose |
|---|---|---|
| CLIP embeddings | `mlx-community/clip-vit-base-patch32`, pure MLX, ~500 img/s, 2 KB/image | default search engine; text→image queries, image→image similarity (`similar`) |
| Perceptual hashes (pHash) | 64-bit DCT hash, no model needed | exact duplicate detection, resolution- and re-encode-independent; stays exact for tiny thumbnails where CLIP degrades |
| Vision-model captions (optional) | `mlx-community/SmolVLM-256M-Instruct-4bit`, ~1 s/image | human-readable descriptions + keywords for `search --fts` and `tags`; run only with `--describe`, and only on the best version of each duplicate group |

The index lives in `<root>/.media-match.db` (SQLite + FTS5). Read-only
commands never create it — run `scan` first.

**Videos** are indexed by sampling keyframes (~1 fps, capped at 120 for long
clips) with `ffmpeg`, then running those frames through the same CLIP + pHash
layers. Keyframes are sampled on a fixed time grid, so duplicate copies line up
frame-for-frame and are matched by *aligned* comparison (O(n)) rather than
all-pairs. The scan prepares frames on a thread pool that feeds the CLIP
batcher, so image and video work share one pipeline and the GPU stays busy
(measured GPU-busy time ~73% → ~99%; see `docs/video-spike.md`).

## Usage

```bash
# 1. index a photo tree: content hash + pHash + CLIP embeddings (~500 img/s)
.venv/bin/media-match scan ~/Pictures

# 1b. optional: add captions with the vision model (~1 s/image, only for the
#     best-resolution version of each duplicate group)
.venv/bin/media-match scan ~/Pictures --describe

# 2. find photos by person / object / free text (CLIP vector search)
.venv/bin/media-match search ~/Pictures "woman red jacket"
.venv/bin/media-match search ~/Pictures --object bicycle
.venv/bin/media-match search ~/Pictures "birthday cake" --fts   # caption search

# 3. duplicate groups; best-resolution version marked with *
.venv/bin/media-match duplicates ~/Pictures
.venv/bin/media-match duplicates ~/Pictures --json        # machine-readable
.venv/bin/media-match duplicates ~/Pictures --threshold 4 # stricter pHash

# 4. semantic near-duplicate groups (CLIP cosine similarity)
.venv/bin/media-match similar ~/Pictures
.venv/bin/media-match similar ~/Pictures --image /tmp/stray.jpg  # one image

# 5. review duplicates and move rejected copies to the OS Trash.
#    Opens a visual bar UI with the key legend pinned at the top:
#      space = keep the file under the bar (unselects the group's others)
#      a     = keep this whole group       A = keep every file
#      v     = Quick Look the file       p = open it in Preview.app
#      x     = trash this file now (y/n) enter = trash all unkept files (y/n)
#      e     = trash only the decisions made so far — every group up to and
#              including the bar's group (y/n) — then keep reviewing
#      q/esc = quit; asks y/N first if files are still marked for the Trash
#    The algorithm's best copy of each group starts selected. Without a
#    terminal it falls back to line-based prompts.
.venv/bin/media-match clean ~/Pictures
.venv/bin/media-match clean ~/Pictures --dry-run  # preview the plan
.venv/bin/media-match clean ~/Pictures --yes      # accept all suggestions

# extras
.venv/bin/media-match tags ~/Pictures    # keyword counts from captions
.venv/bin/media-match stats ~/Pictures   # index, models, coverage
.venv/bin/media-match embed ~/Pictures   # backfill CLIP embeddings only

# videos — pass --video to index/search/group/review video files
.venv/bin/media-match scan ~/Movies --video        # keyframes + CLIP (~1 fps)
.venv/bin/media-match search ~/Movies "dog on a beach" --video
.venv/bin/media-match duplicates ~/Movies --video  # re-encodes, across resolution
.venv/bin/media-match similar ~/Movies --video     # same-subject video groups
.venv/bin/media-match clean ~/Movies --video       # same review UI as images
```

Example `duplicates` output:

```
Group 1 (2 files)
  * ~/Pictures/beach_original.jpg  (4032x3024, 12.2MP) [best]
    ~/Pictures/beach_thumb.jpg  (150x100, 0.0MP) hamming=0
```

## Commands

| Command | What it does | Key options |
|---|---|---|
| `scan [root]` | index new/changed media (hash + pHash + CLIP). Incremental: unchanged files are skipped; `--video` adds videos (keyframes) | `--video`, `--describe`, `--model`, `--clip-model`, `--workers`, `--interval`, `--max-frames`, `--limit`, `--reanalyze`, `--batch` |
| `search [root] [query...]` | find media. CLIP by default when embeddings exist; `--video` restricts to videos | `--person`, `--object`, `--fts`, `--clip`, `--video`, `--image-only`, `--limit` |
| `similar [root]` | group the tree into semantic near-duplicate clusters | `--threshold` (cosine, default 0.85), `--video` |
| `similar [root] --image F` | nearest neighbors of one image (indexed, or any standalone file); with `--video`, ranks videos | `--limit`, `--video` |
| `duplicates [root]` | group exact/near duplicates via pHash + sha1; best resolution wins | `--threshold` (hamming 0–64, default 8), `--json`, `--video`, `--frame-threshold`, `--ratio` |
| `clean [root]` | review duplicate groups in a visual bar UI with the key legend pinned at the top (hjkl/arrows, space=keep file, a=keep group, A=keep all, v=Quick Look, p=Preview, x=trash file, enter=dedupe all, e=dedupe up to the bar's group then keep reviewing, q/esc=quit — asks y/N if files are still marked) and move rejected copies to the system Trash (never deleted outright; index updated); falls back to line prompts without a TTY | `--yes`, `--dry-run`, `--threshold`, `--video`, `--frame-threshold`, `--ratio` |
| `tags [root]` | keyword counts from captions | `--limit`, `--video` |
| `stats [root]` | index info: models, described/embedded coverage | |
| `embed [root]` | (re)compute CLIP embeddings only | `--recompute`, `--clip-model`, `--batch` |

All commands take `--db` to override the index location. `root` defaults to
the current directory. Supported image formats: JPEG, PNG, WebP, GIF, BMP,
TIFF, HEIC/HEIF. Supported video formats (with `--video`): MP4, MOV, M4V, AVI,
MKV, WebM, MPG/MPEG, WMV (anything `ffmpeg` can decode).

Default models are the smallest viable ones; for better accuracy use
`--clip-model` / `--model` with any MLX-converted checkpoint, e.g.
`--model mlx-community/Qwen2-VL-2B-Instruct-4bit`.

## Phase 1

Analyze all image files in a folder with a vision agent, classify the images
so that it is easy to find a certain person, an object and identify
duplicates.

This should work across resolution, so that accidentally saved thumbnails are
identified as duplicates and grouped with their primary file that should have
a better resolution.

## Phase 2 (video)

The same as Phase 1, but on video files: find videos by person/object content,
group duplicates and re-encodes across resolution, and identify the best
version. Each video is reduced to ~1 keyframe per second (capped for long
clips), embedded with CLIP for search and matched by keyframe pHash for
grouping. See `docs/video-spike.md` for the sampling and matching design and
the scan-pipeline measurements.

## Goal

The goal is to group images of different people and objects together and to
be able to find the best version of an image that might be duplicated in the
tree the program is working on.

## Development

- `docs/clip-spike.md` — the CLIP investigation and benchmarks behind the
  hybrid design (branch `spike/clip`, merged).
- `docs/video-spike.md` — Phase-2 video sampling/matching design and the
  scan-pipeline measurements.
- `scripts/bench_pipeline.py` — reproduces the GPU-busy pipeline numbers.
- `tests/` — pytest suite (`pip install -e '.[dev]'`, then `pytest`).
- Test tree used during development: JPEG/WebP thumbnails of real photos,
  validated for grouping at hamming distance 0 across formats and sizes;
  for video, a clip plus a downscaled H.264 re-encode, validated at
  `hamming_min=0` across a 4× resolution change.
