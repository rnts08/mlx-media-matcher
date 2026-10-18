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
- **refine tags into a clean vocabulary** — keywords are lowercased,
  singularized and synonym-merged into canonical tags, then bucketed into
  facets (person/scene/object/food/…); each photo is also *typed*
  (selfie/portrait/group/scene/city/animal/object/food/document/art) with
  zero-shot CLIP
- **group photos of the same person** — photos tagged with the person facet are
  clustered by CLIP similarity; name a cluster ("Alice") and list it with one
  command
- **mark the best version** — the largest image in every group is the primary

Everything runs locally: no cloud services, no PyTorch. Requires Apple Silicon.

## Status

- **Phase 1 (images): implemented** — described below.
- **Phase 2 (video): implemented** — the same capabilities for video files:
  keyframes sampled at ~1 fps, embedded with CLIP for search, and matched by
  keyframe pHash for duplicate/re-encode grouping (see `docs/video-spike.md`).
- **Tag refinement, image types & people clustering: implemented** —
  `refine` cleans captions into canonical tags/facets and (with `--classify`)
  types each photo; `people` clusters person photos and supports naming them.

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

**Scanning is a four-stage pipeline**, not a loop. Raw file bytes are read by a
small pool of reader threads into a bounded ~1 GB ring buffer; CPU workers pull
from that buffer to decode, hash and preprocess; one GPU stage turns the
resulting arrays into CLIP batches; a single writer stores the vectors. Because
each stage has its own bounded queue, disk latency, CPU decode cost and GPU time
overlap instead of running in sequence, and back-pressure keeps memory flat no
matter how big the tree is. `--io-workers` sizes the reader pool. Captions use
the same idea: `--describe-batch N` sends N frames to the vision model per
forward pass instead of one, which is the single biggest win when describing a
large tree (default 1; raise it while watching memory).

Images also record EXIF **capture time, camera and GPS**, so you can search by
date (`--since`/`--until`) or camera, `stats` reports the captured date range,
and `duplicates --prefer date` marks the newest capture as the best copy (the
default picks the largest).

**Videos** are indexed by sampling keyframes (~1 fps, capped at 120 for long
clips) with `ffmpeg`, then running those frames through the same CLIP + pHash
layers. Keyframes are sampled on a fixed time grid, so duplicate copies line up
frame-for-frame and are matched by *aligned* comparison (O(n)) rather than
all-pairs. The scan prepares frames on a thread pool that feeds the CLIP
batcher, so image and video work share one pipeline and the GPU stays busy
(measured GPU-busy time ~73% → ~99%; see `docs/video-spike.md`).

**Tag refinement** turns the loose VLM keywords into a tidy vocabulary.
`refine` lowercases, singularizes (with irregular plurals), merges obvious
synonyms (`guy`→`man`, `bike`→`bicycle`) and drops filler, writing canonical
tags into `keywords` while keeping the originals in `raw_keywords`. Every tag is
also bucketed into a **facet** (person/animal/object/scene/food/plant/text/
activity/color). With `--classify`, each image is additionally given a **type**
via zero-shot CLIP: the category prompt sentences are embedded once, and the
image is labeled with the closest category — no extra model or training. Run it
on its own with `refine`, or let `scan --refine --classify` keep tags and types
fresh automatically (it reuses the CLIP model already loaded for the scan).

**People clustering** takes every photo whose facet is `person`, embeds its CLIP
vector, and links photos whose cosine similarity clears a threshold, so
same-looking people group together. It is honest about its limits: CLIP groups
*similar photos of a person in a similar shot*, not verified identities (that
needs a face-embedding model). `people --label N=Name` names a cluster and
`search --cluster Name` lists it.

## Configuration

Defaults you usually repeat can live in `~/.config/media-match/config.toml`. An
explicit command-line flag always wins, then the config, then the built-in
default:

```toml
[media-match]
clip_model = "mlx-community/clip-vit-base-patch32"
workers = 8
threshold = 8
```

To keep paths out of the index, add a `.media-matchignore` in the scanned root
(globs, one per line, `#` comments). Matching is against the path relative to
the root and its basename, so directory entries and bare names both work:

```
node_modules/
*_export/
*.tmp
```

Ignored paths are skipped by `scan` and dropped by `scan --prune`.

## Usage

```bash
# 1. index a photo tree: content hash + pHash + CLIP embeddings (~500 img/s)
.venv/bin/media-match scan ~/Pictures

#    the built-in guide (every command, every flag, examples)
.venv/bin/media-match help              # all commands + flags + an example each
.venv/bin/media-match help scan         # one command in full
.venv/bin/media-match scan help         # the same page — all three forms match
.venv/bin/media-match scan --help

# 1b. optional: add captions with the vision model (~1 s/image, only for the
#     best-resolution version of each duplicate group) and, in the same pass,
#     refine the tags and type every photo
.venv/bin/media-match scan ~/Pictures --describe --refine --classify

# 2. find photos by person / object / free text (CLIP vector search)
.venv/bin/media-match search ~/Pictures "woman red jacket"
.venv/bin/media-match search ~/Pictures --object bicycle
.venv/bin/media-match search ~/Pictures "birthday cake" --fts   # caption search

#    by capture date / camera (EXIF; falls back to file mtime)
.venv/bin/media-match search ~/Pictures --since 2021-06 --until 2021-12
.venv/bin/media-match search ~/Pictures --camera iPhone

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
.venv/bin/media-match export ~/Pictures --format json --out index.json

# 6. refine the captions into clean tags/facets, and type each photo
.venv/bin/media-match refine ~/Pictures              # canonical tags + facets
.venv/bin/media-match refine ~/Pictures --classify   # + selfie/scene/object types
.venv/bin/media-match tags ~/Pictures --facets       # tags grouped by facet
.venv/bin/media-match tags ~/Pictures --types        # type counts
.venv/bin/media-match search ~/Pictures --type selfie

# 7. group photos of the same person and name the groups
.venv/bin/media-match people ~/Pictures              # cluster person photos
.venv/bin/media-match people ~/Pictures --label 3=Alice
.venv/bin/media-match search ~/Pictures --cluster Alice

# videos — pass --video to index/search/group/review video files
.venv/bin/media-match scan ~/Movies --video        # keyframes + CLIP (~1 fps)
.venv/bin/media-match search ~/Movies "dog on a beach" --video
.venv/bin/media-match duplicates ~/Movies --video  # re-encodes, across resolution
.venv/bin/media-match similar ~/Movies --video     # same-subject video groups
.venv/bin/media-match clean ~/Movies --video       # same review UI as images

# tuning the scan pipeline — the defaults suit most trees
.venv/bin/media-match scan ~/Pictures --io-workers 4    # disk reader threads
.venv/bin/media-match scan ~/Pictures --describe --describe-batch 6  # caption 6/forward pass

# 8. first pass over a messy tree: inventory, drop duplicates, clear out
#    non-media files. Dry-run by default — nothing moves without --yes.
.venv/bin/media-match bootstrap ~/Downloads                    # inventory only
.venv/bin/media-match bootstrap ~/Downloads --prune            # trash exact dupes
.venv/bin/media-match bootstrap ~/Downloads --media-only       # trash non-media files
.venv/bin/media-match bootstrap ~/Downloads --media-only --fix-extensions

#    move the non-media files to a folder instead of trashing them — they
#    keep their layout under DIR and existing files are never overwritten
.venv/bin/media-match bootstrap ~/Downloads --media-only --move-to ~/non-media --yes
```

> **Quote paths that contain spaces or shell metacharacters** (`&`, `$`, `;`,
> quotes, `…`). The program handles them fine, but an *unquoted* `&` makes the
> shell split the command in two, so it fails with "no such file or directory"
> before it ever reaches us:
>
> ```bash
> media-match scan "/Users/me/_Private/Pictures/Timh & Dung"   # ✓ quoted
> media-match scan /Users/me/_Private/Pictures/Timh & Dung      # ✗ shell splits on &
> ```

Example `duplicates` output:

```
Group 1 (2 files)
  * ~/Pictures/beach_original.jpg  (4032x3024, 12.2MP) [best]
    ~/Pictures/beach_thumb.jpg  (150x100, 0.0MP) hamming=0
```

## Commands

| Command | What it does | Key options |
|---|---|---|
| `analyze [root]` | one integrated pass over images **and** videos: index (hash + pHash + CLIP), VLM captions, tag refinement, image typing. The GPU stays busy: CPU decoding overlaps GPU embedding batches, and CPU frame extraction overlaps GPU caption generation | `--video`, `--describe`, `--refine`, `--classify`, `--model`, `--clip-model`, `--workers`, `--io-workers`, `--describe-workers`, `--describe-batch`, `--interval`, `--max-frames`, `--limit`, `--reanalyze`, `--prune`, `--batch` |
| `scan [root]` | index new/changed media (hash + pHash + CLIP). Incremental: unchanged files are skipped; `--video` adds videos (keyframes) | `--video`, `--describe`, `--refine`, `--classify`, `--model`, `--clip-model`, `--workers`, `--io-workers`, `--describe-batch`, `--interval`, `--max-frames`, `--limit`, `--reanalyze`, `--batch` |
| `describe [root]` | caption indexed media that has no caption yet (images directly, videos via a mid-clip still). Best version of each duplicate group only | `--model`, `--video`, `--describe-workers`, `--describe-batch` |
| `clear-metadata [root] [ITEM...]` | clear description/keywords/type/tags from the listed items (paths relative to root), or everything with no items. Media files are never touched | `--kind`, `--yes` |
| `import-metadata [root] FILE` | import description/keywords/type from a JSON or CSV file (as written by `export-metadata`) | `--kind`, `--fields` |
| `export-metadata [root]` | export editable metadata (kind, path, description, keywords, type) as JSON — round-trips with `import-metadata` | `--kind`, `--out` |
| `reset [root]` | delete the index database (media files are never touched); the next `scan` rebuilds from scratch | `--yes` |
| `check [root]` | classify every file as ok / truncated / corrupt; offer to re-encode truncated images (backup kept) and quarantine corrupt ones | `--video`, `--quarantine`, `--yes` |
| `bootstrap [root]` | first pass: walk the tree, record filename/type/size in the index. Dry-run by default; `--prune` trashes exact duplicates (keeps newest), `--media-only` prunes non-media files and consolidates extensions, `--move-to DIR` moves the non-media files out instead of trashing them, `--fix-extensions` adds missing extensions | `--prune`, `--media-only`, `--move-to`, `--fix-extensions`, `--video`, `--yes` |
| `search [root] [query...]` | find media. CLIP by default when embeddings exist; `--video` restricts to videos | `--person`, `--object`, `--fts`, `--clip`, `--video`, `--image-only`, `--type`, `--cluster`, `--since`, `--until`, `--camera`, `--json`, `--limit` |
| `similar [root]` | group the tree into semantic near-duplicate clusters | `--threshold` (cosine, default 0.85), `--video`, `--json` |
| `similar [root] --image F` | nearest neighbors of one image (indexed, or any standalone file); with `--video`, ranks videos | `--limit`, `--video` |
| `duplicates [root]` | group exact/near duplicates via pHash + sha1; best resolution wins | `--threshold` (hamming 0–64, default 8), `--exact-phash` (group only distance-0 matches; faster), `--json`, `--video`, `--frame-threshold`, `--ratio`, `--prefer` |
| `clean [root]` | review duplicate groups in a visual bar UI with the key legend pinned at the top (hjkl/arrows, space=keep file, a=keep group, A=keep all, v=Quick Look, p=Preview, x=trash file, enter=dedupe all, e=dedupe up to the bar's group then keep reviewing, q/esc=quit — asks y/N if files are still marked) and move rejected copies to the system Trash (never deleted outright; index updated); falls back to line prompts without a TTY | `--yes`, `--dry-run`, `--threshold`, `--video`, `--frame-threshold`, `--ratio`, `--json` (with `--dry-run`) |
| `tags [root]` | refined tag counts (`--facets` groups them; `--types` shows image types; `--raw` shows the pre-refinement keywords) | `--facets`, `--types`, `--raw`, `--json`, `--limit`, `--video` |
| `export [root]` | dump the index (path, kind, size, capture date, camera, description, keywords, type, tags, cluster, duration) | `--format` (json/csv/md, default json), `--out` |
| `refine [root]` | rewrite keywords into canonical tags + facets; `--classify` also types each image with zero-shot CLIP | `--classify`, `--clip-model`, `--recompute` |
| `people [root]` | cluster person photos by CLIP similarity (singletons kept); name a cluster with `--label N=Name` | `--threshold` (cosine, default 0.80), `--label`, `--min-size`, `--limit`, `--json` |
| `stats [root]` | index info: models, described/embedded coverage, tags/types/clusters, dates, duplicates, reclaimable space | `--json` |
| `embed [root]` | (re)compute CLIP embeddings only | `--recompute`, `--clip-model`, `--batch` |
| `help [command]` | the full guide: every command, every flag with its default, and a runnable example | optional command name |

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
