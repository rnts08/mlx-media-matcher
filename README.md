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
- **Phase 2 (video): planned** — the same capabilities for video files.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

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
#    Opens a visual bar UI (arrow keys or hjkl to move):
#      space = keep the file under the bar (unselects the group's others)
#      v     = Quick Look the file       p = open it in Preview.app
#      x     = trash this file now (y/n) enter = trash all unkept files (y/n)
#      q     = quit without doing anything
#    The algorithm's best copy of each group starts selected. Without a
#    terminal it falls back to line-based prompts.
.venv/bin/media-match clean ~/Pictures
.venv/bin/media-match clean ~/Pictures --dry-run  # preview the plan
.venv/bin/media-match clean ~/Pictures --yes      # accept all suggestions

# extras
.venv/bin/media-match tags ~/Pictures    # keyword counts from captions
.venv/bin/media-match stats ~/Pictures   # index, models, coverage
.venv/bin/media-match embed ~/Pictures   # backfill CLIP embeddings only
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
| `scan [root]` | index new/changed images (hash + pHash + CLIP). Incremental: unchanged files are skipped | `--describe`, `--model`, `--clip-model`, `--limit`, `--reanalyze`, `--batch` |
| `search [root] [query...]` | find images. CLIP by default when embeddings exist | `--person`, `--object`, `--fts`, `--clip`, `--limit` |
| `similar [root]` | group the tree into semantic near-duplicate clusters | `--threshold` (cosine, default 0.85) |
| `similar [root] --image F` | nearest neighbors of one image (indexed, or any standalone file) | `--limit` |
| `duplicates [root]` | group exact/near duplicates via pHash + sha1; best resolution wins | `--threshold` (hamming 0–64, default 8), `--json` |
| `clean [root]` | review duplicate groups in a visual bar UI (hjkl/arrows, space=keep, v=Quick Look, p=Preview, x=trash file, enter=dedupe) and move rejected copies to the system Trash (never deleted outright; index updated); falls back to line prompts without a TTY | `--yes`, `--dry-run`, `--threshold` |
| `tags [root]` | keyword counts from captions | `--limit` |
| `stats [root]` | index info: models, described/embedded coverage | |
| `embed [root]` | (re)compute CLIP embeddings only | `--recompute`, `--clip-model`, `--batch` |

All commands take `--db` to override the index location. `root` defaults to
the current directory. Supported formats: JPEG, PNG, WebP, GIF, BMP, TIFF,
HEIC/HEIF.

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

## Phase 2 (planned)

The same as above, but on video files: find videos by person/object content,
group duplicates and re-encodes across resolution, and identify the best
version — e.g. by sampling frames, embedding keyframes with CLIP, hashing
frames for cross-resolution matching, and aggregating frame-level results
into per-video scores.

## Goal

The goal is to group images of different people and objects together and to
be able to find the best version of an image that might be duplicated in the
tree the program is working on.

## Development

- `docs/clip-spike.md` — the CLIP investigation and benchmarks behind the
  hybrid design (branch `spike/clip`, merged).
- Test tree used during development: JPEG/WebP thumbnails of real photos,
  validated for grouping at hamming distance 0 across formats and sizes.
