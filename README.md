# Media Match

## Phase 1 (implemented)

Analyze all image files in a folder with a vision agent, classify the images so that it is easy to find a certain person, an object and identify duplicates.

This should work across resolution, so that accidentally saved thumbnails are identified as duplicates and grouped with their primary file that should have a better resolution. 

### Usage (Phase 1)

```bash
python3 -m venv .venv
.venv/bin/pip install -e .

# 1. index a photo tree: hash + perceptual hash + CLIP embeddings (~500 img/s)
.venv/bin/media-match scan ~/Pictures

# 1b. optional: add human-readable captions with the vision model (~1 s/image).
#     Captions are generated for the best-resolution version of each duplicate
#     group only, so thumbnails are never described twice.
.venv/bin/media-match scan ~/Pictures --describe

# 2. find images by person / object / free text.
#    CLIP vector search by default; --fts searches captions/keywords instead.
.venv/bin/media-match search ~/Pictures "woman red jacket"
.venv/bin/media-match search ~/Pictures --object bicycle
.venv/bin/media-match search ~/Pictures "birthday cake" --fts

# 3. semantic near-duplicate groups across the tree (CLIP cosine similarity;
#    also links different photos of the same subject, unlike `duplicates`)
.venv/bin/media-match similar ~/Pictures
# ...or nearest neighbors of one image (indexed, or any standalone file)
.venv/bin/media-match similar ~/Pictures --image /tmp/some-photo.jpg

# 4. list duplicate groups; the best-resolution version is marked with *
.venv/bin/media-match duplicates ~/Pictures
.venv/bin/media-match duplicates ~/Pictures --json      # machine-readable
.venv/bin/media-match duplicates ~/Pictures --threshold 4  # stricter matching

# extras
.venv/bin/media-match tags ~/Pictures      # keyword cloud from captions
.venv/bin/media-match stats ~/Pictures     # index info
.venv/bin/media-match embed ~/Pictures     # backfill CLIP embeddings only
```

Architecture (hybrid — the three layers cover each other's weaknesses):

- **CLIP embeddings** (`mlx-community/clip-vit-base-patch32`, pure MLX,
  ~500 img/s, 2 KB/image) — default search and semantic similarity engine.
  Supports text→image queries, paraphrases and image→image search (`similar`).
- **Perceptual hashes** — resolution- and re-encode-independent duplicate
  detection: thumbnails, WebP conversions and recompressed copies group with
  their full-resolution original (byte-identical files always group). The
  largest image in a group is the primary. Exact even for tiny thumbnails
  where CLIP embeddings degrade (see `docs/clip-spike.md`).
- **Vision-model captions** (optional, `--describe`; default
  `mlx-community/SmolVLM-256M-Instruct-4bit`) — human-readable descriptions
  and keywords for keyword/FTS search (`search --fts`, `tags`). For better
  accuracy pass `--model mlx-community/Qwen2-VL-2B-Instruct-4bit` or any
  other MLX-converted VLM.

Other notes:

- The index is stored in `<root>/.media-match.db` (SQLite, FTS5 search).
  `scan` is incremental: unchanged files are skipped; `--reanalyze` forces a redo.
- Supported formats: JPEG, PNG, WebP, GIF, BMP, TIFF, HEIC/HEIF.


## Phase 2
The same as above, but on video files. 

## Goal
The goal is to group images of different people and objects together and to be able to find the best version of an image that might be duplicated in the tree the program is working on. 

