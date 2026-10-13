# Media Match

## Phase 1 (implemented)

Analyze all image files in a folder with a vision agent, classify the images so that it is easy to find a certain person, an object and identify duplicates.

This should work across resolution, so that accidentally saved thumbnails are identified as duplicates and grouped with their primary file that should have a better resolution. 

### Usage (Phase 1)

```bash
python3 -m venv .venv
.venv/bin/pip install -e .

# 1. index a photo tree (downloads the ~150 MB MLX vision model on first run)
.venv/bin/media-match scan ~/Pictures

# 2. find images by person / object / free text (searches model descriptions + keywords)
.venv/bin/media-match search ~/Pictures --person "woman red jacket"
.venv/bin/media-match search ~/Pictures --object bicycle
.venv/bin/media-match search ~/Pictures "birthday cake"

# 3. list duplicate groups; the best-resolution version is marked with *
.venv/bin/media-match duplicates ~/Pictures
.venv/bin/media-match duplicates ~/Pictures --json      # machine-readable
.venv/bin/media-match duplicates ~/Pictures --threshold 4  # stricter matching

# extras
.venv/bin/media-match tags ~/Pictures      # keyword cloud with counts
.venv/bin/media-match stats ~/Pictures     # index info
```

Notes:

- Model: `mlx-community/SmolVLM-256M-Instruct-4bit` by default (the smallest viable
  MLX vision model, ~1 s/image on Apple Silicon). For better accuracy on larger
  machines, pass `--model mlx-community/Qwen2-VL-2B-Instruct-4bit` (or any other
  MLX-converted VLM) to `scan`.
- Duplicate detection uses 64-bit perceptual hashes (pHash), which are
  resolution- and re-encode-independent, so thumbnails, WebP conversions and
  recompressed copies group with their full-resolution original. Byte-identical
  files are always grouped. The largest image in a group is the primary.
- The index is stored in `<root>/.media-match.db` (SQLite, FTS5 search).
  `scan` is incremental: unchanged files are skipped; `--reanalyze` forces a redo.
- Supported formats: JPEG, PNG, WebP, GIF, BMP, TIFF, HEIC/HEIF.


## Phase 2
The same as above, but on video files. 

## Spike: CLIP (branch `spike/clip`)

An alternative/complementary CLIP-embedding engine (pure MLX, ~490 img/s):
`media-match embed`, `media-match search --clip "..."`,
`media-match similar IMAGE`. See `docs/clip-spike.md` for the comparison
against the generative-VLM approach and the recommended hybrid.

## Goal
The goal is to group images of different people and objects together and to be able to find the best version of an image that might be duplicated in the tree the program is working on. 

