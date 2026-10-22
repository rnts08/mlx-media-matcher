# CLIP spike (branch `spike/clip` — merged into `main`)

> Status: merged. `scan` now produces CLIP embeddings by default and VLM
> captions are added by `describe` or the complete `analyze` pass (best
> version of each duplicate group only); `search` defaults to CLIP when
> embeddings exist, `--fts` forces keyword search, and `similar` groups a tree
> by embedding similarity (or takes `--image` for a single-image query). This
> document records the investigation that led there.

Question: how would Phase 1 (find a person, find an object, group duplicates
across resolution) work with CLIP instead of a generative vision model?

## What was built

A pure-MLX CLIP stack, no PyTorch involved:

- `media_match/clip_model.py` — CLIP ViT + text transformer vendored from
  [mlx-examples/clip](https://github.com/ml-explore/mlx-examples/tree/main/clip)
  (MIT, Apple), adapted to load `mlx-community/clip-vit-base-patch32`
  (npz checkpoint + key remap).
- `media_match/clip_tokenizer.py`, `clip_processor.py` — vendored BPE
  tokenizer and 224px center-crop/normalize preprocessing.
- `media_match/clip.py` — `ClipEncoder`: batched, L2-normalized 512-dim
  image and text embeddings.
- CLI: `embed` (backfill embeddings into the index), `search --clip`
  (text→image ranking), `similar IMAGE` (image→image, indexed or
  computed on the fly). Embeddings live next to the VLM descriptions in
  `.media-match.db` (float32 BLOB, 2 KB/file), so both engines can be
  compared on the same index.

## How each Phase-1 goal maps to CLIP

| Goal | VLM approach (main) | CLIP approach |
|---|---|---|
| Find an object/person | describe image, then FTS/keyword search over text | embed query text, rank images by cosine similarity — no parsing, no keywords, returns a relevance score |
| Classify/group images | cluster extracted keywords | cluster / nearest-neighbor in one shared image+text vector space |
| Duplicates across resolution | pHash (kept) | embedding similarity also links them (0.87 for the 150px↔1200px pair) but degrades for very small images |
| Find the best version | pHash group + max pixels | unchanged — CLIP adds no resolution information |

## Measured on the 7-image test set (Apple M5, fp32)

- **Throughput: ~490 img/s (2 ms/img)** steady state vs ~0.9 s/img for the
  two-pass SmolVLM-256M description — **~450× faster**. A 10k-photo library
  takes ~20 s to embed instead of ~2.5 h.
- Text→image search ranked correctly on every query tried
  ("a person pouring coffee", "strawberries", "a forest with a lake",
  "a cute dog"), including paraphrases the VLM keyword index would miss.
- `similar dog_original.jpg` → `dog_thumb.jpg` first, cosine 0.871
  (next best 0.618).
- A stray **90×60** web thumbnail embedded on the fly: CLIP ranked it
  ambiguously (0.683 coffee vs 0.682 dog — effectively a tie), while pHash
  matched it to the coffee original with **hamming distance 0** (vs 30 to
  the dog photo).

## Findings

1. **CLIP is a much better *search/classification* engine than the 256M
   VLM**: ranked scores instead of parsed text, paraphrase robustness,
   image→image queries, two orders of magnitude faster, no prompt
   fragility (the small VLM needed two simple passes and keyword
   cleanup to be usable).
2. **CLIP is a worse *exact-duplicate* detector than pHash**, especially
   for small thumbnails: it sees "a photo of coffee" whether it is 224px
   or 90px, but its embedding drifts below its input resolution, whereas
   pHash canonicalizes everything to 32px first and stayed exact (0).
   CLIP similarity also cannot distinguish "same photo" from "another
   photo of the same subject" — which is a feature for semantic
   grouping but a bug for dedup.
3. **Storage cost is trivial**: 512-dim float32 = 2 KB/image (a 10k
   library = 20 MB), model download ~600 MB fp32 (quantizable later).

## Recommendation (hybrid)

Keep the three layers, drop nothing:

1. **pHash (existing)** — backbone for duplicate grouping and best-version
   selection; exact across resolutions, zero model dependency, microseconds.
2. **CLIP embeddings (this spike)** — primary search and semantic grouping
   layer: `search --clip` for people/objects, `similar` for "where else does
   this photo appear", nearest-neighbor clusters for "group images of the
   same subject" beyond exact duplicates.
3. **SmolVLM descriptions (existing)** — optional human-readable captions
   and FTS search; only worth running on the best version of each duplicate
   group (~1 s/image), which cuts VLM work dramatically on duplicate-heavy
   trees.
