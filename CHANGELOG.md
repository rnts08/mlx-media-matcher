# Changelog

media-match uses semantic versioning, tuned to how the tool actually grows:

- **major** — reserved for breaking changes to the CLI or the index format.
- **minor** — a large change in operations or functionality. `0.1.0` was the
  image-only tool; `0.2.0` added video indexing with the same machinery.
- **patch** — every feature or fix inside one era. New functionality lands as
  its own commit and bumps the patch number.

The entries below were recorded retroactively so the numbering reads
naturally against the work that produced it.

## 0.2.x — video support

- **0.2.17** feat: `--json` output for `search`, `tags`, `stats`, `similar` and `clean --dry-run`
- **0.2.16** feat: `export` the index to JSON, CSV or Markdown
- **0.2.15** feat: `restore` command — undo `clean` by moving files back from the OS Trash
- **0.2.14** feat: `search --since/--until/--camera`, show capture dates, `duplicates --prefer date`, `stats` date range
- **0.2.13** feat(exif): index capture time, camera and GPS from image EXIF
- **0.2.12** feat: report reclaimable space (extra-copy bytes) in `duplicates` and `stats`
- **0.2.11** fix(db): clear orphaned `media_tags`/clusters on delete; WAL + busy timeout; schema versioning; UTC timestamps; skip dotfiles on scan
- **0.2.10** fix(cli): drop the program blurb from help output; it lives in `about` now
- **0.2.9** feat(cli): page `about` through `$PAGER`/`less`/`more` and lead with the feature blurb
- **0.2.8** feat(cli): show "Support development" details (crypto addresses, Buy Me a Coffee, Ko-fi) in `info`
- **0.2.7** fix(scan/video): skip missing media, add `scan --prune`, report concise ffmpeg errors
- **0.2.6** feat(cli): no-arg help, `version`/`info`/`about` commands, richer `stats`
- **0.2.5** feat: add the `help` command and a detailed per-command guide
- **0.2.4** feat: `scan --refine/--classify` to keep tags and image types fresh
- **0.2.3** feat: refine tags into facets/types and cluster people by similarity
- **0.2.2** perf: bound the scan prep queue so only two batches are in flight
- **0.2.1** perf: stream scan results, scale video grouping, fail fast without ffmpeg
- **0.2.0** feat: Phase 2 — video indexing, aligned duplicate grouping, fed-GPU scan pipeline

## 0.1.x — image-only

- **0.1.10** clean: `e` executes the decisions made so far, then keeps reviewing
- **0.1.9** tui: pin the key legend at the top; confirm before quitting with pending trash
- **0.1.8** clean: `a`/`A` keys to mark a group (or everything) as kept
- **0.1.7** clean: fix arrow-key parsing, bare-ESC hang, and the post-trash size report
- **0.1.6** clean: visual bar UI with Quick Look/Preview and in-session trashing
- **0.1.5** clean: view copies in the OS image viewer while reviewing
- **0.1.4** add `clean`: interactive duplicate review with OS Trash support
- **0.1.3** fix the `similar` CLI footgun; never create empty indexes implicitly
- **0.1.2** hybrid default: CLIP-first scan/search, VLM captions opt-in via `--describe`
- **0.1.1** CLIP embeddings on MLX (`embed`, `search --clip`, `similar`)
- **0.1.0** Phase 1: MLX vision-model photo indexing, search and duplicate grouping
