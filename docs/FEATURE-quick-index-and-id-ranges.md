# FEATURE — quick index and ID-range selection

Status: implemented and shipped in **0.4.0**.

## Request (verbatim)

> yes craete a quick-scan option to populate the database, after that you can
> add range-limits to every function that use the database, instead of
> --limit 1000 only then add --startid 1001 which will scan the NEXT 1000 files
> etc or just no --limit at all, only work on --startid 353 --stopid 905, do you
> understand?

## Why

A full `scan` is expensive: it reads every byte (SHA1), decodes every image,
computes pHash and CLIP embeddings. On a large library the user cannot get an
index at all without paying for the whole analysis, and `--limit N` can only
start at the beginning — every rerun repeats the first N files.

Two changes fix that:

1. `scan --quick` walks the tree and writes **basic rows** with stable IDs,
   without reading file contents. It is a `stat` pass, not an analysis pass.
2. DB-backed commands accept an inclusive **ID window**
   (`--start-id`/`--stop-id`, aliases `--startid`/`--stopid`). IDs come from the
   quick pass, so work can be chunked: analyze 1–1000, then 1001–2000, etc.

## Quick scan

`media-match scan ROOT --quick`:

- Walks media exactly like a normal scan (hidden directories skipped,
  `.media-matchignore` respected, videos only with `--video`).
- Inserts one row per media file into `files` / `videos` with:
  - `path`, `size`, `mtime`
  - `sha1 = ''`, `phash = ''`, `width = 0`, `height = 0`
  - `embedding NULL`; video signature/duration/fps/codec/frame_count defaults
- `ON CONFLICT(path) DO NOTHING`: an existing analysed row is never downgraded,
  and a changed file is left for a normal scan to notice through size/mtime.
- Does not load CLIP, Pillow, EXIF or ffprobe. No ffmpeg requirement.
- `--limit N` inserts at most N **new** rows, so repeated runs advance through
  the tree instead of repeating the first N files.
- `--prune` still reconciles the whole index (ID bounds do not apply to
  pruning).
- Incompatible with `--describe`, `--refine`, `--classify`, `--reanalyze` and
  the ID window: those all need analysis or already-existing rows.

A later normal `scan` treats rows with `sha1 == ''` as incomplete even when
size and mtime match, and fills them in. IDs are preserved by the existing
`ON CONFLICT(path) DO UPDATE`.

### Guard rails for basic rows

- Duplicate detection (`find_duplicate_groups`, `find_video_duplicate_groups`)
  ignores rows with an empty SHA1/pHash/signature. Without this, every quick row
  would look byte-identical and `clean` could delete files.
- `describe_targets` uses the same guarded grouping, so a quick row is
  described as a singleton rather than shadowed by a false duplicate.
- `stats` may count basic rows as indexed media; their `0x0` dimensions are not
  treated as real resolution by grouping because they never enter a group.

## ID windows

Shared flags on every command that selects media rows:

```
--start-id N   only rows with id >= N (inclusive)
--stop-id N    only rows with id <= N (inclusive)
--startid N    alias for --start-id
--stopid N     alias for --stop-id
```

Semantics:

- Positive integers only; `start > stop` is a usage error (status 2).
- `files.id` and `videos.id` are independent sequences. One numeric window
  selects images and videos in that range **separately**. It is not a global
  ordering across the two tables.
- No flags means "all rows", exactly as before.
- `--limit` keeps its meaning as a count. With `--start-id`, scan/analyze select
  indexed rows in ID order and then take at most `--limit` rows; this makes
  `--start-id 1001 --limit 1000` the next window. `--stop-id` sets the end
  explicitly. For ranked commands (`search`, `similar`) `--limit` caps the
  output within the window.
- A range selects **indexed rows only**. New files discovered on disk are not
  in a window; run `--quick` first (or a normal scan without a range).
- `--prune` remains index-wide even when an analysis window is given.

### Commands gaining a window

| Command | Row source | Behaviour |
|---|---|---|
| `scan` | indexed rows in range, or the filesystem walk without a range | analyse only rows in the window; incomplete quick rows are included |
| `analyze` | same as scan plus describe/refine/classify | the window applies to every stage |
| `describe` | `describe_targets` | group and caption only in-range rows |
| `embed` | missing-embedding query / recompute query | embed only in-range images |
| `similar` | embeddings + records | rank/group only in-range candidates (the `--image` target may be outside) |
| `search` | FTS/CLIP/type/cluster | constrain candidates, then apply `--limit` |
| `duplicates` | `all_files` / `all_videos_with_frames` | groups formed inside the window |
| `clean` | same | only in-window files can be selected for the Trash |
| `export` | `all_files` + `all_videos` | export in-window rows |
| `export-metadata` | per-kind `SELECT *` | export in-window rows |
| `clear-metadata` | raw id list / path resolution | clear only in-window rows |
| `import-metadata` | path resolution | overwrite only in-window rows |
| `refine` | `rows_for_refinement` | refine in-window rows; tag deletion is scoped to those ids |

### Deliberate exclusions

- `restore` — operates on `trash_log.id`, not media ids.
- `bootstrap` — writes the separate, re-numbered `bootstrap` table.
- `people` — `set_clusters` rebuilds the whole cluster table; a partial run
  would erase assignments outside the window.
- `tags`, `stats`, `info` — whole-index aggregates / metadata.
- `folders`, `check`, `reset` — no media rows.

These are documented in the help and README so the flag set is predictable.

## Examples

```
media-match scan ~/Pictures --quick          # IDs for everything, stat-only
media-match scan ~/Pictures --quick --limit 50000   # first 50k new rows
media-match scan ~/Pictures --start-id 1 --stop-id 1000
media-match scan ~/Pictures --start-id 1001 --limit 1000
media-match describe ~/Pictures --startid 353 --stopid 905
media-match embed ~/Pictures --start-id 1 --stop-id 5000
media-match duplicates ~/Pictures --start-id 1 --stop-id 5000
media-match clean ~/Pictures --start-id 1 --stop-id 5000 --dry-run
```

## Safety notes

- A range-scoped duplicate decision can pick a lower-resolution "best" if the
  true best is outside the window. `duplicates`/`clean` therefore stay
  range-scoped by design; use `--dry-run` first.
- Ranges never create work for unindexed files, and never touch rows outside
  the window.
- Quick rows have no hashes, so `duplicates`/`search --clip` ignore them until
  a normal scan or `embed` runs.

## Tickets

### T1 — Basic rows
- DB insert helpers for quick image/video rows (`ON CONFLICT DO NOTHING`)
- unit tests: empty hashes, stable ids on rerun, no downgrade of an analysed row

### T2 — `scan --quick`
- flag, walk, limit semantics, incompatible-flag validation
- no CLIP/ffprobe imports on this path
- tests: new tree, rerun, `--limit` advance, `--video`, `--prune`

### T3 — Completion detection
- `build_jobs` treats `sha1 == ''` as incomplete and forces analysis
- tests: quick row at unchanged mtime becomes a job; complete row is up to date

### T4 — Shared window
- `_add_id_range`, aliases, positive-int validation, `start <= stop`
- DB helpers and SQL predicates applied to both tables independently
- tests: bounds inclusive, sparse ids, invalid bounds status 2

### T5 — Processing commands
- scan/analyze range mode in ID order, then `--limit`
- describe/embed/similar/search/refine filters
- tests: only in-range rows change; output counts are honest

### T6 — Dedupe and clean
- empty-hash guard in both groupers (also fixes quick-row grouping)
- range-scoped input records; fix the `duplicates --video` branch overwrite
- tests: quick rows never group; image/video ranges

### T7 — Metadata and export
- export, export-metadata, clear-metadata, import-metadata
- tests: in-window only; explicit paths outside the window are ignored

### T8 — Docs and release
- README command table and a chunking section
- CHANGELOG 0.4.0; `pyproject.toml` and `_VERSION`
- help text for `--quick` and both range flags on every supported page

## Implemented deltas (as shipped in 0.4.0)

Everything above shipped, with these concrete details:

- `scan --quick` on a missing root exits 2 with `No folder at <root>.`, matching
  `folders`.
- Window validation: argparse rejects non-positive ids (`expected a positive
  integer`, exit 2); an inverted window (`--start-id > --stop-id`) is rejected
  once in `main()` with status 2 before any command opens an index.
- Window row selection loads `files` then `videos` (each ordered by id), caps
  the window with `--limit`, and reports
  `N indexed row(s) in the id window, M up to date, K to analyze (new files are
  not scanned)`. `analyze` shares this path, and every later stage (describe,
  refine, classify) receives the same bounds.
- `describe_targets` takes `start_id`/`stop_id`; its `pending` counts are
  window-scoped by construction because the candidate records are filtered
  before grouping.
- `refine` tag deletion is scoped to the ids the run is rebuilding. This also
  fixes a latent bug: the old code deleted every tag of a kind even when the
  run had produced no items.
- The `duplicates --video` branch computed video groups and then unconditionally
  overwrote them with image groups; fixed while wiring the window (its `else`
  was missing, `clean` already had it right).
- Duplicate groupers drop rows with an empty SHA1/pHash (images) or
  SHA1/keyframes (videos), so basic rows can never form groups — this is what
  makes `clean` safe right after `--quick`.
- `search` honours the window through the existing filter plumbing, so FTS,
  CLIP, `--type` and `--cluster` all constrain candidates before `--limit`.
- `clear-metadata`/`import-metadata` report explicit paths whose resolved id
  falls outside the window as `outside id window: <rel>` and leave them alone.
- Tests: 26 new cases in `tests/test_quick_index.py`; full suite 346 passed,
  1 skipped.
- Docs: README usage block (1a), a table row note for `scan --quick`,
  `--start-id`/`--stop-id` on every supported row, an "ID windows" section, and
  `help scan` / `help analyze` prose. Version 0.4.0 in `pyproject.toml` and
  `_VERSION`, with a `0.4.x` CHANGELOG section.

## Acceptance

1. `scan --quick` on a fresh tree creates one row per media file with empty
   hashes and stable IDs, without loading CLIP or ffprobe.
2. Running it again inserts no rows; `--limit N` advances over reruns.
3. A normal `scan` analyses quick rows even when size/mtime already match.
4. Every media-row command in the table above honours an inclusive ID window on
   both `files.id` and `videos.id`; rows outside it are untouched.
5. `--start-id 1001 --limit 1000` selects the next window, and
   `--start-id 353 --stop-id 905` selects exactly that inclusive range.
6. Invalid windows exit 2 with a clear message.
7. Quick rows never participate in duplicate/clean grouping.
8. Help, README and CHANGELOG describe the exclusions and the separate
   image/video ID sequences.
