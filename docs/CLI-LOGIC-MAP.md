# CLI logic map and command tree

Status: describes **0.4.7**. This is a map and a decision record, not a
breaking CLI change. Audio is inventory-only today; the planned extension is
in [`FEATURE-audio.md`](FEATURE-audio.md).

The short version:

- A **command** is a user goal or a lifecycle stage.
- A **flag** selects a variation of that goal: kind (`--video`, later
  `--audio`), scope (`--start-id`), output (`--json`) or safety
  (`--yes`, `--dry-run`).
- A flag must not silently run a *different* stage. `analyze` is the one
  deliberate aggregate; `scan` is indexing, and the individual stages remain
  commands because they are expensive, resumable and scriptable.

## 1. The model

Every run works on one **root** (a media tree) and usually one **index**
(`<root>/.media-match.db`).

There are four layers of work:

1. **Inventory** — walk the tree without decoding: path, size, mtime, type.
2. **Index** — read the bytes and derive searchable structure: SHA1, size,
   dimensions/duration, pHash or audio fingerprint, capture metadata, CLIP or
   audio embedding.
3. **Interpret** — add human-facing meaning: captions/transcripts, keywords,
   tags/facets, image types, person clusters.
4. **Query and act** — search, group, export, deduplicate, move to the Trash,
   consolidate folders, or reset the index.

Media kinds are `image`, `video` and (planned) `audio`. A kind is a flag on a
command, not a separate command family: `scan --video`, later `scan --audio`.
That keeps the lifecycle identical across media while letting each kind use
different primitives behind the same interface.

### Stages and what they write

| Stage | Command | Reads | Writes | Media files changed? |
|---|---|---|---|---|
| Inventory | `bootstrap` | tree `stat` | `bootstrap` table | only with `--prune`, `--media-only`, `--fix-extensions` |
| Loadability | `check` / `bootstrap` default | image decode / ffprobe | none | only after confirmation (`repair`/quarantine) |
| Index | `scan` / `scan --quick` | bytes, EXIF, ffprobe | `files`, `videos`, `video_frames` | no |
| Caption | `describe` | image/video frame | `description`, `keywords`, FTS | no |
| Embed backfill | `embed` | image bytes | `files.embedding` | no |
| Type | `classify` | existing CLIP embeddings | `files.type`, `type_score` | no |
| Tag refinement | `refine` | captions/keywords | `media_tags`, `raw_keywords` | no |
| People | `people` | person-facet embeddings | `clusters` | no |
| Duplicate report | `duplicates` | rows + hashes/fingerprints | nothing | no |
| Semantic report | `similar` | embeddings | nothing | no |
| Clean | `clean` | duplicate groups | deletes rows, `trash_log` | moves files to OS Trash |
| Folder work | `folders` | filesystem only | none | moves/merges folders |
| Metadata I/O | `export-metadata`, `import-metadata`, `clear-metadata` | index text fields | index text fields | no |
| Index dump/reset | `export`, `reset` | index | `reset` deletes the database | no |

`scan` already computes image CLIP embeddings. `embed` is therefore a backfill
command, not the normal way to embed a fresh tree. Videos are embedded by
`scan --video` / `analyze --video`; `embed` currently handles image rows only.

### Operation graph

```
bootstrap ──> check ──> scan ──> describe ──> refine ──> people ──> search / tags / similar
   │                     │           │
   │                     └──> embed ─┴──> classify
   │
   └──> optional file cleanup: --prune / --media-only / --fix-extensions

analyze = scan ──> describe ──> embed (backfill) ──> classify ──> refine
```

`analyze` is the complete pass. It is intentionally the only command that runs
several expensive stages in sequence. Every other stage command is idempotent
and can be resumed with `--start-id`/`--stop-id`:

- `scan` skips unchanged rows unless `--reanalyze` is given.
- `describe` only touches items with no description.
- `embed` only touches missing embeddings unless `--recompute` is given.
- `classify` only touches untyped images unless `--recompute` is given.
- `refine` rebuilds tags for the rows in scope from their current captions.

## 2. Command tree

Cross-cutting flags (`--db`, `--start-id`/`--stop-id`, `--json`,
`--no-progress`) are omitted below unless they are the point of the branch.

```
media-match
├── orientation
│   ├── `help` [COMMAND]
│   └── `about` [bench [ROOT] | ROOT]
│       ├── --db                    index to summarise
│       ├── bench [ROOT]            benchmark this machine instead
│       │   ├── --files N
│       │   ├── --repeat N
│       │   ├── --clip-model ID
│       │   └── --video
│       └── (0.4.x aliases, hidden: `info` [ROOT], `version`)
│
├── build the index
│   ├── `bootstrap` [ROOT]
│   │   ├── --prune                 exact duplicates, keep newest
│   │   ├── --media-only
│   │   │   ├── --move-to DIR       move non-media instead of trashing
│   │   │   └── --fix-extensions    detect missing media extensions
│   │   ├── --no-check              skip the default loadability pass
│   │   ├── --quarantine DIR
│   │   ├── --video
│   │   └── --yes
│   ├── `check` [ROOT]
│   │   ├── --video
│   │   ├── --quarantine DIR
│   │   └── --yes
│   └── `scan` [ROOT]
│       ├── --quick                 stat-only rows, stable ids
│       ├── --video
│       ├── --reanalyze
│       ├── --prune                 remove rows for files that vanished
│       ├── --limit N
│       └── legacy 0.4.x stage switches (hide in 0.5):
│           --describe, --refine, --classify
│
├── go deeper
│   ├── `analyze` [ROOT]              scan + describe + embed + classify + refine
│   │   └── --video, stage tuning, id window
│   ├── `describe` [ROOT]             caption missing descriptions
│   │   └── --video, --model, --describe-workers
│   ├── `embed` [ROOT]                backfill missing image embeddings
│   │   └── --recompute, --clip-model
│   ├── `classify` [ROOT]             type untyped images
│   │   └── --recompute, --clip-model
│   ├── `refine` [ROOT]               captions/keywords -> tags + facets
│   │   └── legacy --classify (prefer the classify command)
│   └── `people` [ROOT]               cluster person photos
│       └── --label N=Name, --threshold, --min-size
│
├── find things
│   ├── `search` [ROOT] [QUERY...]
│   │   ├── --clip | --fts
│   │   ├── --person / --object
│   │   ├── --type TYPE             browse by classified type
│   │   ├── --cluster NAME          browse a named person cluster
│   │   ├── --since / --until / --camera
│   │   └── --video / --image-only
│   ├── `tags` [ROOT]
│   │   └── --facets | --types | --raw, --video
│   └── `similar` [ROOT]
│       ├── default                 group semantic near-duplicates
│       ├── --image FILE            neighbours of one file
│       └── --video
│
├── quality and file management
│   ├── `duplicates` [ROOT]           report only
│   │   └── --video, --exact-phash, --threshold, --prefer
│   ├── `folders` [ROOT]              filesystem only
│   │   ├── --list
│   │   ├── --collapse
│   │   └── --replace | --rename, --dry-run, --yes
│   ├── `clean` [ROOT]                the destructive counterpart of duplicates
│   │   └── --dry-run, --yes, --video
│   └── `restore` [ROOT]              undo clean, best effort
│       └── --batch, --dry-run, --yes
│
├── metadata
│   ├── `export-metadata` [ROOT]      editable JSON, round-trips with import
│   ├── `import-metadata` [ROOT] FILE
│   └── `clear-metadata` [ROOT] [ITEM...]
│
└── index maintenance
    ├── `export` [ROOT]               dump the index: json / csv / md
    ├── `reset` [ROOT]                delete the database only
    └── `stats` [ROOT]                index statistics (media, compute, formats)
```

## 3. Command or argument?

The current surface is mostly right. The only structural smell is duplicated
stage selection.

| Surface | What it really is | Verdict |
|---|---|---|
| `scan --describe/--refine/--classify` | three different stages hidden as flags | **deprecate**. They duplicate `analyze` and violate "scan is indexing". Kept silently for 0.4.x scripts; remove from help and eventually the parser in a future breaking release. |
| `refine --classify` | the classify stage attached to refine | **deprecate** in favour of the standalone `classify`. Keep the old spelling for 0.4.x, but point docs and errors at `classify`. |
| `scan --quick` | a cheap mode of indexing | **keep as a flag**. Same goal, different cost. |
| `bootstrap --prune/--media-only/--fix-extensions` | inventory plus optional filesystem cleanup | **keep as flags**. They only make sense on the inventory walk. |
| `check` and `bootstrap`'s default check | diagnostic pass vs. first-pass flow | **keep both**. `check` is useful on its own; `bootstrap` runs it by default and `--no-check` opts out. |
| `search --type` / `--cluster` | browse/list modes, not text search | **keep for now**. Moving them would churn the CLI for little gain. If they move, use `tags --types` and `people --cluster NAME`. |
| `tags --facets/--types/--raw` | output modes over the same tag data | **keep as flags**. |
| `similar --image` / `--video` | query mode and media kind | **keep as flags**. |
| `duplicates` vs `clean` | report vs destructive action | **keep separate**. Never make `duplicates --clean`; the destructive step needs its own confirmation surface. |
| `folders --list/--collapse` | modes of one filesystem operation | **keep as flags**. |
| `people --label` | a write mode on the cluster listing | **keep as a flag**. The command is still "people". |
| `about` vs `info` vs `version` | three commands answering "what is this / what is installed" | **collapse into `about`** (0.4.7). `about [ROOT]` shows blurb, version, models, index summary, storage, support and licence; `about bench` benchmarks. `info`/`version` stay runnable as hidden 0.4.x aliases, not in the guide. |
| `stats` in "get oriented" | index statistics read like orientation | **move to index maintenance** (0.4.7). After the initial scans, stats changes only when the index is maintained, so it belongs next to `export` and `reset`. |
| `about bench` | benchmark topic of program information | **keep as a topic**. `bench` is not a media lifecycle stage. |
| `export-metadata` / `import-metadata` / `clear-metadata` | a metadata command family | **keep for now**. A `metadata export|import|clear` namespace is cosmetic and would break scripts. |
| `export` vs `export-metadata` | index dump vs editable metadata | **keep distinct names**. They answer different questions. |

### Decision rules for future commands

1. A new **goal** gets a command. A variation of an existing goal gets a flag.
2. A flag must not select a different lifecycle stage.
3. Cross-cutting selectors (`--db`, id windows, `--json`, `--no-progress`) stay
   flags on every command that can honour them.
4. Anything that modifies media gets confirmation, `--yes`, or a separate
   report command. `duplicates` reports; `clean` acts.
5. A command family is only worth a breaking change if the top-level list is
   genuinely hard to scan. Today it is not; the guide's workflow groups are
   doing that job.
6. New media kinds are kind flags (`--audio`), not new families such as
   `music scan` / `music search`. Audio and video should share the lifecycle.

## 4. Audio: the same map, different primitives

Audio already fits the inventory layer: `bootstrap` classifies audio
extensions and counts them, and `--media-only` treats them as media rather than
pruning them. Everything after inventory is deliberately absent until the
audio stage exists. See [`FEATURE-audio.md`](FEATURE-audio.md) for the schema,
pipeline, dedupe and implementation plan.

| Lifecycle step | Image | Video | Audio (planned) |
|---|---|---|---|
| Inventory | `bootstrap` | `bootstrap` | already: `kind="audio"` |
| Exact duplicate | SHA1 | SHA1 | SHA1 |
| Perceptual duplicate | pHash | keyframe pHash | Chromaprint/AcoustID fingerprint |
| Embedding | CLIP image vector | CLIP keyframe vectors | CLAP segment vectors |
| Interpretation | VLM caption + keywords | VLM mid-frame caption | Whisper transcript + CLAP/music tags |
| Type | zero-shot CLIP | (not implemented) | zero-shot CLAP category |
| Refinement | tags + facets | tags + facets | tags + music facets |
| Search | CLIP text→image, FTS | CLIP, FTS | CLAP text→audio, FTS |
| Dedupe/clean | `duplicates`/`clean` | `duplicates --video` | `duplicates --audio` |
| Loadability | Pillow diagnosis | ffprobe | ffprobe decode check |

The CLI shape stays the same: `scan --audio`, `analyze --audio`,
`describe --audio`, `classify --audio`, `search --audio`, `tags --audio`,
`similar --audio`, `duplicates --audio`, `clean --audio`, `check --audio`.
`bootstrap` needs no `--audio`: audio is part of the inventory and
`--media-only` keeps it.

## 5. Model loading and progress

Two cross-cutting rules apply to every stage:

1. **A model is loaded before the stage's progress display starts.** A
   first-run download or MLX kernel compile is a named step (`loading ...`,
   `model ready in ...`), not a pause hidden inside a bar.
2. **A stage never draws two displays at once.** The model spinner finishes,
   then the bar starts. When a stage ends, its bar leaves one final line:
   `✓` on success, `!` if any item failed.

`media_match.models.ModelCache` resolves and owns the models for one command
(flag, then config, then the index's `clip_model`/`vlm_model` metadata, then the
built-in default); `media_match.console` owns the terminal status and model-load animation;
`media_match.progress` owns the bars and the rolling ETA. The CLI keeps thin
aliases (`_load_model`, `_progress_enabled`, `_status`) so command code and
tests have one entry point.

| Command | Model work | Display |
|---|---|---|
| `scan` | CLIP before indexing; caption model before `--describe`; CLIP reloaded before `--classify` | index, describe and classify bars |
| `analyze` | CLIP before indexing; released while the caption model runs; caption model before describe; CLIP reloaded before embed/classify | index/describe/embed bars, then a classify spinner; no model load inside either |
| `describe` | caption model before the bar | describe bar |
| `embed` | CLIP before the bar | embed bar |
| `classify` / `refine --classify` | CLIP before the bar | classify bar |
| `search` | CLIP (index model, or `--clip-model`) before the text query | model spinner; query is a single forward pass |
| `similar` (group mode) | no model | grouping spinner |
| `similar --image` | CLIP only when the target is not in the index | model spinner |
| `duplicates` / `people` | no model | grouping/clustering spinner |
| `check` | no model | check bar |
| `bootstrap` | no model | inventory spinner, then check bar |

The ETA is extrapolated from the trailing **60 seconds** of completed items,
with the sample just before the window used as a baseline. It is deliberately
not a lifetime average: a pipeline that slows down halfway through a folder
should change its estimate. While a stage has less than a second of samples it
shows no rate or ETA rather than an absurd number. A `↗`/`↘` beside the rate
shows whether throughput is rising or falling.

`--no-progress`, `MEDIA_MATCH_NO_PROGRESS=1` and a non-terminal stderr all
disable the live block; the commands fall back to periodic one-line status
(and the scan heartbeat). This is a cross-cutting output flag, like `--json`,
not a lifecycle stage.

## 6. Where the code encodes this

| Concern | Location |
|---|---|
| extension registry / audio inventory | `media_match/bootstrap.py`, `media_match/pipeline.py` |
| stage pipeline | `media_match/pipeline.py` |
| describe pipeline | `media_match/analyze.py` |
| duplicate grouping | `media_match/groups.py`, `media_match/video_groups.py` |
| schema | `media_match/db.py` |
| CLI grammar and help groups | `media_match/cli.py` (`_COMMAND_GROUPS`, parsers) |
| model selection / preload lifetime | `media_match/models.py` |
| status, model-load animation, progress flag | `media_match/console.py` |
| rolling ETA, bar rendering, spinner | `media_match/progress.py` |
| audio design | [`FEATURE-audio.md`](FEATURE-audio.md) |

Code comments marked `TODO(audio)` point at the seams that must change first.
They are intentionally comments, not dead code: the current scan must never
return an audio job until `prepare()` and `store_result()` have real audio
branches, or an MP3 would fall through the video path and be misindexed.
