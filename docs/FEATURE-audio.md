# FEATURE — audio: fingerprint, interpret, categorize, deduplicate

Status: **design only**. Audio is inventoried today; it is not scanned,
indexed, transcribed, embedded or deduplicated yet. This document is the
contract for adding it without forking the CLI into a separate "music" tool.

## Request

> make provisions for (todo:s and comments in the code) music/audio files to be
> treated the same, fingerprinted, interpreted, categorized and deduplicated.

## Current state

- `bootstrap` recognizes `.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.oga` and
  `.m4a`, records them as `kind="audio"`, and keeps them when `--media-only`
  prunes non-media files.
- `detect_media_type` returns `.mp3` for an audio stream and can add a missing
  extension.
- `scan` ignores audio: `pipeline.iter_media()` returns images and, with
  `--video`, videos only.
- There is no `audio` table, fingerprint column, audio embedding, transcript,
  audio tag/facet, audio duplicate group or `--audio` flag.
- The CLI only mentions audio in the `bootstrap` inventory count.

The silence is intentional in the current code. Letting an audio file into
`iter_media()` before the scan has an audio branch would send it through
`probe()` / `extract_frames()` as if it were a video.

## Goals

1. **Same lifecycle.** Audio uses the same bootstrap → scan → describe →
   classify → refine → search → deduplicate/clean sequence as images and
   videos. The kind is selected with `--audio`, not a new command family.
2. **Cheap first pass.** Inventory stays `stat`-only. Expensive work is opt-in
   through `scan --audio` / `analyze --audio`.
3. **Two layers of identity.** Exact SHA1 for byte-identical files, plus a
   perceptual audio fingerprint for transcodes and container changes.
4. **Local interpretation.** Speech becomes a transcript; music gets
   acoustic tags, ID3/Vorbis metadata and a CLAP embedding. No cloud call is
   required.
5. **Safe cleanup.** Audio deduplication reuses `duplicates` / `clean`:
   report first, move to the OS Trash only after confirmation, never delete.
6. **Idempotence.** Re-running a stage only fills missing work unless
   `--reanalyze` / `--recompute` is explicit.

## Extension registry

Audio extensions are part of the same media registry as images and videos:

```
.mp3  .wav  .flac  .aac  .ogg  .oga  .m4a
```

`bootstrap.CANONICAL_EXTENSIONS` already contains them. The planned cleanup is
one shared registry in `pipeline.py` (`IMAGE_EXTENSIONS`, `VIDEO_EXTENSIONS`,
`AUDIO_EXTENSIONS`) so a new extension cannot be recognized by one command and
forgotten by another.

## Proposed schema (v4)

Do not reuse `videos` or `video_frames`: an audio file has no frame geometry
and its dedupe primitive is a fingerprint, not a pHash. Add audio-specific
tables and let `media_tags.kind = 'audio'` reuse the existing tag/facet table.

```sql
CREATE TABLE audio (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha1 TEXT NOT NULL,
    duration REAL NOT NULL DEFAULT 0,
    codec TEXT NOT NULL DEFAULT '',
    container TEXT NOT NULL DEFAULT '',
    bitrate INTEGER NOT NULL DEFAULT 0,
    sample_rate INTEGER NOT NULL DEFAULT 0,
    channels INTEGER NOT NULL DEFAULT 0,
    title TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    album TEXT NOT NULL DEFAULT '',
    track TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL DEFAULT '',
    genre TEXT NOT NULL DEFAULT '',
    fingerprint BLOB,             -- Chromaprint raw fingerprint
    fingerprint_duration REAL NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '',
    embedding BLOB,               -- whole-track CLAP vector, optional
    scanned_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_audio_sha1 ON audio(sha1);

CREATE TABLE audio_segments (
    id INTEGER PRIMARY KEY,
    audio_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    t REAL NOT NULL,              -- start time in seconds
    duration REAL NOT NULL,
    embedding BLOB,               -- CLAP vector for this segment
    energy REAL
);
CREATE INDEX idx_audio_segments_audio ON audio_segments(audio_id);

CREATE VIRTUAL TABLE audio_fts USING fts5(
    title, artist, album, description, keywords,
    content='audio', content_rowid='id', tokenize='trigram'
);
```

Notes:

- `fingerprint` is a variable-length byte string from `fpcalc -raw`, not a
  fixed 64-bit hash. Do not compare it with `hamming()` from `hashing.py`.
- `audio_segments` is the audio counterpart of `video_frames`: a long track
  gets one CLAP vector per window (for example 10 s with 5 s hop, capped).
- `audio_fts` mirrors the image/video FTS design and indexes metadata plus the
  transcript/summary in `description`/`keywords`.
- `AudioRecord` should duck-type the fields the existing group/cleanup/TUI code
  needs (`id`, `path`, `size`, `mtime`, `sha1`, `description`, `keywords`) and
  expose `duration`, `fingerprint`, `bitrate`, `sample_rate`, `channels`.
  `width` / `height` should not be invented for audio; the group code needs an
  explicit audio path rather than pretending audio is a 0×0 image.

## Pipeline

### Inventory

`bootstrap` already does this: the inventory counts audio and keeps it when
`--media-only` prunes non-media files. With no `--audio` flag, bootstrap still
records audio. `--fix-extensions` should keep returning canonical audio
extensions.

### Scan (`scan --audio` / `analyze --audio`)

Use the existing staged pipeline, with `ScanJob.kind = "audio"`:

1. **Reader / SHA1** — stream the file; no need to hold it in memory.
2. **Probe** — `ffprobe` for duration, codec, container, bitrate, sample rate
   and channels. Reuse the existing ffmpeg/ffprobe requirement.
3. **Tags** — read ID3/Vorbis/MP4 tags (mutagen, or ffprobe `format_tags`).
   Normalize a small set: title, artist, album, track, year, genre.
4. **Fingerprint** — run Chromaprint (`fpcalc -raw`) and store the raw
   fingerprint plus its duration. Fingerprinting is CPU/IO work and belongs on
   the CPU stage. If `fpcalc` is unavailable, index the file with a NULL
   fingerprint and say so; do not fail the whole scan.
5. **Decode + embed** — decode mono PCM (ffmpeg pipe, no temp files), make CLAP
   mel windows, and run the audio encoder in batches on the GPU. Store one
   vector per window in `audio_segments`; optionally mean-pool into
   `audio.embedding` for whole-track similarity.
6. **Store** — `upsert_audio()` plus `set_audio_segments()`, with the same
   `ON CONFLICT(path) DO UPDATE` and FTS refresh shape as images/videos.

The pipeline must keep CPU decode/probe/fingerprint overlapped with GPU CLAP
batches, exactly as video keyframes overlap CLIP. Long tracks should be capped
like videos (`--max-frames` analogue: `--max-audio-windows`, default e.g. 120).

### Describe (`describe --audio`)

Audio has two useful "captions":

- **Speech-like** audio: a Whisper-family MLX model produces a transcript. The
  transcript becomes `description`; a text model or the existing keyword
  extractor produces `keywords`.
- **Music / sound** audio: no transcript. Use CLAP zero-shot tags and the
  embedded metadata to build a short description such as
  `"music, piano, calm, 3:42, Artist — Album"`, with the same keyword format as
  images so `search --fts` and `refine` work unchanged.

The describe stage should choose by CLAP category/energy: if speech likelihood
is high, transcribe; otherwise generate acoustic tags. A hybrid file (podcast
with music) can carry both.

### Classify (`classify --audio`)

Zero-shot CLAP categories, in the same spirit as `imagetype.py`:

```
music, speech, podcast, audiobook, sound effect, ambient, silence,
instrument, live recording
```

Genre, mood and instrument tags are refinement inputs, not hard types. Store
the top category in an `audio.type` / `audio.type_score` pair (mirroring
`files.type`) or in `media_tags` with a dedicated facet; choose one before
implementation and keep `tags --types` working.

### Refine (`refine --audio`)

Reuse `tags.py`. Add music-relevant facets:

```
artist, album, genre, mood, instrument, activity
```

`media_tags.kind = 'audio'` already works. `people` remains image-only; audio
has no person clusters.

### Search (`search --audio`)

- **Text search** — FTS over title, artist, album, transcript, description and
  refined tags.
- **Acoustic text search** — CLAP text→audio: embed the query and rank
  `audio_segments` (best moment, like video).
- **Metadata filters** — artist/album/genre/year, duration range, kind.
- **Similarity** — `similar --audio` groups by whole-track or segment CLAP
  vectors; `similar --audio --image` is not meaningful, but
  `similar --audio --file TRACK` is (a future spelling).

### Deduplicate (`duplicates --audio` / `clean --audio`)

Two separate questions, kept separate in the report:

1. **Same recording, different encode/container** — exact SHA1 catches
   byte-identical files; Chromaprint catches MP3 vs FLAC vs M4A, different
   bitrates and small trims. Compare fingerprints by bit error rate over an
   alignment window, require a minimum overlap, and expose a threshold
   (`--audio-threshold`) with a conservative default. This is the group
   `clean --audio` may trash.
2. **Same song, different recording** — live versions, remixes and covers are
   *similar*, not duplicates. Put them in `similar --audio` or a separate
   report section, never in the destructive group.

Best-version policy for audio: lossless over lossy, then higher bitrate, then
higher sample rate/channels, then longer duration as a tie-break. Do not use
pixel count. The existing `_reclaimable()` and `print_plan()` helpers assume
images/videos; add an audio branch or a small `AudioRecord` adapter.

### Check (`check --audio`)

Probe the container and decode enough audio to catch truncation/corruption.
Reuse `ffprobe` for metadata and a short `ffmpeg -f null -` decode for
loadability. Report `ok` / `truncated` / `corrupt` / `notable` consistently
with images and videos. Quarantine only on confirmation.

### Export, stats and metadata

- `export` and `export-metadata` gain `kind: "audio"` rows with duration,
  artist/album and the same description/keywords fields.
- `import-metadata` / `clear-metadata` accept `--kind audio`.
- `stats` adds audio counts, total duration, fingerprints present, transcripts
  present and reclaimable duplicate bytes.
- `reset` needs no change.

## CLI matrix (planned)

| Command | Audio spelling | Notes |
|---|---|---|
| `bootstrap` | no flag | audio already inventoried, kept by `--media-only` |
| `scan` | `--audio` | probe, fingerprint, CLAP segments |
| `analyze` | `--audio` | scan + describe + embed + classify + refine |
| `describe` | `--audio` | transcript or acoustic description |
| `embed` | `--audio` | backfill CLAP segments |
| `classify` | `--audio` | CLAP categories |
| `refine` | `--audio` | music facets/tags |
| `search` | `--audio` | FTS + CLAP + metadata filters |
| `tags` | `--audio` | tags/facets/type counts |
| `similar` | `--audio` | whole-track/segment grouping |
| `duplicates` | `--audio` | fingerprint groups, report only |
| `clean` | `--audio` | trash confirmed duplicates |
| `check` | `--audio` | probe/decode loadability |
| `export` | includes audio | no flag needed if it exports every kind |
| `export-metadata` / `import-metadata` / `clear-metadata` | `--kind audio` | same metadata contract |
| `people` | not applicable | image-only |

This is deliberately the same shape as `--video`. A `music`/`audio` command
family would duplicate every stage and violate the command/flag rule in the
[CLI logic map](CLI-LOGIC-MAP.md).

## Dependencies and model choices (open)

- **Chromaprint** — prefer the `fpcalc` executable (already a small external
  dependency, like ffmpeg) over a Python binding, with a clear "fingerprint
  skipped" message when it is absent. AcoustID web lookup is optional and
  **off by default**; local fingerprinting must never require a network.
- **CLAP on MLX** — choose a converted checkpoint and verify it loads through
  the existing MLX stack. If no suitable MLX CLAP checkpoint exists, fall back
  to the existing CLIP text encoder plus a mel-spectrogram adapter only if the
  quality holds up; otherwise keep audio search FTS-only until CLAP is ready.
- **Whisper on MLX** — `mlx-whisper` or a compatible local model. Do not block
  `scan --audio` on the transcript model; fingerprint and CLAP are the cheap
  core.
- **Tag reading** — `mutagen` is the smallest focused dependency; ffprobe
  `format_tags` avoids a new dependency but exposes less consistent tag names.

## Safety and performance

- Audio decoding is CPU/IO heavy; never read a whole track into memory. Use
  ffmpeg pipes and a bounded window queue.
- Cap segments per track (default similar to video's 120 keyframes) so a
  10-hour audiobook does not produce tens of thousands of vectors.
- Fingerprints and CLAP vectors are derived data; `reset` and `--prune` must
  clean up `audio_segments` and FTS rows alongside `audio`.
- `clean --audio` never runs by default. It must print the group, the best
  version and reclaimable bytes, and require confirmation or `--yes`.
- Do not add audio to `iter_media()` until `prepare()` and `store_result()` have
  the audio branch. That is the failure mode the current TODO comments guard.
- Audio stages follow the same model/display contract as images and video:
  load the fingerprint/CLAP/transcription model before the stage's progress
  bar, and let the bar's rolling 60-second rate/ETA drive the estimate. A
  music library is often thousands of small files, exactly where a lifetime
  average becomes misleading.

## Implementation tickets

### T1 — Registry and types
- central `AUDIO_EXTENSIONS`; keep `bootstrap.classify_extension()` working
- `AudioRecord` shape for grouping/cleanup/TUI
- tests: extension count, `--media-only` keeps audio

### T2 — Schema v4
- `audio`, `audio_segments`, `audio_fts`; migration and cleanup on delete
- `upsert_audio`, `set_audio_segments`, `all_audio`, `count_audio`
- tests: round-trip, FTS, stale-row pruning, reset

### T3 — Probe and tags
- ffprobe metadata + tag normalization
- `scan --audio` CPU stage with SHA1 and fingerprint
- tests: missing ffprobe/fpcalc, zero-duration file, tag-less file

### T4 — Fingerprints
- `fpcalc -raw` integration, overlap-aware comparison, threshold
- exact SHA1 + fingerprint groupers
- tests: MP3 vs FLAC copy groups; different recordings do not

### T5 — CLAP embeddings
- decode windows, batch on GPU, store segments
- `embed --audio` backfill and `--recompute`
- tests with a fake encoder, no real model download

### T6 — Interpretation
- speech/music branch, Whisper transcript or acoustic description
- CLAP categories, refine facets, tags
- tests: description present, keywords refine into music facets

### T7 — Query and cleanup
- `search --audio`, `tags --audio`, `similar --audio`
- `duplicates --audio`, `clean --audio`, `check --audio`
- tests: no audio in image/video paths; confirmed-only trash; restore

### T8 — CLI, docs, release
- `--audio` on every relevant parser, help groups and examples
- README audio section, `CLI-LOGIC-MAP.md`, CHANGELOG, version bump
- tests: help pages, README flags, all commands grouped

## Non-goals for the first audio release

- Cloud music recognition (AcoustID/Shazam-style lookup) as a requirement.
- Perfect transcription or music genre accuracy.
- Face/person clustering for audio.
- A separate `music` command family.
- Treating a live recording and the studio track as duplicates for `clean`.
