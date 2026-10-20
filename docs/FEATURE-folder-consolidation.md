# FEATURE: folder-level consolidation (`folders`)

- **Status:** implementing, first shipped in **0.3.0**
- **Date:** 2026-10-20
- **Command:** `media-match folders`
- **Module:** `media_match/folders.py`
- **Tests:** `tests/test_folders.py`

This is the feature request that starts the 0.3 era. The raw request is kept
below verbatim; the rest of the document turns it into a specification and a
ticket list, and records every decision made where the request left a choice
open.

## Raw request (verbatim)

> Add a command to media-match that basically does this: scans the target folder recursively for a foldername, match regardles of upper cases or lower case, basically a find . -type d -iname \*$NAME$\*' and do that for every folder in the target folder, the arguments/options should be --collapse automatically creates a folder in the root called , --list
>
> Add a command to media-match that basically does this: scans the target folder recursively for a foldername, match regardles of upper cases or lower case, basically a find . -type d -iname \*$NAME$\*' and do that for every folder in the target folder, the arguments/options should be --collapse automatically creates a folder in the root called $NAME_collapsed/ and moves all the other folders there as $NAME_1,2,3,4,5 etc; --list does what it says on the packaging, simply lists all groups of folders found with the same name, --replace replace all files in the root / destination folder with all other folders content if the filenames are the same, i.e `mv -rf`, --rename if a file exists in both the source and target, rename the sourcefile with a number, --yes answers yes to all questions, and the default behavior should be to show the user a similar interface as with duplicates; group all folders with the same name from the root folder, then the user should be allowed to pick which folder should be the destination (shortest path suggested), and then the behavior for files; replace, rename, skip (replace, rename, skip should have an 'all' feature, so r for replace R for replace all, n for rename, N for rename all, s for skip, S for skip all), compare: show exif-data for each file and let the user select replace,rename,skip, at any point if the user press q or escape, ask the user if they want to execute up until their marked folder, excluding current (unless it is finished). Very much similar features as when dealing with duplicate files, but working on a folder level. Take this feature, make into a proper feature request/breakdown with ticket-items that adds each function and feature to be able to deliver the whole feature in one go.

## Summary

A folder-level counterpart to `duplicates`/`clean`. Media tools routinely end
up with the same folder name in several places — `2019/christmas`,
`2020/Christmas`, `2021/CHRISTMAS` — because exports, phone backups and manual
sorting all invent their own copy. `folders` finds those groups, lets you pick
one copy as the destination, and merges the others into it, resolving
same-name file conflicts one at a time exactly the way `clean` reviews
duplicate files.

It is a filesystem command: it reads no index and changes no index. It works on
any folder tree, media or not.

## Command surface

```
media-match folders [root] [--name NAME] [--list]
                    [--replace | --rename] [--yes] [--dry-run] [--json]
media-match folders [root] --collapse [--dry-run] [--json]
```

| Flag | Meaning |
|---|---|
| `root` | tree to scan (default `.`), scanned recursively |
| `--name NAME` | only consider folder names containing NAME, case-insensitively (the `find -iname '*NAME*'` filter) |
| `--list` | print every group found and exit; changes nothing |
| `--collapse` | do not merge files; create `<root>/<Name>_collapsed/` and move every non-destination member into it as `<Name>_1`, `<Name>_2`, … |
| `--replace` | at a filename conflict, the incoming (source) file overwrites the destination file (`mv -f`) |
| `--rename` | at a filename conflict, the incoming file is kept under a numbered name (`photo_1.jpg`) |
| `--yes` | do not ask: use the suggested destination, and apply the conflict policy to every conflict |
| `--dry-run` | compute and print the plan, change nothing, ask nothing |
| `--json` | machine-readable form of `--list`, `--collapse` or `--dry-run` |

`--replace` and `--rename` are mutually exclusive. `--collapse` cannot be
combined with a file-conflict policy because it never touches files.

## Behaviour specification

### 1. Discovery

1. Walk `root` recursively without following directory symlinks (the
   equivalent of `find . -type d`, not `find -L`).
2. Skip hidden directories (`.git`, `.Trash`, …) and anything matched by
   `<root>/.media-matchignore` — the same ignore file the scanner uses.
3. Group the remaining directories by `basename.casefold()`. Matching is
   case-insensitive and does not normalise accents or Unicode compatibility
   forms: `Christmas`, `christmas` and `CHRISTMAS` are one group;
   `Noël` and `Noel` are not.
4. With `--name`, first filter directories whose casefolded name contains the
   casefolded NAME, then group the survivors by exact casefolded name. This is
   deliberate: `--name christmas` must not merge `Christmas 2020` into
   `christmas`, because those are different folder names. The filter narrows
   what is considered; the group is still "same name".
5. Keep only groups with two or more members. A single folder is not a group.
6. Groups and members are ordered deterministically (name, then path) so runs
   are reproducible and `--list` diffs cleanly.

### 2. Destination

Each group has one destination and one or more sources.

- The **suggested destination** is the member with the shortest root-relative
  path (fewest path components, then shortest string, then lexicographic). It
  is the copy closest to the root, which is the least surprising place to keep.
- In interactive mode the suggestion is preselected; `1..N` picks another
  member.
- With `--yes` the suggestion is used without a prompt.
- The destination folder itself never moves. Sources are merged into it.
- A source that is an ancestor or descendant of the destination is refused
  (with a printed reason): moving a tree into itself is never intended.
  Example: `a/Christmas` and `a/Christmas/Christmas`.

### 3. Merge (default mode)

For each group, then each source folder (sorted), merge recursively:

1. Every file under the source is moved to the same root-relative path under
   the destination. A file `source/sub/photo.jpg` lands at
   `destination/sub/photo.jpg`; the source's subdirectory layout is preserved.
2. A file with no counterpart is moved directly.
3. A file whose destination path already exists is a **conflict** and gets one
   decision:

   | Key | Decision |
   |---|---|
   | `r` | replace: the source file overwrites the destination file |
   | `R` | replace, for every remaining conflict in this run |
   | `n` | rename: the destination keeps its name; the source file is moved as `photo_1.jpg` |
   | `N` | rename, for every remaining conflict in this run |
   | `s` | skip: the source file stays where it is |
   | `S` | skip, for every remaining conflict in this run |
   | `c` | compare: print size, timestamps and EXIF for both files, then ask again |
   | `q` / `Esc` | stop and offer to keep the completed work (see §5) |

4. Numbering starts at 1 and skips names already present, so `photo.jpg`,
   `photo_1.jpg`, `photo_2.jpg` can all coexist.
5. A path where the destination has a **directory** and the source has a file
   (or the reverse) is reported and skipped; it is a tree-shape disagreement,
   not a filename conflict, and is never resolved by deleting either side.
6. After a source folder has been fully reviewed, its empty directories are
   removed bottom-up. Directories that still contain skipped files remain, so
   nothing is ever silently lost.
7. Moves use `os.rename` where possible and fall back to `shutil.move` across
   volumes.
8. Every move is printed as it is applied.

### 4. Compare

`c` at a conflict prints, for the destination file and the source file:

- absolute and root-relative path
- size and modification time
- image EXIF where readable: capture time, camera, GPS
- for videos, duration/resolution if `ffprobe` is available; otherwise the
  same file facts

It then returns to the same conflict prompt. Compare is read-only and never
changes a decision by itself.

### 5. Quit and partial execution

All decisions are buffered; nothing moves until the user confirms.

- `q` or `Esc` stops the review.
- If at least one source folder was fully decided, the prompt is:
  *"Execute everything decided so far, excluding the current folder? [y/N]"*
- The **current** source folder is excluded because its review is incomplete.
  If `q` arrives when the current folder is already finished (for example at
  the next group's destination prompt), there is no unfinished folder to
  exclude and the wording drops that clause.
- Answering yes applies the completed plans; no applies nothing.
- At the natural end of the review, the same confirmation is asked once:
  *"Apply N move(s) from M folder(s)? [y/N]"*.
- `--yes` skips the confirmation and applies everything.

This mirrors `clean`: review first, execute once, and a failed or abandoned
review leaves the filesystem exactly as it was.

### 6. Non-interactive combinations

| Invocation | Destination | Conflicts | Confirmation |
|---|---|---|---|
| `folders` | prompt (suggested) | prompt per conflict | prompt |
| `folders --replace` | prompt (suggested) | replace all | prompt |
| `folders --rename` | prompt (suggested) | rename all | prompt |
| `folders --yes` | suggested | skip all | none |
| `folders --replace --yes` | suggested | replace all | none |
| `folders --rename --yes` | suggested | rename all | none |
| `folders --dry-run` | suggested | shown as "would ask" | none |
| `folders --replace --dry-run` | suggested | shown as replace | none |

`--yes` without a policy deliberately skips conflicts rather than guessing:
the safe interpretation of "yes to everything" is "do everything that is
unambiguous". `--replace --yes` is the explicit `mv -rf` form.

A non-interactive stdin that reaches EOF is treated as `q`; the confirmation
question then defaults to *no*, so a broken pipe cannot move files.

### 7. Collapse mode

`--collapse` is the non-destructive sibling of merge: it gathers whole
folders instead of combining their contents.

1. Pick the suggested destination (or the only sensible one) and keep it where
   it is.
2. Create `<root>/<Name>_collapsed/`, where `<Name>` is the destination
   folder's own spelling.
3. Move every other member into it as `<Name>_1`, `<Name>_2`, … skipping names
   already present, so a second run continues the numbering instead of
   overwriting.
4. The original parent directories of the moved folders are removed if they
   become empty; the moved folder keeps everything inside it.
5. Nothing is merged and no file is renamed individually.

### 8. List mode

`--list` prints every group with its members, file count, total size and the
suggested destination, then exits without a prompt. With `--json` the same
data is emitted as a list of objects. `--collapse --json` and the dry-run
report use the same machine-readable style: the planned (or performed) folder
moves, their sources and any failures.

## Safety rules

- Never delete a file. `--replace` overwrites a destination file; that is the
  documented cost of `mv -rf`, and `--rename` is the non-destructive answer.
- Never merge a folder into itself or into a folder nested inside it.
- Never follow directory symlinks while discovering or moving.
- Refuse to act when two members of a group would resolve to the same
  destination path.
- Skip and report anything that raises `OSError`; finish the rest and return a
  non-zero exit code if any move failed.
- Buffered execution: an interrupted review changes nothing until the user
  says yes.
- Moves make any existing index stale. The command says so at the end and
  points at `media-match scan --prune`.

## Output shape

Interactive group header, in the same spirit as `duplicates`:

```
Group 1/2 — christmas (3 folders)
  1. [dest*] 2019/christmas        (12 files, 48.2 MB)
  2.         2020/Christmas         (8 files, 31.0 MB)
  3.         2021/CHRISTMAS        (15 files, 60.1 MB)
  * suggested destination (shortest path)

destination? [Enter]=2019/christmas, 1-3, q=quit >
```

Conflict:

```
  conflict  sub/photo.jpg
    destination  2019/christmas/sub/photo.jpg  (2.4 MB, 2020-12-24 18:31)
    source       2021/CHRISTMAS/sub/photo.jpg  (3.1 MB, 2021-01-02 09:12)
  [r]eplace [R]eplace all [n]ame [N]ame all [s]kip [S]kip all [c]ompare [q]uit >
```

Final summary:

```
2 group(s): 3 folder(s) merged, 1 kept, 27 files moved (12 replaced, 4 renamed),
2 skipped, 0 failed.
```

## Ticket breakdown

Every ticket is independently testable and lands together in one release.

### T1 — Discovery and grouping

- `find_folder_groups(root, name=None, ignore=())`
- recursive walk, no symlink following, hidden and ignored directories pruned
- case-insensitive grouping by basename; `--name` substring filter
- deterministic ordering; groups of one dropped
- unit tests: case variants, nested duplicates, hidden/ignored, `--name`,
  symlink not followed, single folder excluded

### T2 — Destination suggestion and group rendering

- `suggest_destination(group, root)` — shortest path, deterministic tie-break
- `folder_stats(path)` — file count and byte total
- `render_group(...)` in the duplicate-review style
- unit tests: deepest vs shallowest, ties, stable ordering, stats

### T3 — Plan and apply primitives

- `plan_folder(source, destination, policy=None, resolve=None)`
- `apply_moves(moves)` — move / replace / rename / skip, cross-volume fallback
- `numbered_path(directory, name)` — `photo_1.jpg`, `folder_1`
- `prune_empty_dirs(source)`
- conflict classification: file/file, file/dir, dir/file
- unit tests: every action, numbering, nested layout, empty-dir cleanup,
  skipped files keeping their folder, failures reported not raised

### T4 — Interactive review

- destination prompt with suggested default and `q`
- per-conflict prompt with `r/R/n/N/s/S`
- shared "all" state across the run
- buffered decisions; final confirmation
- unit tests driven by a scripted input function

### T5 — Compare

- `compare_files(source, destination, out)` — path, size, mtime, image EXIF,
  video probe when available
- `c` returns to the same prompt without changing the decision
- unit tests: EXIF present, EXIF absent, non-image file

### T6 — Quit and partial execution

- `QuitReview` raised from the prompt layer
- completed source folders retained, current one discarded
- prompt wording depends on whether a folder is open
- yes applies completed plans, no applies nothing
- unit tests: quit mid-folder, quit at destination prompt, EOF as quit,
  confirmation refusal leaving the tree untouched

### T7 — Non-interactive flags

- `--replace`, `--rename`, `--yes`, `--dry-run`
- mutually exclusive policies; `--yes` without a policy skips conflicts
- dry-run never prompts and never writes
- unit tests: every combination in §6

### T8 — Collapse

- `<root>/<Name>_collapsed/` creation, numbered moves, continuation of an
  existing numbering, destination left in place
- `--dry-run` and `--json` reporting
- unit tests: two groups, existing collapsed folder, name-case preservation

### T9 — CLI, help and documentation

- `folders` parser with complete help for every flag
- entry in `_COMMAND_SUMMARIES`, `_EXAMPLES` and exactly one `_COMMAND_GROUPS`
  heading ("quality and file management")
- README commands row and a dedicated folder-consolidation section
- `--list` and `--json` output contracts
- tests: the existing help invariants cover the new command automatically;
  plus a CLI end-to-end test for `--list`, `--collapse`, `--replace --yes`,
  `--rename --yes`, and the dry-run default

### T10 — Release housekeeping

- version **0.3.0** in `pyproject.toml` and `_VERSION`
- new `0.3.x — folder consolidation` section in `CHANGELOG.md`
- this feature document kept as the design record
- full suite green, including the undefined-name lint test

## Acceptance criteria

1. `folders --list` on a tree with `A/x`, `B/X`, `C/x` reports one group with
   three members and suggests the shallowest.
2. `folders --replace --yes` merges all sources into the suggested
   destination, overwrites same-path files, and leaves no source folder that
   became empty.
3. `folders --rename --yes` keeps every file: conflicts land as
   `name_1.ext`, `name_2.ext`, …
4. `folders --yes` never overwrites or renames; conflicting source files stay
   in place and are reported as skipped.
5. Interactive `r`, `n`, `s`, `c`, `q` do what §3 says; `R`/`N`/`S` apply for
   the rest of the run.
6. `q` asks before applying completed folders and excludes the open one;
   refusing leaves the tree byte-for-byte unchanged.
7. `folders --collapse` creates `<Name>_collapsed` and moves every
   non-destination member into it as `<Name>_1..n`, without touching files.
8. `--dry-run` prints the same plan and writes nothing.
9. No path is ever merged into itself, no directory symlink is followed, and
   a failed move is reported with a non-zero exit code.
10. `media-match help folders`, `folders --help` and `help folders` are
    identical, and every flag has help text.

## Non-goals

- No use of the media index, CLIP, or EXIF extraction beyond what compare
  prints. Folder consolidation is a filesystem tool.
- No undo journal. `--rename` is the reversible path; `--replace` is
  explicitly destructive, like `mv -f`.
- No cross-root moves (destination selection is always within the group).
- No adaptation of the full-screen `clean` bar UI. The review loop is
  line-based, matching `clean`'s non-TTY interface and the requested
  `r/R/n/N/s/S` keys.
- No automatic merging of groups whose names merely start the same
  (`Christmas` vs `Christmas 2020`); only identical casefolded names group.
