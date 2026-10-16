"""User configuration and per-tree ignore rules.

Two knobs, both optional:

- ``~/.config/media-match/config.toml`` sets defaults for the flags that are
  usually repeated (``clip_model``, ``model``, ``workers``, ``batch``,
  ``interval``, ``max_frames``, ``threshold``, ``prefer``). An explicit
  command-line flag always wins; a config value wins over the built-in default.

  ```toml
  [media-match]
  clip_model = "mlx-community/clip-vit-base-patch32"
  workers = 8
  threshold = 8
  ```

- ``<root>/.media-matchignore`` lists glob patterns (one per line, ``#``
  comments) for paths to leave out of the index, e.g.::

      # caches and exports
      *_export/
      node_modules/
      *.tmp

Matching is against the path relative to the scanned root and its basename,
so both ``node_modules/`` and ``**/cache`` style entries work.
"""

from __future__ import annotations

import fnmatch
import os

CONFIG_PATH = os.path.expanduser("~/.config/media-match/config.toml")
IGNORE_FILE = ".media-matchignore"

# config key -> the flag it can default. None means "no config setting".
CONFIG_KEYS = (
    "clip_model", "model", "workers", "batch", "interval", "max_frames",
    "threshold", "prefer",
)


def load_config() -> dict:
    """Read ~/.config/media-match/config.toml; {} if absent or unparsable."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        return {}
    try:
        with open(CONFIG_PATH, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, ValueError):
        return {}
    section = data.get("media-match", data)
    return {k: section[k] for k in CONFIG_KEYS if k in section}


def load_ignore(root: str) -> list[str]:
    """Non-empty, non-comment lines of <root>/.media-matchignore."""
    path = os.path.join(root, IGNORE_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            return [
                line.strip()
                for line in fh
                if line.strip() and not line.strip().startswith("#")
            ]
    except OSError:
        return []


def is_ignored(rel_path: str, patterns: list[str]) -> bool:
    """True if a root-relative path matches any ignore glob.

    Handles directory entries (``node_modules/``), bare names at any depth
    (``cache``), and globs (``*.tmp``); the basename is matched too.
    """
    base = os.path.basename(rel_path)
    segments = rel_path.split("/")
    for pat in patterns:
        stem = pat.rstrip("/")
        if not stem:
            continue
        if rel_path.startswith(pat) or base == stem or stem in segments:
            return True
        if fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(base, pat):
            return True
    return False
