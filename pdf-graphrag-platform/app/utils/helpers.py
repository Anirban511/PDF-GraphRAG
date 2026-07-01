"""
helpers.py — Shared micro-utilities.

WHY THIS EXISTS:
  Keeps one-liner utilities (hashing, JSON I/O, timing) out of business
  logic so every module stays focused on its own concern.

  file_hash()     — SHA-256 of a PDF byte-for-byte; used as a stable,
                    collision-resistant document ID and deduplication key.
  sanitize_filename() — makes doc IDs safe to use as part of filenames.
  save_json/load_json — thin wrappers that ensure consistent UTF-8 encoding
                    and auto-create parent directories.
  Timer           — context manager; avoids littering code with time.perf_counter
                    calls and keeps latency measurement declarative.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    """SHA-256 hex digest — reads in 8 KB blocks to stay memory-efficient on large PDFs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def sanitize_filename(name: str) -> str:
    """Replace any character that is unsafe in filenames with an underscore."""
    return re.sub(r"[^\w\-.]", "_", name)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """
    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed)   # seconds as float
        print(t)           # "1.234s"
    """
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self):
        return f"{self.elapsed:.3f}s"
