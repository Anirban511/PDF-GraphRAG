"""
pdf_utils.py — Low-level PDF I/O helpers.

WHY THIS EXISTS:
  pdfplumber extracts text with better layout awareness (handles columns,
  tables, spacing) than pypdf alone. pypdf is kept as a lightweight fallback
  for malformed files that pdfplumber cannot open.  Both libraries are
  already in requirements.txt, so there is no extra cost.

  get_metadata() reads XMP/Info metadata in a single PdfReader pass that
  also counts pages — avoiding a second file open in loader.py.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator

import pdfplumber
from pypdf import PdfReader

from app.utils.logger import logger


def iter_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    """Yield (1-indexed page_num, stripped text) for every page."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                yield i, (page.extract_text() or "").strip()
    except Exception as exc:
        logger.warning(f"pdfplumber failed on {pdf_path.name}: {exc} — using pypdf")
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            yield i, (page.extract_text() or "").strip()


def get_metadata(pdf_path: Path) -> dict:
    """
    Return a dict with title, author, subject, creator, page_count.
    Opens the file once with pypdf (lighter than pdfplumber for metadata only).
    """
    reader = PdfReader(str(pdf_path))
    meta = reader.metadata or {}
    return {
        "title":      meta.get("/Title",   ""),
        "author":     meta.get("/Author",  ""),
        "subject":    meta.get("/Subject", ""),
        "creator":    meta.get("/Creator", ""),
        "page_count": len(reader.pages),
    }
