"""
chunker.py — Ingestion Stage 2: PageRecord list → Chunk list.

WHY THIS EXISTS:
  LLMs and embedding models have fixed context windows.  Splitting text into
  smaller, overlapping pieces lets us (a) stay within token limits and
  (b) avoid cutting a relevant sentence in half at a chunk boundary.

ALGORITHM — sliding window with sentence-boundary snapping:
  1. Walk through the page text in steps of (chunk_size - overlap).
  2. At each step, look for the last ". " in the current window.
     If one is found past the midpoint, end the chunk there instead of at a
     hard character boundary — this keeps sentences intact.
  3. Advance the start pointer to (previous_end - overlap), so consecutive
     chunks share *overlap* characters of context.

WHY CHARACTER-BASED (not token-based):
  Token counts vary by model and add a dependency.  Character counts are
  deterministic and fast; ~512 chars ≈ 100–130 tokens for English text,
  which fits comfortably inside all tested embedding models.
"""

from __future__ import annotations
from dataclasses import dataclass

from app.config import settings
from app.ingestion.loader import PageRecord
from app.utils.logger import logger


@dataclass
class Chunk:
    """A text segment with full provenance for retrieval and citation."""
    chunk_id:    str    # "<doc_id[:8]>_p<page>_c<idx>" — unique, human-readable
    doc_id:      str
    filename:    str
    page_num:    int
    chunk_index: int    # position within the page (0-indexed)
    text:        str
    metadata:    dict


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Return a list of overlapping text segments from *text*."""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        segment = text[start:end]

        # Snap to sentence boundary if one exists past the midpoint
        if end < len(text):
            bp = segment.rfind(". ")
            if bp > chunk_size // 2:
                end = start + bp + 1
                segment = text[start:end]

        if segment.strip():
            chunks.append(segment.strip())
        start = end - overlap

    return chunks


def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Convert PageRecords → Chunks using settings defaults (overridable for tests)."""
    chunk_size = chunk_size or settings.chunk_size
    overlap    = overlap    or settings.chunk_overlap

    chunks = [
        Chunk(
            chunk_id    = f"{p.doc_id[:8]}_p{p.page_num}_c{idx}",
            doc_id      = p.doc_id,
            filename    = p.filename,
            page_num    = p.page_num,
            chunk_index = idx,
            text        = seg,
            metadata    = p.metadata,
        )
        for p in pages
        for idx, seg in enumerate(_split_text(p.text, chunk_size, overlap))
    ]

    logger.info(
        f"Chunking: {len(pages)} pages → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap})"
    )
    return chunks
