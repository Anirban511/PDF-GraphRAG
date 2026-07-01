"""
loader.py — Ingestion Stage 1: PDF → PageRecord list.

WHY THIS EXISTS:
  Separates I/O (reading bytes off disk, calling PDF libraries) from all
  downstream logic.  Every later stage receives plain Python dataclasses,
  never raw file handles or library objects.

DESIGN NOTES:
  • doc_id is a SHA-256 hash of the file bytes, not the filename.  This means
    renaming a file does not re-index it, and two identical files (different
    names) share one ID — natural deduplication.
  • Processed JSON is written to data/processed/ so the extracted text can
    be inspected or reloaded without re-parsing the PDF.
  • load_all_pdfs() is a convenience wrapper for the CLI / batch scripts;
    the API uses load_pdf() directly per upload.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.utils.helpers import file_hash, sanitize_filename, save_json
from app.utils.logger import logger
from app.utils.pdf_utils import get_metadata, iter_pages


@dataclass
class PageRecord:
    """One page of text with its full provenance."""
    doc_id:   str           # SHA-256 of source PDF (stable, collision-resistant)
    filename: str           # original filename, kept for citation display
    page_num: int           # 1-indexed
    text:     str
    metadata: dict = field(default_factory=dict)


def load_pdf(pdf_path: Path) -> list[PageRecord]:
    """
    Extract every page of *pdf_path* and return a list of PageRecords.
    Also writes a JSON snapshot to data/processed/ for auditability.

    Raises FileNotFoundError if the path does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = file_hash(pdf_path)
    meta   = get_metadata(pdf_path)
    logger.info(f"Loading '{pdf_path.name}' — {meta['page_count']} pages (id={doc_id[:8]}…)")

    records = [
        PageRecord(doc_id=doc_id, filename=pdf_path.name,
                   page_num=pn, text=txt, metadata=meta)
        for pn, txt in iter_pages(pdf_path)
    ]

    out_path = settings.processed_dir / f"{sanitize_filename(pdf_path.stem)}_{doc_id[:8]}.json"
    save_json(
        {"doc_id": doc_id, "filename": pdf_path.name, "metadata": meta,
         "pages": [{"page_num": r.page_num, "text": r.text} for r in records]},
        out_path,
    )
    logger.success(f"Extracted {len(records)} pages → {out_path.name}")
    return records


def load_all_pdfs(directory: Path | None = None) -> list[PageRecord]:
    """Load every *.pdf in *directory* (default: settings.raw_pdfs_dir)."""
    directory = Path(directory or settings.raw_pdfs_dir)
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {directory}")
        return []

    records: list[PageRecord] = []
    for pdf in pdfs:
        try:
            records.extend(load_pdf(pdf))
        except Exception as exc:
            logger.error(f"Failed to load {pdf.name}: {exc}")
    return records
