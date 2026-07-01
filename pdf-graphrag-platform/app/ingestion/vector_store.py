"""
vector_store.py — Ingestion Stage 4: FAISS index wrapper.

WHY THIS EXISTS:
  FAISS (Facebook AI Similarity Search) runs approximate/exact nearest-
  neighbour search over millions of vectors in milliseconds on CPU.
  This wrapper encapsulates the index + aligned metadata so callers never
  touch raw FAISS arrays.

INDEX CHOICE — IndexFlatIP:
  "Flat" = exact (no approximation).  "IP" = inner product.
  Because our vectors are L2-normalised, inner product equals cosine
  similarity.  Flat search is accurate and fast enough for up to ~500 k
  chunks; swap for IndexIVFFlat for larger corpora.

PERSISTENCE STRATEGY:
  Two files kept in sync on every write:
    index.faiss    — binary FAISS index (only the vectors)
    metadata.json  — list of chunk dicts, positionally aligned with the index
  Separation lets us inspect/debug chunk text without loading FAISS.

build() vs add():
  build() creates a fresh index (first ingest).
  add()   appends to an existing index (subsequent uploads).
  Both call _save() so disk and in-memory state are always consistent.
"""

from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from app.config import settings
from app.ingestion.chunker import Chunk
from app.utils.logger import logger

_INDEX_FILE = "index.faiss"
_META_FILE  = "metadata.json"


class VectorStore:
    def __init__(self, db_dir: Path | None = None):
        self.db_dir = Path(db_dir or settings.vector_db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._index:  faiss.IndexFlatIP | None = None
        self._chunks: list[dict]               = []

    # ── Build / update ────────────────────────────────────────────────

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Create a brand-new index from *chunks* and their *vectors*."""
        assert len(chunks) == vectors.shape[0], "chunks/vectors length mismatch"
        self._index  = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._chunks = [asdict(c) for c in chunks]
        self._save()
        logger.success(f"FAISS index built: {self._index.ntotal} vectors, dim={self._index.d}")

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Append new chunks to an existing index (loads from disk if needed)."""
        if self._index is None:
            self.load()
        assert vectors.shape[1] == self._index.d, "embedding dimension mismatch"
        self._index.add(vectors)
        self._chunks.extend(asdict(c) for c in chunks)
        self._save()
        logger.info(f"Index updated: {self._index.ntotal} total vectors")

    # ── Search ────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, top_k: int | None = None) -> list[dict]:
        """
        Return up to *top_k* chunk dicts ranked by cosine similarity,
        each augmented with a 'score' key.
        """
        if self._index is None:
            self.load()
        top_k = top_k or settings.top_k
        scores, indices = self._index.search(query_vec, top_k)
        return [
            {**self._chunks[i], "score": float(s)}
            for s, i in zip(scores[0], indices[0])
            if i != -1
        ]

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        faiss.write_index(self._index, str(self.db_dir / _INDEX_FILE))
        with open(self.db_dir / _META_FILE, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        idx_path, meta_path = self.db_dir / _INDEX_FILE, self.db_dir / _META_FILE
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Vector store not found. Ingest PDFs first.")
        self._index = faiss.read_index(str(idx_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        logger.info(f"Index loaded: {self._index.ntotal} vectors, dim={self._index.d}")

    # ── Helpers ───────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return (self.db_dir / _INDEX_FILE).exists()

    def stats(self) -> dict:
        if self._index is None and self.is_ready:
            self.load()
        return {
            "total_chunks": self._index.ntotal if self._index else 0,
            "dimension":    self._index.d      if self._index else 0,
            "unique_docs":  len({c["doc_id"] for c in self._chunks}),
        }
