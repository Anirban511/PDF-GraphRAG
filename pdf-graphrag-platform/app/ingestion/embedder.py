"""
embedder.py — Ingestion Stage 3: text → float32 vectors.

WHY THIS EXISTS:
  Dense embeddings map text into a high-dimensional space where semantic
  similarity ≈ geometric proximity.  This is what makes keyword-free search
  possible (e.g. "heart attack" matches "myocardial infarction").

MODEL CHOICE — all-MiniLM-L6-v2:
  384-dim, ~22 MB, MIT licence.  Runs on CPU in <1 s per batch.
  Produces L2-normalised vectors so inner-product == cosine similarity,
  which is what our FAISS index expects.

SINGLETON PATTERN:
  SentenceTransformer loads ~100 MB of weights into memory.  Keeping one
  module-level instance avoids reloading on every request.  _get_model() is
  thread-safe for read access (Python GIL protects the lazy-init assignment).

embed_chunks() vs embed_query():
  Chunks are embedded in batches (fast, GPU-friendly if available).
  Queries are single strings; batching would add unnecessary latency.
  Both use normalize_embeddings=True so scores are comparable.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingestion.chunker import Chunk
from app.utils.logger import logger

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.success("Embedding model ready.")
    return _model


def embed_chunks(chunks: list[Chunk], batch_size: int = 64) -> np.ndarray:
    """
    Encode *chunks* in batches.
    Returns float32 array of shape (N, embedding_dim).
    """
    model  = _get_model()
    texts  = [c.text for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks…")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.success(f"Embeddings done — shape {vecs.shape}")
    return vecs.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string.
    Returns float32 array of shape (1, embedding_dim).
    """
    vecs = _get_model().encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs.astype(np.float32)
