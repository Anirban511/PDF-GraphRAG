"""
reranker.py — Retrieval Stage 2: cross-encoder reranking.

WHY THIS EXISTS:
  Bi-encoder retrieval (Stage 1) is fast but coarse: both query and chunks
  are embedded independently, so fine-grained query–chunk interactions are
  lost.  A cross-encoder processes query + chunk *together*, enabling much
  richer relevance scoring at the cost of higher latency.

  We use a two-stage approach:
    Stage 1 (bi-encoder)   — fast, recalls top-K candidates
    Stage 2 (cross-encoder) — slow, precise, re-ranks a small candidate set

GRACEFUL DEGRADATION:
  If the cross-encoder fails to load (network issue, missing dep), the
  reranker silently falls back to sorting by the original cosine scores.
  The pipeline keeps working; only precision is slightly reduced.

MODEL — cross-encoder/ms-marco-MiniLM-L-6-v2:
  Trained on MS MARCO passage ranking, ~67 MB.  Produces a relevance
  logit per (query, passage) pair; higher is better.
"""

from __future__ import annotations
from app.config import settings
from app.utils.logger import logger

try:
    from sentence_transformers import CrossEncoder as _CE
    _cross_encoder = _CE("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    _HAS_CE = True
    logger.info("Cross-encoder reranker loaded.")
except Exception as exc:
    _HAS_CE = False
    logger.warning(f"Cross-encoder unavailable ({exc}); falling back to cosine scores.")


class Reranker:
    def __init__(self, top_k: int | None = None):
        self.top_k = top_k or settings.rerank_top_k

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-score *candidates* against *query* and return the top-k."""
        if not candidates:
            return []

        if _HAS_CE:
            scores = _cross_encoder.predict([(query, c["text"]) for c in candidates])
            for chunk, score in zip(candidates, scores):
                chunk["rerank_score"] = float(score)
            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        else:
            ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

        top = ranked[:self.top_k]
        logger.debug(f"Reranked {len(candidates)} → kept {len(top)}")
        return top
