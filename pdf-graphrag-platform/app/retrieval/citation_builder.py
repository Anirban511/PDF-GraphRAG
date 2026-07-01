"""
citation_builder.py — Retrieval Stage 3: chunks → LLM context + citation objects.

WHY THIS EXISTS:
  Two separate concerns are handled here:

  build_context():
    Formats the ranked chunks into a single string injected into the LLM
    prompt.  Each chunk is labelled with its source file and page so the
    model can write inline citations without any post-processing.

  extract_citations():
    Produces a clean list of citation objects for the API response and UI.
    Deduplicates by (filename, page_num) because multiple chunks from the
    same page would otherwise produce redundant footnotes.
    Truncates excerpts to 200 chars — enough to give the user context,
    short enough not to clutter the UI.
"""

from __future__ import annotations


def build_context(chunks: list[dict]) -> str:
    """
    Return a formatted string of chunk text blocks for LLM injection.
    Each block is headed by its source reference.
    """
    return "\n\n---\n\n".join(
        f"[Chunk {i} | Source: {c['filename']}, Page {c['page_num']}]\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def extract_citations(chunks: list[dict]) -> list[dict]:
    """
    Return deduplicated citation objects for the API response.
    Order mirrors the reranker ranking (most relevant first).
    """
    seen, citations = set(), []
    for c in chunks:
        key = (c["filename"], c["page_num"])
        if key not in seen:
            seen.add(key)
            excerpt = c["text"]
            citations.append({
                "filename": c["filename"],
                "page_num": c["page_num"],
                "doc_id":   c["doc_id"],
                "excerpt":  excerpt[:200] + ("…" if len(excerpt) > 200 else ""),
            })
    return citations
