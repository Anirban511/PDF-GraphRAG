"""
Tests for the citation builder.
"""

from app.retrieval.citation_builder import build_context, extract_citations


SAMPLE_CHUNKS = [
    {
        "chunk_id": "abc_p1_c0",
        "doc_id": "abc123",
        "filename": "paper.pdf",
        "page_num": 1,
        "text": "This is the first chunk of text.",
        "score": 0.92,
    },
    {
        "chunk_id": "abc_p3_c1",
        "doc_id": "abc123",
        "filename": "paper.pdf",
        "page_num": 3,
        "text": "This is a chunk from page three.",
        "score": 0.85,
    },
    {
        "chunk_id": "def_p2_c0",
        "doc_id": "def456",
        "filename": "report.pdf",
        "page_num": 2,
        "text": "Content from a different document.",
        "score": 0.80,
    },
]


def test_build_context_contains_filenames():
    ctx = build_context(SAMPLE_CHUNKS)
    assert "paper.pdf" in ctx
    assert "report.pdf" in ctx


def test_build_context_contains_page_numbers():
    ctx = build_context(SAMPLE_CHUNKS)
    assert "Page 1" in ctx
    assert "Page 3" in ctx


def test_build_context_contains_text():
    ctx = build_context(SAMPLE_CHUNKS)
    assert "first chunk" in ctx
    assert "page three" in ctx


def test_extract_citations_deduplication():
    # Add a duplicate of the first chunk
    chunks = SAMPLE_CHUNKS + [SAMPLE_CHUNKS[0]]
    citations = extract_citations(chunks)
    # Should deduplicate by (filename, page_num)
    keys = [(c["filename"], c["page_num"]) for c in citations]
    assert len(keys) == len(set(keys))


def test_extract_citations_fields():
    citations = extract_citations(SAMPLE_CHUNKS)
    for c in citations:
        assert "filename" in c
        assert "page_num" in c
        assert "doc_id" in c
        assert "excerpt" in c
