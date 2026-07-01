"""
Tests for the chunker module.
"""

import pytest
from app.ingestion.chunker import _split_text, chunk_pages, Chunk
from app.ingestion.loader import PageRecord


def test_split_short_text():
    """Text shorter than chunk_size returns a single chunk."""
    text = "Hello world."
    result = _split_text(text, chunk_size=512, overlap=64)
    assert result == ["Hello world."]


def test_split_long_text():
    """Long text is split into multiple overlapping chunks."""
    text = "sentence one. " * 100
    chunks = _split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # Every chunk should be non-empty
    assert all(c.strip() for c in chunks)


def test_split_empty_text():
    """Empty input returns empty list."""
    assert _split_text("", chunk_size=512, overlap=64) == []


def test_chunk_pages_produces_chunk_objects():
    pages = [
        PageRecord(
            doc_id="abc123",
            filename="test.pdf",
            page_num=1,
            text="This is a test. " * 50,
            metadata={},
        )
    ]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.filename == "test.pdf" for c in chunks)
    assert all(c.page_num == 1 for c in chunks)


def test_chunk_id_uniqueness():
    pages = [
        PageRecord(
            doc_id="abc123",
            filename="test.pdf",
            page_num=1,
            text="word " * 200,
            metadata={},
        )
    ]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk IDs must be unique"
