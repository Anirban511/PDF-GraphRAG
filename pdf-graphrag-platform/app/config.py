"""
config.py — Central configuration loaded from .env

No external API keys required. Uses Ollama for local LLM inference
and Neo4j for the knowledge-graph layer.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "PDF GraphRAG Platform"
    app_version: str = "3.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Paths
    raw_pdfs_dir: Path = Path("data/raw_pdfs")
    processed_dir: Path = Path("data/processed")
    vector_db_dir: Path = Path("data/vector_db")
    reports_dir: Path = Path("data/reports")

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 5
    rerank_top_k: int = 3

    # GraphRAG retrieval
    graph_hops: int = 2            # how far to traverse from a seed entity
    graph_max_chunks: int = 5      # cap on chunks pulled via graph expansion

    # Analytics / extraction caps (LLM-per-chunk is slow on CPU)
    max_extraction_chunks: int = 0  # 0 = process all; set e.g. 150 to cap cost

    # Generation — LLM provider
    # provider: "ollama" (local, free, default) | "openai" | "anthropic"
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    max_tokens: int = 2048
    temperature: float = 0.2

    # Hosted API keys (only needed if llm_provider is openai/anthropic)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # Redis (caching, rate limiting, job queue)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: int = 86400          # 24h cache for ingested docs/queries

    # Rate limiting
    rate_limit_requests: int = 30           # requests per window per client
    rate_limit_window_seconds: int = 60

    # Auth (comma-separated keys; empty = open API for local dev)
    api_keys: str = ""

    # Async ingestion
    async_ingestion: bool = True            # enqueue ingest jobs vs. run inline

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Frontend
    streamlit_port: int = 8501
    api_base_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        for d in (self.raw_pdfs_dir, self.processed_dir,
                  self.vector_db_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
