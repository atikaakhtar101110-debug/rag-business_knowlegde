"""Runtime configuration, read from environment variables.

Only ANTHROPIC_API_KEY is needed for answers to be generated. Everything else
has a working default. The service still starts without the key; /ask then
returns a clear 503 instead of crashing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass(frozen=True)
class Settings:
    # --- secrets (names of the environment variables) ---
    anthropic_api_key: str = ""   # ANTHROPIC_API_KEY (required for answers)
    rag_api_key: str = ""         # RAG_API_KEY (optional: protects the API)
    admin_key: str = ""           # RAG_ADMIN_KEY (optional: enables /ingest/sync)

    # --- models ---
    model: str = "claude-sonnet-5-5"
    rewrite_model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1024

    # --- storage ---
    data_dir: Path = field(default_factory=lambda: Path("data"))
    docs_dir: Path = field(default_factory=lambda: Path("docs"))

    # --- ingestion ---
    auto_ingest: bool = True
    chunk_target_tokens: int = 450
    chunk_overlap_tokens: int = 50

    # --- embeddings and reranking ---
    embedding_backend: str = "fastembed"  # "fastembed" or "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    allow_embed_fallback: bool = True
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_min_score: float = -4.0

    # --- retrieval ---
    candidates_k: int = 30
    final_k: int = 6
    history_turns: int = 6

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.sqlite"

    @property
    def audit_path(self) -> Path:
        return self.data_dir / "audit.jsonl"

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        s = cls(
            anthropic_api_key=_str("ANTHROPIC_API_KEY", ""),
            rag_api_key=_str("RAG_API_KEY", ""),
            admin_key=_str("RAG_ADMIN_KEY", ""),
            model=_str("CLAUDE_MODEL", "claude-sonnet-5-5"),
            rewrite_model=_str("CLAUDE_REWRITE_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=_int("MAX_OUTPUT_TOKENS", 1024),
            data_dir=Path(_str("DATA_DIR", "data")),
            docs_dir=Path(_str("DOCS_DIR", "docs")),
            auto_ingest=_bool("AUTO_INGEST", True),
            chunk_target_tokens=_int("CHUNK_TARGET_TOKENS", 450),
            chunk_overlap_tokens=_int("CHUNK_OVERLAP_TOKENS", 50),
            embedding_backend=_str("EMBEDDING_BACKEND", "fastembed").lower(),
            embedding_model=_str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            allow_embed_fallback=_bool("ALLOW_EMBED_FALLBACK", True),
            rerank_enabled=_bool("RERANK_ENABLED", True),
            rerank_model=_str("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
            rerank_min_score=_float("RERANK_MIN_SCORE", -4.0),
            candidates_k=_int("CANDIDATES_K", 30),
            final_k=_int("FINAL_K", 6),
            history_turns=_int("HISTORY_TURNS", 6),
        )
        return replace(s, **overrides) if overrides else s
