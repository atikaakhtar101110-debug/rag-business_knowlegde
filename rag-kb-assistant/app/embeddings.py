"""Embedding and reranking backends.

Default: FastEmbed (ONNX, no PyTorch) with BAAI/bge-small-en-v1.5.
Fallback: a deterministic hashing embedder, used only if the model cannot be
loaded and ALLOW_EMBED_FALLBACK is true. It keeps the service running with
weaker semantic search (keyword search still works fully). The active backend
is reported by /health and recorded in the index; if it changes, the index is
rebuilt automatically.
"""
from __future__ import annotations

import logging
import re
import zlib
from typing import Protocol, Sequence

import numpy as np

from .config import Settings

log = logging.getLogger("rag.embeddings")

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Embedder(Protocol):
    id: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    """Feature-hashing bag of words and bigrams. Deterministic, no downloads."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.id = f"hash:{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = tokenize(text)
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for gram in grams:
            h = zlib.crc32(gram.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[h % self.dim] += sign
        return np.sign(vec) * np.log1p(np.abs(vec))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text))[0]


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.id = f"fastembed:{model_name}"
        probe = np.array(list(self._model.embed(["dimension probe"])))
        self.dim = int(probe.shape[1])

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self._model.embed(list(texts), batch_size=32))
        return _normalize(np.vstack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        vectors = list(self._model.query_embed(text))
        return _normalize(np.vstack(vectors))[0]


class Reranker(Protocol):
    id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)
        self.id = f"fastembed-rerank:{model_name}"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(passages))]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    try:
        return FastEmbedEmbedder(settings.embedding_model)
    except Exception as exc:
        if not settings.allow_embed_fallback:
            raise RuntimeError(
                f"Could not load embedding model '{settings.embedding_model}': {exc}"
            ) from exc
        log.warning(
            "Embedding model '%s' could not be loaded (%s). Falling back to the hashing "
            "embedder: keyword search stays accurate, semantic search is weaker. "
            "Fix network or cache access and restart to use the real model.",
            settings.embedding_model,
            exc,
        )
        return HashEmbedder()


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.rerank_enabled:
        return None
    try:
        return FastEmbedReranker(settings.rerank_model)
    except Exception as exc:
        log.warning(
            "Reranker '%s' could not be loaded (%s). Continuing without reranking.",
            settings.rerank_model,
            exc,
        )
        return None
