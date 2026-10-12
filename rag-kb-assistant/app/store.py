"""SQLite persistence plus an immutable in-memory search snapshot.

SQLite is the source of truth (documents, chunks, vectors, feedback).
A Snapshot holds the vectors in a NumPy matrix and a BM25 index for fast
hybrid search; a new Snapshot is built after each sync and swapped in
atomically, so queries never see a half-updated index.

This design is meant for corpora up to roughly 100k chunks on one node. For
more, move the Snapshot's job to pgvector or OpenSearch behind the same
interface (see README).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math
from collections import Counter

import numpy as np

from .embeddings import tokenize


class BM25:
    """Okapi BM25 over an inverted index (non-negative IDF variant)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.doc_len = np.array([len(doc) for doc in corpus], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n and self.doc_len.sum() else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(corpus):
            for term, freq in Counter(doc).items():
                self.postings.setdefault(term, []).append((i, freq))
        self.idf = {
            term: math.log(1.0 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float64)
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for i, freq in postings:
                norm = freq + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * freq * (self.k1 + 1.0) / norm
        return scores

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    acl TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    rating TEXT NOT NULL,
    comment TEXT NOT NULL,
    created TEXT NOT NULL
);
"""


@dataclass
class ChunkRow:
    chunk_id: str
    doc_id: str
    path: str
    title: str
    section: str
    text: str
    position: int
    acl: frozenset
    updated: str


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---- meta ----
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    # ---- documents ----
    def document_hashes(self) -> dict[str, tuple[str, str]]:
        """path -> (doc_id, content_hash)"""
        with self._lock:
            rows = self._conn.execute("SELECT path, doc_id, content_hash FROM documents").fetchall()
        return {r["path"]: (r["doc_id"], r["content_hash"]) for r in rows}

    def replace_document(
        self,
        *,
        doc_id: str,
        path: str,
        title: str,
        content_hash: str,
        acl: list[str],
        updated: str,
        chunks: Iterable[tuple[str, int, str, str, np.ndarray]],
    ) -> int:
        """Atomically replace a document and its chunks. Returns chunk count."""
        count = 0
        with self._lock:
            try:
                self._conn.execute("DELETE FROM documents WHERE doc_id=? OR path=?", (doc_id, path))
                self._conn.execute(
                    "INSERT INTO documents(doc_id,path,title,content_hash,acl,updated) VALUES(?,?,?,?,?,?)",
                    (doc_id, path, title, content_hash, json.dumps(sorted(acl)), updated),
                )
                for chunk_id, position, section, text, vector in chunks:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO chunks(chunk_id,doc_id,position,section,text,vector) "
                        "VALUES(?,?,?,?,?,?)",
                        (chunk_id, doc_id, position, section, text,
                         np.asarray(vector, dtype=np.float32).tobytes()),
                    )
                    count += 1
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return count

    def delete_path(self, path: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM documents WHERE path=?", (path,))
            self._conn.commit()

    def clear_all(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM documents")
            self._conn.commit()

    def counts(self) -> dict[str, int]:
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
            chunks = self._conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        return {"documents": docs, "chunks": chunks}

    # ---- feedback ----
    def add_feedback(self, answer_id: str, user_id: str, rating: str, comment: str, created: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback(answer_id,user_id,rating,comment,created) VALUES(?,?,?,?,?)",
                (answer_id, user_id, rating, comment, created),
            )
            self._conn.commit()

    # ---- snapshot ----
    def snapshot(self, dim: int | None) -> "Snapshot":
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.chunk_id, c.doc_id, c.position, c.section, c.text, c.vector, "
                "d.path, d.title, d.acl, d.updated "
                "FROM chunks c JOIN documents d ON d.doc_id=c.doc_id "
                "ORDER BY d.path, c.position"
            ).fetchall()
        chunks: list[ChunkRow] = []
        vectors: list[np.ndarray] = []
        for r in rows:
            vec = np.frombuffer(r["vector"], dtype=np.float32)
            if dim is not None and vec.shape[0] != dim:
                continue  # stale vector from a different model; sync will rebuild
            chunks.append(
                ChunkRow(
                    chunk_id=r["chunk_id"], doc_id=r["doc_id"], path=r["path"], title=r["title"],
                    section=r["section"], text=r["text"], position=r["position"],
                    acl=frozenset(json.loads(r["acl"])), updated=r["updated"],
                )
            )
            vectors.append(vec)
        matrix = np.vstack(vectors) if vectors else np.zeros((0, dim or 1), dtype=np.float32)
        return Snapshot(chunks, matrix)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Snapshot:
    """Read-only search structures built from the store."""

    def __init__(self, chunks: list[ChunkRow], matrix: np.ndarray):
        self.chunks = chunks
        self.matrix = matrix
        corpus = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        self._bm25 = BM25(corpus) if corpus and any(corpus) else None
        self._mask_cache: dict[frozenset, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def allowed_mask(self, groups: frozenset) -> np.ndarray:
        """Boolean mask of chunks this caller may read (document ACL ∩ user groups)."""
        cached = self._mask_cache.get(groups)
        if cached is not None:
            return cached
        mask = np.fromiter(
            (("all" in c.acl) or bool(c.acl & groups) for c in self.chunks),
            dtype=bool,
            count=len(self.chunks),
        )
        self._mask_cache[groups] = mask
        return mask

    def vector_search(self, query_vec: np.ndarray, mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self.chunks) == 0 or not mask.any():
            return []
        scores = self.matrix @ query_vec.astype(np.float32)
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def keyword_search(self, query: str, mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None or not mask.any():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]
