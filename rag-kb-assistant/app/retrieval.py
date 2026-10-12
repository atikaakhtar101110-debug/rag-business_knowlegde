"""Hybrid retrieval: vector + BM25, reciprocal rank fusion, optional rerank.

Permission filtering happens inside the search (masked scores), never after
results are returned.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder, Reranker
from .store import ChunkRow, Snapshot

RRF_K = 60


@dataclass
class Passage:
    label: str          # "S1", "S2", ...
    chunk: ChunkRow
    score: float        # reranker score, or fused rank score if no reranker


@dataclass
class Retrieval:
    passages: list[Passage]
    reranked: bool
    top_score: float | None
    candidates: int


def _fuse(*rankings: list[tuple[int, float]]) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def retrieve(
    snapshot: Snapshot,
    embedder: Embedder,
    reranker: Reranker | None,
    query: str,
    groups: frozenset,
    settings: Settings,
) -> Retrieval:
    if len(snapshot) == 0 or not query.strip():
        return Retrieval([], False, None, 0)

    mask = snapshot.allowed_mask(groups)
    k = settings.candidates_k
    vector_hits = snapshot.vector_search(embedder.embed_query(query), mask, k)
    keyword_hits = snapshot.keyword_search(query, mask, k)
    fused = _fuse(vector_hits, keyword_hits)[:k]
    if not fused:
        return Retrieval([], False, None, 0)

    candidates = [(snapshot.chunks[i], score) for i, score in fused]

    if reranker is not None:
        try:
            texts = [f"{c.title} > {c.section}\n{c.text}" if c.section else f"{c.title}\n{c.text}"
                     for c, _ in candidates]
            scores = reranker.score(query, texts)
            ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
            top = ranked[: settings.final_k]
            passages = [Passage(f"S{n}", c, float(s)) for n, ((c, _), s) in enumerate(top, start=1)]
            return Retrieval(passages, True, passages[0].score if passages else None, len(candidates))
        except Exception:
            pass  # reranker failure must not break answering; use fused order

    top = candidates[: settings.final_k]
    passages = [Passage(f"S{n}", c, float(s)) for n, (c, s) in enumerate(top, start=1)]
    return Retrieval(passages, False, passages[0].score if passages else None, len(candidates))
