"""Score retrieval against the golden question set (no LLM or API key needed).

    python -m scripts.evaluate --golden eval/golden.jsonl --docs docs

Reports recall@k, MRR, permission leaks (must be 0) and refusal behaviour.
Exit code is 1 if recall@k is below --min-recall or any permission leak occurs,
so it can gate a release in CI.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from app import ingest
from app.config import Settings
from app.embeddings import build_embedder, build_reranker
from app.retrieval import retrieve
from app.store import Store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.8)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings.from_env(docs_dir=Path(args.docs), data_dir=Path(tmp), final_k=args.k)
        store = Store(settings.db_path)
        embedder = build_embedder(settings)
        reranker = build_reranker(settings)
        report = ingest.sync(store, embedder, settings)
        if report["errors"]:
            print("Ingestion errors:", *report["errors"], sep="\n  ")
        snapshot = store.snapshot(embedder.dim)

        rows = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
        hits = reciprocal = answerable = 0
        leaks = 0
        refusal_checks = refusal_ok = 0

        for row in rows:
            groups = frozenset(row.get("groups", []))
            result = retrieve(snapshot, embedder, reranker, row["question"], groups, settings)
            paths = []
            for p in result.passages:
                if p.chunk.path not in paths:
                    paths.append(p.chunk.path)
                if not ("all" in p.chunk.acl or p.chunk.acl & groups):
                    leaks += 1
            expected = row.get("expected_sources", [])
            if expected:
                answerable += 1
                ranks = [paths.index(e) + 1 for e in expected if e in paths]
                hits += 1 if ranks else 0
                reciprocal += 1.0 / min(ranks) if ranks else 0.0
            if row.get("should_refuse"):
                refusal_checks += 1
                refused = (not result.passages) or (
                    result.reranked and result.top_score is not None
                    and result.top_score < settings.rerank_min_score
                )
                refusal_ok += 1 if refused else 0
            status = "ok " if (not expected or any(e in paths for e in expected)) else "MISS"
            print(f"[{status}] {row['question'][:70]:<70} -> {paths[:3]}")

        recall = hits / answerable if answerable else 1.0
        mrr = reciprocal / answerable if answerable else 1.0
        print("\n--- summary ---")
        print(f"embedder:        {embedder.id}")
        print(f"reranker:        {reranker.id if reranker else 'off'}")
        print(f"recall@{args.k}:        {recall:.2f}  ({hits}/{answerable})")
        print(f"MRR:             {mrr:.2f}")
        print(f"permission leaks: {leaks}  (must be 0)")
        if refusal_checks:
            note = "" if reranker else "  (needs the reranker to be meaningful)"
            print(f"refusal correct: {refusal_ok}/{refusal_checks}{note}")
        store.close()
        return 0 if (recall >= args.min_recall and leaks == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
