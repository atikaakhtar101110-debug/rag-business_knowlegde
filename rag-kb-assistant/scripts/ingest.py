"""Run one ingestion sync from the command line.

    python -m scripts.ingest
"""
from __future__ import annotations

import json
import logging

from app import ingest
from app.config import Settings
from app.embeddings import build_embedder
from app.store import Store


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.db_path)
    embedder = build_embedder(settings)
    report = ingest.sync(store, embedder, settings)
    print(json.dumps({**report, "embedder": embedder.id, "index": store.counts()}, indent=2))
    store.close()
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
