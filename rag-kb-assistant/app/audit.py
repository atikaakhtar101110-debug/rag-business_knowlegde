"""Append-only JSONL audit log: who asked what, which chunks were used, what came back."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("rag.audit")


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Cannot create audit directory: %s", exc)

    def write(self, **fields) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # auditing must never take the API down
            log.warning("Audit write failed: %s", exc)
