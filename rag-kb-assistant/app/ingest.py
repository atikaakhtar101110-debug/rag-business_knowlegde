"""Reconcile the index with the documents folder.

- new or changed files (content or permissions) are re-parsed and re-embedded
- files that disappeared are removed from the index
- unchanged files are skipped (content hash)
- one bad file never aborts the sync; it is reported in `errors`

Permissions live in docs/acl.json:

    {
      "default": ["all"],
      "rules": [
        {"path": "hr/*", "groups": ["hr", "admins"]},
        {"path": "finance/*", "groups": ["finance"]}
      ]
    }

The first matching rule wins. "all" means every authenticated user.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_document, chunk_id
from .config import Settings
from .embeddings import Embedder
from .parsing import SUPPORTED_EXTENSIONS, ParseError, parse_file
from .store import Store

log = logging.getLogger("rag.ingest")

ACL_FILE = "acl.json"


def load_acl(docs_dir: Path) -> tuple[list[tuple[str, list[str]]], list[str]]:
    path = docs_dir / ACL_FILE
    if not path.exists():
        return [], ["all"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rules = [(str(r["path"]), [str(g) for g in r["groups"]]) for r in data.get("rules", [])]
        default = [str(g) for g in data.get("default", ["all"])] or ["all"]
        return rules, default
    except Exception as exc:
        # Fail closed: a broken ACL file must never widen access.
        log.error("Invalid %s (%s). Treating every document as admin-only.", ACL_FILE, exc)
        return [], ["admins"]


def resolve_acl(rel_path: str, rules: list[tuple[str, list[str]]], default: list[str]) -> list[str]:
    for pattern, groups in rules:
        if fnmatch.fnmatch(rel_path, pattern):
            return groups
    return default


def _discover(docs_dir: Path) -> list[Path]:
    found = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(docs_dir)
        if any(part.startswith(".") for part in rel.parts) or path.name == ACL_FILE:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    return found


def sync(store: Store, embedder: Embedder, settings: Settings) -> dict:
    report = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "chunks": 0, "errors": []}

    if store.get_meta("embed_id") != embedder.id:
        if store.get_meta("embed_id") is not None:
            log.info("Embedding backend changed; rebuilding the whole index.")
        store.clear_all()
        store.set_meta("embed_id", embedder.id)

    docs_dir = settings.docs_dir
    if not docs_dir.exists():
        report["errors"].append(f"documents folder not found: {docs_dir}")
        return report

    rules, default = load_acl(docs_dir)
    known = store.document_hashes()
    seen: set[str] = set()

    for path in _discover(docs_dir):
        rel = path.relative_to(docs_dir).as_posix()
        seen.add(rel)
        acl = resolve_acl(rel, rules, default)
        digest = hashlib.sha256()
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            report["errors"].append(f"{rel}: {exc}")
            continue
        digest.update(json.dumps(sorted(acl)).encode())
        digest.update(f"{settings.chunk_target_tokens}:{settings.chunk_overlap_tokens}".encode())
        content_hash = digest.hexdigest()

        existing = known.get(rel)
        if existing and existing[1] == content_hash:
            report["unchanged"] += 1
            continue

        try:
            parsed = parse_file(path)
            chunks = chunk_document(parsed.text, settings.chunk_target_tokens, settings.chunk_overlap_tokens)
            if not chunks:
                raise ParseError(f"{rel}: no text found")
            doc_id = hashlib.sha1(rel.encode()).hexdigest()[:16]
            vectors = embedder.embed_documents([c.embed_text(parsed.title) for c in chunks])
            updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
            count = store.replace_document(
                doc_id=doc_id,
                path=rel,
                title=parsed.title,
                content_hash=content_hash,
                acl=acl,
                updated=updated,
                chunks=[
                    (chunk_id(doc_id, c.position, c.text), c.position, c.section, c.text, vectors[i])
                    for i, c in enumerate(chunks)
                ],
            )
        except Exception as exc:  # one bad file must not stop the sync
            log.warning("Skipping %s: %s", rel, exc)
            report["errors"].append(f"{rel}: {exc}")
            continue
        report["updated" if existing else "added"] += 1
        report["chunks"] += count

    for rel in set(known) - seen:
        store.delete_path(rel)
        report["removed"] += 1

    return report
