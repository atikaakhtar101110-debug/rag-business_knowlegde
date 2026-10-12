"""Core pipeline tests. Run with:  python -m unittest discover -t . -s tests -v

They use the hashing embedder and a fake LLM, so they need no downloads and no API key.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app import ingest
from app.chunking import chunk_document
from app.config import Settings
from app.embeddings import HashEmbedder
from app.generation import Pipeline, Turn, check_citations, sanitize_history
from app.retrieval import retrieve
from app.store import Store

HANDBOOK = """# Handbook

## Expenses

Hotels are reimbursed up to 150 USD per night. Meals are reimbursed up to 60 USD per day.

| Type | Limit |
| --- | --- |
| Hotel | 150 USD |
| Meals | 60 USD |

## Security

Report a lost laptop to the IT service desk within one hour.
"""

LEAVE = """# Leave

## Annual leave

Full-time employees receive 24 days of paid annual leave.
"""


class FakeLLM:
    configured = True

    def __init__(self, reply="Hotels are capped at 150 USD per night [S1]."):
        self.reply = reply
        self.calls = []

    def complete(self, *, model, system, user, max_tokens):
        self.calls.append(("complete", user))
        return "What is the hotel limit?"

    def stream(self, *, model, system, messages, max_tokens):
        self.calls.append(("stream", messages))
        for i in range(0, len(self.reply), 10):
            yield self.reply[i : i + 10]


class Env:
    """A throwaway docs folder + index."""

    def __init__(self, files: dict[str, str], acl: dict | None = None):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.docs = root / "docs"
        self.docs.mkdir()
        for name, content in files.items():
            path = self.docs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if acl is not None:
            (self.docs / "acl.json").write_text(json.dumps(acl), encoding="utf-8")
        self.settings = Settings(docs_dir=self.docs, data_dir=root / "data", embedding_backend="hash",
                                 rerank_enabled=False)
        self.settings.data_dir.mkdir()
        self.store = Store(self.settings.db_path)
        self.embedder = HashEmbedder()

    def sync(self):
        return ingest.sync(self.store, self.embedder, self.settings)

    def snapshot(self):
        return self.store.snapshot(self.embedder.dim)

    def close(self):
        self.store.close()
        self.tmp.cleanup()


class ChunkingTests(unittest.TestCase):
    def test_sections_and_tables(self):
        chunks = chunk_document(HANDBOOK, target_tokens=60, overlap_tokens=0)
        sections = {c.section for c in chunks}
        self.assertIn("Handbook > Expenses", sections)
        self.assertIn("Handbook > Security", sections)
        table_chunks = [c for c in chunks if "| Hotel | 150 USD |" in c.text]
        self.assertEqual(len(table_chunks), 1)
        self.assertIn("| Meals | 60 USD |", table_chunks[0].text)  # table not cut

    def test_long_paragraph_is_split(self):
        text = "# T\n\n" + " ".join(f"Sentence number {i} is here." for i in range(300))
        chunks = chunk_document(text, target_tokens=100, overlap_tokens=10)
        self.assertGreater(len(chunks), 3)
        self.assertEqual([c.position for c in chunks], list(range(len(chunks))))

    def test_empty_text(self):
        self.assertEqual(chunk_document("   \n\n"), [])


class IngestTests(unittest.TestCase):
    def test_sync_add_update_remove(self):
        env = Env({"handbook.md": HANDBOOK, "notes.txt": "Plain notes about onboarding."})
        self.addCleanup(env.close)
        first = env.sync()
        self.assertEqual((first["added"], first["errors"]), (2, []))
        again = env.sync()
        self.assertEqual((again["unchanged"], again["added"]), (2, 0))

        (env.docs / "notes.txt").write_text("Updated notes about onboarding and laptops.", encoding="utf-8")
        (env.docs / "handbook.md").unlink()
        third = env.sync()
        self.assertEqual((third["updated"], third["removed"]), (1, 1))
        paths = {c.path for c in env.snapshot().chunks}
        self.assertEqual(paths, {"notes.txt"})

    def test_bad_file_does_not_abort(self):
        env = Env({"ok.md": HANDBOOK})
        self.addCleanup(env.close)
        (env.docs / "broken.pdf").write_bytes(b"not a real pdf")
        report = env.sync()
        self.assertEqual(report["added"], 1)
        self.assertEqual(len(report["errors"]), 1)

    def test_embedder_change_rebuilds(self):
        env = Env({"handbook.md": HANDBOOK})
        self.addCleanup(env.close)
        env.sync()
        env.embedder = HashEmbedder(dim=128)
        report = env.sync()
        self.assertEqual(report["added"], 1)
        self.assertEqual(env.snapshot().matrix.shape[1], 128)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.env = Env(
            {"handbook.md": HANDBOOK, "hr/leave.md": LEAVE},
            acl={"default": ["all"], "rules": [{"path": "hr/*", "groups": ["hr"]}]},
        )
        self.addCleanup(self.env.close)
        self.env.sync()
        self.snap = self.env.snapshot()

    def _paths(self, query, groups=()):
        result = retrieve(self.snap, self.env.embedder, None, query, frozenset(groups), self.env.settings)
        return [p.chunk.path for p in result.passages]

    def test_finds_relevant_document(self):
        self.assertEqual(self._paths("hotel limit per night")[0], "handbook.md")

    def test_exact_term_via_keyword_search(self):
        self.assertEqual(self._paths("IT service desk laptop")[0], "handbook.md")

    def test_acl_blocks_restricted_documents(self):
        self.assertNotIn("hr/leave.md", self._paths("annual leave days"))
        self.assertNotIn("hr/leave.md", self._paths("annual leave days", groups=["finance"]))

    def test_acl_allows_group_members(self):
        self.assertEqual(self._paths("annual leave days", groups=["hr"])[0], "hr/leave.md")

    def test_acl_change_triggers_reindex(self):
        (self.env.docs / "acl.json").write_text(json.dumps({"default": ["all"], "rules": []}), encoding="utf-8")
        report = self.env.sync()
        self.assertEqual(report["updated"], 1)
        self.snap = self.env.snapshot()
        self.assertIn("hr/leave.md", self._paths("annual leave days"))

    def test_broken_acl_fails_closed(self):
        (self.env.docs / "acl.json").write_text("{not json", encoding="utf-8")
        self.env.sync()
        self.snap = self.env.snapshot()
        self.assertEqual(self._paths("hotel limit per night"), [])
        self.assertNotEqual(self._paths("hotel limit per night", groups=["admins"]), [])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.env = Env({"handbook.md": HANDBOOK})
        self.addCleanup(self.env.close)
        self.env.sync()
        self.snap = self.env.snapshot()

    def test_citation_validation(self):
        env = self.env
        result = retrieve(self.snap, env.embedder, None, "hotel limit", frozenset(), env.settings)
        valid, invalid = check_citations("A [S1]. B [S1] [S9].", result.passages)
        self.assertEqual((valid, invalid), (["S1"], ["S9"]))

    def test_history_is_sanitized(self):
        history = [Turn("assistant", "stray"), Turn("user", "q1"), Turn("assistant", "a1"), Turn("user", "dangling")]
        cleaned = sanitize_history(history, max_turns=6)
        self.assertEqual([m["role"] for m in cleaned], ["user", "assistant"])

    def test_answer_with_citations(self):
        llm = FakeLLM()
        pipe = Pipeline(self.env.settings, self.env.embedder, None, llm)
        prep = pipe.prepare(self.snap, "hotel limit per night", [], frozenset())
        out = pipe.answer(prep)
        self.assertFalse(out["refused"])
        self.assertEqual(out["invalid_citations"], [])
        self.assertTrue(out["sources"][0]["cited"])
        sent = llm.calls[-1][1][-1]["content"]
        self.assertIn("<sources>", sent)
        self.assertIn("Hotels are reimbursed", sent)

    def test_invalid_citation_is_flagged(self):
        pipe = Pipeline(self.env.settings, self.env.embedder, None, FakeLLM("Made up [S42]."))
        out = pipe.answer(pipe.prepare(self.snap, "hotel limit", [], frozenset()))
        self.assertEqual(out["invalid_citations"], ["S42"])

    def test_refuses_without_calling_model_when_nothing_retrieved(self):
        llm = FakeLLM()
        empty = Env({})
        self.addCleanup(empty.close)
        empty.sync()
        pipe = Pipeline(empty.settings, empty.embedder, None, llm)
        out = pipe.answer(pipe.prepare(empty.snapshot(), "anything", [], frozenset()))
        self.assertTrue(out["refused"])
        self.assertEqual(llm.calls, [])

    def test_follow_up_is_rewritten(self):
        llm = FakeLLM()
        pipe = Pipeline(self.env.settings, self.env.embedder, None, llm)
        history = [Turn("user", "What is the hotel limit?"), Turn("assistant", "150 USD [S1].")]
        prep = pipe.prepare(self.snap, "and for meals?", history, frozenset())
        self.assertEqual(prep.standalone, "What is the hotel limit?")  # FakeLLM rewrite output
        self.assertEqual(llm.calls[0][0], "complete")

    def test_stream_events(self):
        pipe = Pipeline(self.env.settings, self.env.embedder, None, FakeLLM())
        prep = pipe.prepare(self.snap, "hotel limit per night", [], frozenset())
        events = list(pipe.answer_stream(prep))
        self.assertEqual(events[-1]["event"], "done")
        self.assertTrue(all(e["event"] == "token" for e in events[:-1]))
        streamed = "".join(e["data"]["text"] for e in events[:-1])
        self.assertEqual(streamed.strip(), events[-1]["data"]["answer"])


if __name__ == "__main__":
    unittest.main()
