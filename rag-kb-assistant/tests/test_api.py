"""HTTP API tests. Need `pip install -r requirements-dev.txt` (fastapi + httpx).

    python -m unittest discover -t . -s tests -v
"""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

try:
    from fastapi.testclient import TestClient

    HAVE_FASTAPI = True
except Exception:  # fastapi or httpx not installed
    HAVE_FASTAPI = False

from app.embeddings import HashEmbedder
from app.llm import AnthropicLLM
from tests.test_core import HANDBOOK, LEAVE, Env, FakeLLM

ACL = {"default": ["all"], "rules": [{"path": "hr/*", "groups": ["hr"]}]}
KEY = {"X-API-Key": "k"}


@unittest.skipUnless(HAVE_FASTAPI, "fastapi / httpx not installed")
class ApiTests(unittest.TestCase):
    def make_client(self, llm=None, **overrides):
        env = Env({"handbook.md": HANDBOOK, "hr/leave.md": LEAVE}, acl=ACL)
        self.addCleanup(env.close)
        env.sync()
        settings = replace(env.settings, auto_ingest=False, rag_api_key="k", **overrides)
        from app.main import create_app

        app = create_app(settings, llm=llm or FakeLLM(), embedder=HashEmbedder(), reranker=None)
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_health_needs_no_auth(self):
        body = self.make_client().get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["ready"])
        self.assertTrue(body["llm_configured"])

    def test_ask_requires_api_key(self):
        client = self.make_client()
        self.assertEqual(client.post("/ask", json={"question": "hotel limit"}).status_code, 401)

    def test_ask_returns_cited_answer(self):
        client = self.make_client()
        r = client.post("/ask", json={"question": "hotel limit per night"}, headers=KEY)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["refused"])
        self.assertIn("[S1]", body["answer"])
        self.assertTrue(body["sources"][0]["cited"])

    def test_permissions_follow_groups_header(self):
        client = self.make_client()
        q = {"question": "annual leave days paid"}
        no_group = client.post("/ask", json=q, headers=KEY).json()
        self.assertNotIn("hr/leave.md", [s["path"] for s in no_group["sources"]])
        hr = client.post("/ask", json=q, headers={**KEY, "X-User-Groups": "hr"}).json()
        self.assertIn("hr/leave.md", [s["path"] for s in hr["sources"]])

    def test_stream_emits_meta_tokens_done(self):
        client = self.make_client()
        with client.stream("POST", "/ask/stream", json={"question": "hotel limit per night"}, headers=KEY) as r:
            self.assertEqual(r.status_code, 200)
            text = "".join(r.iter_text())
        events = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
        self.assertEqual(events[0], "meta")
        self.assertEqual(events[-1], "done")
        self.assertIn("token", events)
        done = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ")][-1]
        self.assertIn("answer", done)

    def test_feedback(self):
        client = self.make_client()
        r = client.post("/feedback", json={"answer_id": "abc", "rating": "up"}, headers=KEY)
        self.assertEqual(r.json(), {"ok": True})
        bad = client.post("/feedback", json={"answer_id": "abc", "rating": "meh"}, headers=KEY)
        self.assertEqual(bad.status_code, 422)

    def test_admin_sync_disabled_without_key(self):
        client = self.make_client()
        self.assertEqual(client.post("/ingest/sync").status_code, 403)

    def test_admin_sync_with_key(self):
        client = self.make_client(admin_key="adm")
        self.assertEqual(client.post("/ingest/sync", headers={"X-Admin-Key": "wrong"}).status_code, 401)
        r = client.post("/ingest/sync", headers={"X-Admin-Key": "adm"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["errors"], [])

    def test_missing_anthropic_key_gives_503_not_crash(self):
        client = self.make_client(llm=AnthropicLLM(""))
        r = client.post("/ask", json={"question": "hotel limit"}, headers=KEY)
        self.assertEqual(r.status_code, 503)
        self.assertIn("ANTHROPIC_API_KEY", r.json()["detail"])
        self.assertEqual(client.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
