"""FastAPI application.

Run:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000

Identity: put this service behind your gateway / SSO proxy. The gateway
authenticates the user, then calls this API with
    X-API-Key:      shared secret (RAG_API_KEY)
    X-User-Id:      the user's id or email
    X-User-Groups:  comma-separated groups (matched against docs/acl.json)
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import ingest
from .audit import AuditLog
from .config import Settings
from .embeddings import Embedder, Reranker, build_embedder, build_reranker
from .generation import Pipeline, Prepared, Turn
from .llm import LLM, AnthropicLLM, LLMError
from .store import Snapshot, Store

log = logging.getLogger("rag.app")


# ---------------- request / response models ----------------
class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=40)


class FeedbackRequest(BaseModel):
    answer_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=2000)


@dataclass
class User:
    id: str
    groups: frozenset


# ---------------- application state ----------------
class AppState:
    def __init__(self, settings: Settings, llm: LLM | None, embedder: Embedder | None, reranker: Reranker | None):
        self.settings = settings
        self.store = Store(settings.db_path)
        self.embedder = embedder or build_embedder(settings)
        self.reranker = reranker if reranker is not None else build_reranker(settings)
        # line 69, inside AppState.__init__: the key from Settings goes to the client
        self.llm = llm or AnthropicLLM(settings.anthropic_api_key)
        self.pipeline = Pipeline(settings, self.embedder, self.reranker, self.llm)
        self.audit = AuditLog(settings.audit_path)
        self.sync_lock = threading.Lock()
        self.sync_status: dict = {"state": "idle", "last_report": None, "finished": None}
        self.snapshot: Snapshot = self._empty()
        self.refresh_snapshot()

    def _empty(self) -> Snapshot:
        return Snapshot([], np.zeros((0, self.embedder.dim), dtype=np.float32))

    def refresh_snapshot(self) -> None:
        if self.store.get_meta("embed_id") == self.embedder.id:
            self.snapshot = self.store.snapshot(self.embedder.dim)  # atomic reference swap
        else:
            self.snapshot = self._empty()

    def run_sync(self) -> dict | None:
        if not self.sync_lock.acquire(blocking=False):
            return None
        try:
            self.sync_status["state"] = "running"
            report = ingest.sync(self.store, self.embedder, self.settings)
            self.refresh_snapshot()
            self.sync_status.update(
                state="idle", last_report=report,
                finished=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            log.info("Sync finished: %s", {k: v for k, v in report.items() if k != "errors"})
            return report
        except Exception as exc:
            log.exception("Sync failed")
            self.sync_status.update(state="error", last_report={"errors": [str(exc)]})
            return {"errors": [str(exc)]}
        finally:
            self.sync_lock.release()


def _state(request: Request) -> AppState:
    return request.app.state.rag


def require_user(
    request: Request,
    x_api_key: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_user_groups: str | None = Header(default=None),
) -> User:
    settings = _state(request).settings
    if settings.rag_api_key:
        if not x_api_key or not secrets.compare_digest(x_api_key, settings.rag_api_key):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")
    groups = frozenset(g.strip() for g in (x_user_groups or "").split(",") if g.strip())
    return User(id=(x_user_id or "anonymous")[:200], groups=groups)


def require_admin(request: Request, x_admin_key: str | None = Header(default=None)) -> None:
    settings = _state(request).settings
    if not settings.admin_key:
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled (RAG_ADMIN_KEY is not set).")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_key):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Key.")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------- app factory ----------------
def create_app(
    settings: Settings | None = None,
    llm: LLM | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        state = AppState(settings, llm, embedder, reranker)
       # lines 153-154, inside the startup (lifespan) function: warn if the key is missing
        if not settings.anthropic_api_key and llm is None:
            log.warning("ANTHROPIC_API_KEY is not set: the API will start, but /ask returns 503 until it is.")
        if not settings.rag_api_key:
            log.warning("RAG_API_KEY is not set: anyone who can reach this service can call it and claim any group.")
        if settings.auto_ingest and settings.docs_dir.exists():
            threading.Thread(target=state.run_sync, name="initial-sync", daemon=True).start()
        elif settings.auto_ingest:
            log.warning("Documents folder %s does not exist; nothing to ingest.", settings.docs_dir)
        yield
        state.store.close()

    app = FastAPI(title="Business Knowledge Base Assistant", version="1.0.0", lifespan=lifespan)

    def _prepare(request: Request, req: AskRequest, user: User) -> tuple[AppState, Prepared]:
        st = _state(request)
        # line 169, inside _prepare(): block requests when there is no key
       if not st.llm.configured:
           raise HTTPException(status_code=503, detail="The server has no ANTHROPIC_API_KEY configured.")
        if len(st.snapshot) == 0 and st.sync_status["state"] == "running":
            raise HTTPException(status_code=503, detail="The knowledge base index is still being built. Retry shortly.")
        history = [Turn(h.role, h.content) for h in req.history]
        prep = st.pipeline.prepare(st.snapshot, req.question.strip(), history, user.groups)
        return st, prep

    def _audit(st: AppState, user: User, prep: Prepared, result: dict) -> None:
        st.audit.write(
            answer_id=prep.answer_id,
            user_id=user.id,
            groups=sorted(user.groups),
            question=prep.question,
            standalone_question=prep.standalone,
            chunk_ids=[p.chunk.chunk_id for p in prep.retrieval.passages],
            cited=[s["label"] for s in result["sources"] if s["cited"]],
            invalid_citations=result["invalid_citations"],
            refused=result["refused"],
            answer=result["answer"],
            timings=result["timings"],
        )

    @app.get("/")
    def root() -> dict:
        return {"service": "Business Knowledge Base Assistant", "docs": "/docs", "health": "/health"}

    @app.get("/health")
    def health(request: Request) -> dict:
        st = _state(request)
        return {
            "status": "ok",
            "ready": len(st.snapshot) > 0,
            "llm_configured": st.llm.configured,
            "embedder": st.embedder.id,
            "reranker": st.reranker.id if st.reranker else None,
            "index": st.store.counts(),
            "sync": st.sync_status,
        }

    @app.post("/ask")
    def ask(req: AskRequest, request: Request, user: User = Depends(require_user)) -> dict:
        st, prep = _prepare(request, req, user)
        try:
            result = st.pipeline.answer(prep)
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        _audit(st, user, prep, result)
        return result

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest, request: Request, user: User = Depends(require_user)):
        st, prep = _prepare(request, req, user)

        def events():
            yield _sse(
                "meta",
                {
                    "answer_id": prep.answer_id,
                    "standalone_question": prep.standalone,
                    "refused": prep.refused,
                    "sources": [
                        {"label": p.label, "title": p.chunk.title, "section": p.chunk.section, "path": p.chunk.path}
                        for p in prep.retrieval.passages
                    ],
                },
            )
            for item in st.pipeline.answer_stream(prep):
                if item["event"] == "done":
                    _audit(st, user, prep, item["data"])
                yield _sse(item["event"], item["data"])

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/feedback")
    def feedback(req: FeedbackRequest, request: Request, user: User = Depends(require_user)) -> dict:
        _state(request).store.add_feedback(
            req.answer_id, user.id, req.rating, req.comment,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        return {"ok": True}

    @app.post("/ingest/sync", dependencies=[Depends(require_admin)])
    def ingest_sync(request: Request) -> dict:
        report = _state(request).run_sync()
        if report is None:
            raise HTTPException(status_code=409, detail="A sync is already running.")
        return report

    return app
