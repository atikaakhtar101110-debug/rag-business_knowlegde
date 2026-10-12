# Business Knowledge Base Assistant (RAG)

A deployable question-answering service over your company documents. It implements the design in the RAG Development Guide: structure-aware chunking, hybrid search (vectors + BM25) with rank fusion, optional cross-encoder reranking, permission-aware retrieval, cited answers with citation validation, refusal when the knowledge base has no answer, streaming, feedback, audit logging and an evaluation script.

## Quick start

**The only secret you must provide is the environment variable `ANTHROPIC_API_KEY`.**

### Docker (recommended)

```bash
cp .env.example .env          # then put your key in ANTHROPIC_API_KEY=
docker compose up --build
```

Put your documents in `./docs` (Markdown, TXT, HTML, PDF, DOCX). The service indexes them at startup. Open http://localhost:8000/docs for the interactive API page.

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

First start downloads two small models (about 100 MB total) unless they were baked into the Docker image. Until the index is built, `/ask` returns 503 "still being built"; check `GET /health`.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | **Yes** | Claude API key. Without it the service starts but `/ask` returns 503. |
| `RAG_API_KEY` | Recommended | If set, every call must send `X-API-Key: <value>`. |
| `RAG_ADMIN_KEY` | Optional | Enables `POST /ingest/sync`; callers send `X-Admin-Key: <value>`. |
| `CLAUDE_MODEL` | No | Answer model. Default `claude-sonnet-5-5`. |
| `CLAUDE_REWRITE_MODEL` | No | Cheap model for follow-up rewriting. Default `claude-haiku-4-5-20251001`. |
| `RERANK_ENABLED` | No | `true` (default) or `false`. |
| `RERANK_MIN_SCORE` | No | Reranker score below which the service refuses to answer. Default `-4.0`; tune it on your golden set. |
| `CANDIDATES_K`, `FINAL_K` | No | Candidates fetched (30) and passages sent to the model (6). |
| `CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP_TOKENS` | No | Chunk size (450) and overlap (50). |
| `DOCS_DIR`, `DATA_DIR`, `AUTO_INGEST` | No | Folders and whether to sync on start. |

## Using the API

Your gateway or SSO proxy authenticates the user, then calls this service with the user's identity:

```bash
curl -s http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAG_API_KEY" \
  -H "X-User-Id: alice@example.com" \
  -H "X-User-Groups: hr,finance" \
  -d '{"question": "What is the hotel limit in a high-cost city?"}'
```

| Endpoint | Purpose |
| --- | --- |
| `POST /ask` | JSON answer with `answer`, `sources` (label, title, section, path, updated, cited), `refused`, `invalid_citations`, `timings`. |
| `POST /ask/stream` | Same, as server-sent events: `meta` (sources), many `token`, then `done` (or `error`). |
| `POST /feedback` | `{"answer_id": "...", "rating": "up"|"down", "comment": "..."}` |
| `POST /ingest/sync` | Admin only. Re-syncs the documents folder. |
| `GET /health` | Status, index size, active embedder and reranker, sync state. |

For follow-up questions, send earlier turns in `history` (`[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`); the service rewrites the follow-up into a standalone search query.

## Permissions

Edit `docs/acl.json`. The first matching rule wins; `"all"` means every authenticated user:

```json
{
  "default": ["all"],
  "rules": [{"path": "hr/*", "groups": ["hr", "admins"]}]
}
```

Permissions are enforced inside the search, before ranking. Changing a rule re-indexes the affected documents on the next sync. A broken `acl.json` fails closed (documents become admin-only). Important: the service trusts `X-User-Groups`, so it must only be reachable through your gateway and `RAG_API_KEY` must be set.

## Keeping the index current

Add, edit or delete files in `docs/`, then either restart, call `POST /ingest/sync`, or run `python -m scripts.ingest`. Unchanged files are skipped, removed files are removed from the index, and one unreadable file never stops the sync (it is listed in `errors`). If you change the embedding model, the index rebuilds itself.

## Evaluate before you ship

Edit `eval/golden.jsonl` with real questions, expected source files and the user's groups, then:

```bash
python -m scripts.evaluate --golden eval/golden.jsonl --docs docs
```

It reports recall@k, MRR and permission leaks (must be 0), and exits non-zero on failure, so it can gate releases in CI. Run the tests with `python -m unittest discover -t . -s tests -v`.

## Data produced

`data/index.sqlite` (index and feedback) and `data/audit.jsonl` (one line per question: user, question, chunk IDs, answer). Both contain business content; restrict access and set a retention policy.

## Limits and scaling

- One worker, one node: the search index is held in memory and rebuilt after each sync. This suits corpora up to roughly 100,000 chunks. Beyond that, move storage and search to pgvector or OpenSearch behind `app/store.py`.
- Scanned PDFs have no text and are reported as errors; OCR them first.
- If the embedding model cannot be loaded, the service falls back to a keyword-strong hashing embedder and says so in the logs and in `/health`. Semantic quality is lower in that mode.
- Retrieval refusal depends on the reranker; with `RERANK_ENABLED=false`, refusals rely on the model following the prompt.
