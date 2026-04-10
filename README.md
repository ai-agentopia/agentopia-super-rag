# agentopia-super-rag

Governed, multi-tenant retrieval service for the Agentopia platform.

This is the production source for the knowledge-api service. Extraction from `agentopia-protocol` is complete. See [docs/migration.md](docs/migration.md) for the historical record.

---

## Repo Purpose

`agentopia-super-rag` provides scoped, governed knowledge retrieval for Agentopia bots. Each bot has a bounded set of knowledge scopes it can query. The service enforces scope isolation at the API level — a bot cannot retrieve content outside its subscribed scopes regardless of query content.

This is not a general-purpose RAG library. It is a platform service designed for:
- multi-tenant, per-bot knowledge isolation
- operator-curated corpora (not open-ended crawling)
- governed ingest lifecycle with document versioning
- reproducible retrieval quality evaluation

---

## Architecture Role

```
bot-config-api (agentopia-protocol)
    │ internal X-Internal-Token
    ▼
agentopia-super-rag  ◄──── bot gateway (bearer token, per-bot K8s Secret)
    │
    ├── Qdrant          (vector store, per-scope collection)
    ├── PostgreSQL      (document lifecycle: active/superseded/deleted)
    └── OpenRouter      (embedding API: text-embedding-3-small, 1536d)
```

**Owns:**
- Knowledge ingest, chunking, embedding, and indexing
- Scoped semantic search (dense vector, production baseline)
- Document lifecycle management (two-phase atomic replace)
- Bot-scope binding lifecycle (K8s ArgoCD CRD annotation read)
- Per-scope auth enforcement (internal token + bot bearer)
- Retrieval quality evaluation framework

**Does not own:**
- Workflow orchestration (Temporal, `agentopia-protocol`)
- Bot identity and deployment (bot-config-api, `agentopia-protocol`)
- LLM routing and provider failover (`agentopia-llm-proxy`)
- Reasoning plane / planner graph (`agentopia-graph-executor`)
- UI / operator console (`agentopia-ui`)
- Infrastructure and deployment manifests (`agentopia-infra`)

---

## Runtime Dependencies

| Dependency | Purpose | Required |
|---|---|---|
| Qdrant | Vector storage and ANN search | Yes |
| PostgreSQL | Document lifecycle store | Yes |
| Redis | (future) Caching layer | No (not yet used) |
| OpenRouter | Embedding API (`text-embedding-3-small`) | Yes |
| K8s API | Bot-scope binding cache (reads ArgoCD CRD annotations) | Yes (production) |

K8s API access uses in-cluster service account. In local dev, binding cache falls back gracefully when K8s is unreachable.

---

## Local Dev Workflow

```bash
# Clone and install dependencies
git clone git@github.com:ai-agentopia/agentopia-super-rag.git
cd agentopia-super-rag
pip install -r requirements.txt

# Start service (requires Qdrant + Postgres)
export QDRANT_URL=http://localhost:6333
export POSTGRES_DSN=postgresql://user:pass@localhost:5432/agentopia
export EMBEDDING_API_URL=https://openrouter.ai/api/v1/embeddings
export EMBEDDING_API_KEY=<key>
export KNOWLEDGE_API_INTERNAL_TOKEN=<token>
export K8S_NAMESPACE=agentopia

cd src && uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# Fast gate (no external deps)
PYTHONPATH=src python -m pytest tests/ -m "not integration and not e2e" -x -q

# Integration gate (requires Qdrant + Postgres)
PYTHONPATH=src python -m pytest tests/ -m "integration" -x -q
```

> **Known local env limitation:** `python-multipart` is required by FastAPI for the file upload endpoint (`POST /{scope}/ingest`). Without it, `TestClient`-based API tests fail at import. Install with `pip install python-multipart` to run the full test suite. Service startup and all non-API-layer tests are unaffected.

---

## CI/CD Overview

- CI runs on push to `dev` and `main`: fast gate (`.github/workflows/ci.yml`) + Docker build
- CI pushes `ghcr.io/ai-agentopia/knowledge-api:dev-{sha}` from this repo
- ArgoCD Image Updater picks up new `dev-{sha}` tags and deploys to `agentopia-dev`
- `agentopia-protocol` no longer builds or pushes the knowledge-api image (retired 2026-04-09)

---

## Deploy Overview

Deployed via `agentopia-infra` Helm chart (`agentopia-base`), ArgoCD-managed. Service runs in `agentopia-dev` and `agentopia-uat` namespaces.

Runtime config is injected via K8s Secrets. See [docs/operations.md](docs/operations.md) for env var reference and health check endpoints.

---

## Evaluation Philosophy

Retrieval quality is measured, not assumed. Every scope change and every retrieval pipeline change is evaluated against a labeled golden question set before promotion.

Current production baseline (dense-only, `text-embedding-3-small`, 1536d):
- nDCG@5 = 0.925
- MRR = 0.96
- P@5 = 0.84
- R@5 = 1.0

### Retrieval feature roadmap

| Item | Status |
|---|---|
| Dense-only search | Production baseline |
| W1 — Markdown-aware chunking | **Accepted (opt-in)** — use `chunking_strategy: "markdown_aware"` in IngestConfig |
| W1.5 — Section path context | **Accepted** — `section_path` field in Citation, populated for MARKDOWN_AWARE chunks |
| W2 — BM25/hybrid retrieval | **Frozen** — conditional reopen only, no implementation |
| W3a — Query expansion | **Not approved** — implemented, default-off, did not clear production gate |
| W3b — HyDE | **Not approved** — implemented, default-off, did not clear production gate |
| W4 — LLM listwise reranking | **Not approved** — implemented, default-off, actively regressed retrieval quality |

See [docs/evaluation.md](docs/evaluation.md) for gate definitions and W-series evidence.

---

## Link Map

| Topic | File |
|---|---|
| Service boundary and interactions | [docs/architecture.md](docs/architecture.md) |
| Extraction from monorepo | [docs/migration.md](docs/migration.md) |
| Retrieval quality evaluation | [docs/evaluation.md](docs/evaluation.md) |
| Runtime config and operations | [docs/operations.md](docs/operations.md) |
| Wiki-RAG evolution plan | [docs/architecture.md#planned-evolution](docs/architecture.md#planned-evolution) |
| Public architecture docs | [ai-agentopia/docs/architecture/super-rag-blueprint.md](https://github.com/ai-agentopia/docs/blob/main/architecture/super-rag-blueprint.md) |
