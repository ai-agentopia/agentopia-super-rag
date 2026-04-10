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

### Prerequisites

- Python ≥ 3.12
- Docker or Podman (for container workflows)
- Qdrant and Postgres (for full ingest + search — see [Local Dependencies](#local-dependencies) below)
- OpenRouter API key (for embedding — `text-embedding-3-small` via OpenRouter)

### 1. Native Python

```bash
git clone git@github.com:ai-agentopia/agentopia-super-rag.git
cd agentopia-super-rag

# Create venv and install all dependencies (includes python-multipart, psycopg, etc.)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set up env vars
cp .env.example .env.local
# edit .env.local with your OpenRouter key, token, and local service URLs
set -a && source .env.local && set +a

# Run the service
cd src && uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

Smoke check:
```bash
curl http://localhost:8002/health
# → {"status":"ok","service":"knowledge-api","version":"1.0.0"}
```

Without Qdrant/Postgres configured, the service starts in in-memory fallback mode (no persistent search or document lifecycle — useful for feature development only).

### 2. Test gate

```bash
# Fast gate — no external dependencies required (runs from repo root)
python -m pytest tests/ -m "not integration and not e2e" -x -q
# → 421 passed, 23 skipped (validated)
```

The `integration` and `e2e` markers are defined for future use. No tests are currently marked with them — the fast gate is the complete test gate.

### 3. Local Dependencies

Start Qdrant and Postgres locally with Podman (or Docker, substituting `podman` → `docker`):

```bash
# Qdrant vector store
podman run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# Postgres document store
podman run -d --name agentopia-pg \
  -e POSTGRES_DB=agentopia \
  -e POSTGRES_USER=agentopia \
  -e POSTGRES_PASSWORD=agentopia \
  -p 5432:5432 \
  postgres:16

# Apply schema migrations (wait ~5s for Postgres to initialize)
sleep 5
PGPASSWORD=agentopia psql -h localhost -U agentopia -d agentopia \
  -f db/022_document_records.sql \
  -f db/023_source_type.sql
```

### 4. Container paths

**Build:**

```bash
# Docker
docker build -t agentopia-super-rag:local .

# Podman
podman build -t agentopia-super-rag:local .
```

**Podman smoke run (in-memory mode — no Qdrant/Postgres needed):**

```bash
podman run --rm -p 8002:8002 \
  -e KNOWLEDGE_API_INTERNAL_TOKEN=local-test \
  agentopia-super-rag:local

curl http://localhost:8002/health
# → {"status":"ok","service":"knowledge-api","version":"1.0.0"}
```

**Podman full stack (all three services in a shared pod):**

```bash
podman pod create --name agentopia-local -p 8002:8002 -p 6333:6333 -p 5432:5432

podman run -d --pod agentopia-local --name qdrant qdrant/qdrant:latest

podman run -d --pod agentopia-local --name agentopia-pg \
  -e POSTGRES_DB=agentopia \
  -e POSTGRES_USER=agentopia \
  -e POSTGRES_PASSWORD=agentopia \
  postgres:16

sleep 5
PGPASSWORD=agentopia psql -h localhost -U agentopia -d agentopia \
  -f db/022_document_records.sql \
  -f db/023_source_type.sql

podman run -d --pod agentopia-local --name agentopia-rag \
  --env-file .env.local \
  agentopia-super-rag:local
```

Within a pod all containers share the same network namespace, so `QDRANT_URL=http://localhost:6333` and `DATABASE_URL=postgresql://...@localhost:5432/agentopia` work correctly on both Linux and macOS.

Smoke checks after startup:
```bash
curl http://localhost:8002/health
# → {"status":"ok","service":"knowledge-api","version":"1.0.0"}

curl -H "X-Internal-Token: local-dev-token" http://localhost:8002/internal/health
# → {"status":"ok","qdrant":"ok",...}

curl -H "X-Internal-Token: local-dev-token" http://localhost:8002/api/v1/knowledge/scopes
# → {"scopes":[],"count":0}
```

---

## CI/CD Overview

- **Main-only repo.** CI triggers on push to `main` only (`.github/workflows/ci.yml`, `.github/workflows/build-image.yml`). No `dev` or `uat` branches.
- Push to `main` → fast gate → Docker build → push `ghcr.io/ai-agentopia/knowledge-api:dev-{sha}` (tag format is `dev-{sha}`, not a branch name)
- ArgoCD Image Updater picks up new `dev-{sha}` tags and deploys to `agentopia-dev` namespace
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
