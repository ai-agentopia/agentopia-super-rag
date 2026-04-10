# agentopia-super-rag

Super RAG is a governed retrieval system for production knowledge access.

This repository is the production source for the `knowledge-api` service currently deployed in Agentopia, but the system design is broader than Agentopia itself: scoped retrieval, document lifecycle control, reproducible evaluation, and safe rollout of retrieval changes.

Extraction from `agentopia-protocol` is complete. See [docs/migration.md](docs/migration.md) for the historical record.

---

## What Super RAG Is

Super RAG is not "RAG with more prompts." It is a retrieval system with platform-grade constraints:

- governed scope isolation
- operator-curated corpora
- explicit document lifecycle and provenance
- measured retrieval quality gates
- rollout discipline for retrieval changes

In practical terms, Super RAG answers this question:

> How do you let many runtime agents retrieve from many knowledge domains without turning retrieval into an ungoverned blob of vectors, prompt hacks, and silent regressions?

The answer in this repo is:

- one retrieval service
- one scoped auth model
- one document lifecycle model
- one evaluation framework
- explicit opt-in gates for every non-baseline retrieval enhancement

This repo is one implementation of that model. Today it runs inside Agentopia. The underlying system pattern is reusable anywhere you need governed retrieval rather than ad hoc app-local RAG.

---

## Why It Exists

Traditional RAG implementations usually optimize for one application at a time:

- ingest some documents
- embed them
- search top-k
- hand results to the LLM

That is fine for prototypes. It is weak for multi-tenant systems, multi-bot systems, or any environment where retrieval behavior becomes a platform dependency.

Super RAG exists because production retrieval usually needs stricter guarantees:

- one bot or tenant must not retrieve another bot's corpus
- document replacements must be traceable and atomic
- retrieval changes must be benchmarked before rollout
- experimental techniques must stay dormant until they prove ROI
- operators need health, provenance, and reindex controls

---

## Super RAG vs Traditional RAG vs WikiRAG-Style Systems

The useful comparison is not "which one is smarter." It is "what problem is each one solving."

| Dimension | Traditional RAG | WikiRAG-style systems | Super RAG |
|---|---|---|---|
| Primary goal | Add retrieval to one app quickly | Improve retrieval on structured docs / wiki-like corpora | Run retrieval as a governed system |
| Corpus model | Usually one app corpus, loosely managed | Documentation-heavy, markdown/wiki-first corpora | Multiple scopes / domains with explicit isolation |
| Access control | Often application-level, coarse | Usually secondary concern | First-class: scope isolation enforced in the retrieval layer |
| Document lifecycle | Often overwrite-and-forget | Usually optimized for chunk quality, less for lifecycle | Active / superseded / deleted record model with provenance |
| Retrieval experiments | Frequently enabled ad hoc | Strong focus: chunking, path context, query variants, rerank, HyDE | Allowed, but gated behind evaluation and rollout controls |
| Evaluation discipline | Often sparse or app-local | Usually benchmark-driven on retrieval quality | Required for promotion; regressions block rollout |
| Best fit | Small product feature or prototype | Knowledge-heavy docs search | Multi-tenant / multi-bot / platform retrieval |

### Traditional RAG

Traditional RAG is the baseline pattern:

1. chunk documents
2. embed documents
3. retrieve top-k
4. send chunks to the LLM

It is simple and fast to build. It is also where people usually accumulate:

- unclear access boundaries
- no document provenance
- no controlled rollout for retrieval changes
- no reproducible benchmark history

Traditional RAG is a good starting point. Super RAG is what you build when retrieval becomes infrastructure.

### WikiRAG-style systems

WikiRAG-style systems are strong where the corpus is highly structured:

- markdown docs
- wiki pages
- section-heavy technical documentation
- path / heading context

This repo deliberately borrowed that family of ideas for evaluation:

- markdown-aware chunking
- section-path context
- query expansion
- HyDE
- reranking

But Super RAG is not "WikiRAG runtime pasted into production." It is a stricter system:

- WikiRAG-style techniques are inputs into the experimentation roadmap
- only techniques that clear evaluation gates become acceptable runtime options
- failed techniques remain implemented but dormant/default-off

So the relationship is:

- traditional RAG = generic baseline pattern
- WikiRAG-style = technique family for docs-heavy retrieval
- Super RAG = governed retrieval system that can selectively absorb those techniques when they prove value

---

## What This Repository Owns

This repository owns the retrieval plane:

- knowledge ingest, parsing, chunking, embedding, indexing
- scoped semantic search
- document lifecycle management
- bot-scope binding cache
- per-scope auth enforcement
- retrieval quality evaluation

It does **not** own:

- workflow orchestration
- bot identity and deployment
- UI / operator console
- general LLM routing beyond what retrieval requires
- planner / reasoning logic

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

Current deployment context:

- this repo runs as `knowledge-api` inside Agentopia
- `agentopia-protocol` uses it for operator proxying and bot runtime retrieval
- `agentopia-infra` deploys it

That deployment context is current reality. It is not the whole definition of Super RAG.

---

## Core Properties

### 1. Scoped retrieval

Every query executes inside explicitly allowed scopes. A caller cannot expand scope by changing query wording.

### 2. Governed ingest lifecycle

Documents are not just "present" or "missing." They move through a lifecycle:

- active
- superseded
- deleted

That gives operators provenance, atomic replace semantics, and a real source of truth across restarts.

### 3. Reproducible retrieval evaluation

Retrieval changes are evaluated against labeled question sets. A technique does not become production-worthy because it sounds good in theory.

### 4. Experimental features stay default-off

This repo keeps unsuccessful retrieval techniques dormant rather than pretending they worked:

- W3a query expansion: implemented, evaluated, not approved
- W3b HyDE: implemented, evaluated, not approved
- W4 reranking: implemented, evaluated, not approved

That is a feature, not a failure. It keeps the runtime honest.

---

## Current System Status

### Accepted

- Dense-only vector retrieval: current production baseline
- W1 markdown-aware chunking: accepted as opt-in for documentation-heavy scopes
- W1.5 section/path-aware retrieval context: accepted

### Implemented but not approved

- W3a query expansion
- W3b HyDE
- W4 listwise reranking

### Frozen

- W2 hybrid sparse+dense retrieval

See [docs/evaluation.md](docs/evaluation.md) for the evidence behind each decision.

---

## Benchmark and Evaluation Summary

Retrieval quality in this repo is measured against labeled golden question sets. That is the core difference between "interesting retrieval code" and a real retrieval system.

### Current production baseline

Dense-only retrieval, `text-embedding-3-small`, top-5 results:

| Metric | Value |
|---|---|
| nDCG@5 | 0.925 |
| MRR | 0.96 |
| P@5 | 0.84 |
| R@5 | 1.0 |

### Accepted retrieval improvement

#### W1 markdown-aware chunking

Real pilot gate on `joblogic-kb/api-docs`:

| Metric | Fixed size | Markdown aware | Delta |
|---|---|---|---|
| nDCG@5 | 0.7440 | 0.7917 | +0.0477 |
| MRR | 0.7500 | 0.7667 | +0.0167 |
| P@5 | 0.4200 | 0.4400 | +0.0200 |

Outcome:

- passed the real pilot gate
- safe for opt-in on documentation-heavy scopes
- default remains `fixed_size`

### Evaluated and rejected retrieval improvements

#### W3a query expansion

Live evaluation did not clear the gate:

| Model | nDCG@5 delta vs baseline | Result |
|---|---|---|
| `openai/gpt-4o-mini` | -0.0050 | fail |
| `openai/gpt-4.1-mini` | +0.0050 | fail |

Interpretation:

- paraphrase-style expansion did not add enough retrieval signal on the real pilot corpus
- cost/latency overhead was not justified by the gain

#### W3b HyDE

| Metric | Baseline | HyDE | Delta |
|---|---|---|---|
| nDCG@5 | 0.9201 | 0.9175 | -0.0026 |
| MRR | 0.9000 | 0.9000 | +0.0000 |
| Avg latency | 577ms | 3313ms | +2736ms |

Interpretation:

- HyDE increased latency materially
- retrieval quality did not improve

#### W4 listwise reranking

| Metric | Baseline | Reranked | Delta |
|---|---|---|---|
| nDCG@5 | 0.5262 | 0.4024 | -0.1238 |
| MRR | 0.5000 | 0.3333 | -0.1667 |
| Avg rerank latency | — | 908ms | — |

Interpretation:

- generic LLM reranking actively hurt domain-specific ranking quality on the evaluated corpus

### Benchmark philosophy

This repo treats benchmarks as promotion gates, not marketing numbers.

The rule is:

- if a retrieval change helps, it can become opt-in
- if it fails the gate, it stays dormant/default-off
- if it is frozen, it is not quietly revived without new evidence

Full evidence, scripts, and artifacts:

- [docs/evaluation.md](docs/evaluation.md)
- `evaluation/`
- `evaluation/results/`

---

## Runtime Dependencies

| Dependency | Purpose | Required |
|---|---|---|
| Qdrant | Vector storage and ANN search | Yes |
| PostgreSQL | Document lifecycle store | Yes |
| Redis | Future caching layer | No |
| OpenRouter | Embedding API (`text-embedding-3-small`) | Yes |
| K8s API | Bot-scope binding cache in production | Yes (production) |

K8s API access uses in-cluster service account. In local dev, binding cache falls back gracefully when K8s is unreachable.

---

## Local Dev Workflow

### Prerequisites

- Python >= 3.12
- Docker or Podman
- OpenRouter API key for embedding

### Quick start — full stack via compose

The fastest way to run the complete service locally. `compose.yaml` starts Qdrant, Postgres, and the knowledge-api in the correct order. Postgres schema is applied automatically on first start.

```bash
cp .env.example .env.local
# Set EMBEDDING_API_KEY and optionally KNOWLEDGE_API_INTERNAL_TOKEN

podman compose --env-file .env.local up --build
# or: docker compose --env-file .env.local up --build
```

Smoke checks:

```bash
curl http://localhost:8002/health

curl -H "X-Internal-Token: local-dev-token" \
  http://localhost:8002/internal/health
```

Stop and reset:

```bash
podman compose --env-file .env.local down
podman compose --env-file .env.local down -v
```

Use the manual paths below if you need explicit control over each dependency.

### 1. Native Python

```bash
git clone git@github.com:ai-agentopia/agentopia-super-rag.git
cd agentopia-super-rag

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env.local
set -a && source .env.local && set +a

cd src && uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

Without `QDRANT_URL` / `DATABASE_URL`, the service runs in an in-memory fallback mode useful for feature work, not for full retrieval validation.

### 2. Test gate

```bash
python -m pytest tests/ -m "not integration and not e2e" -x -q
```

### 3. Manual local dependencies

```bash
podman run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

podman run -d --name agentopia-pg \
  -e POSTGRES_DB=agentopia \
  -e POSTGRES_USER=agentopia \
  -e POSTGRES_PASSWORD=agentopia \
  -p 5432:5432 \
  postgres:16

sleep 5
PGPASSWORD=agentopia psql -h localhost -U agentopia -d agentopia \
  -f db/022_document_records.sql \
  -f db/023_source_type.sql
```

### 4. Container build

```bash
docker build -t agentopia-super-rag:local .
podman build -t agentopia-super-rag:local .
```

---

## CI/CD Overview

- Main-only repo: no `dev` or `uat` branches
- Push to `main` -> Fast Gate -> Build Image -> publish `ghcr.io/ai-agentopia/knowledge-api:dev-{sha}`
- ArgoCD Image Updater tracks `dev-{sha}` and deploys to the non-prod Agentopia environment

---

## Deployment

This repo is currently deployed via `agentopia-infra` Helm configuration and runs in Agentopia non-prod environments.

Runtime config and health endpoints:

- [docs/operations.md](docs/operations.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/evaluation.md](docs/evaluation.md)
