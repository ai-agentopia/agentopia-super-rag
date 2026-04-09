# Architecture

## Service Boundary

`agentopia-super-rag` is a standalone FastAPI service. It owns one well-defined responsibility: **scoped, governed knowledge retrieval for Agentopia bots**.

It does not make workflow decisions, does not call LLMs for reasoning, and does not hold any bot state beyond the scope binding cache (which is derived from K8s CRD annotations maintained by `bot-config-api`).

### What this service exposes

```
/api/v1/knowledge/search                        GET    — scoped semantic search (scopes via query param)
/api/v1/knowledge/scopes                        GET    — list scopes accessible to caller
/api/v1/knowledge/stale                         GET    — scopes with outdated embeddings
/api/v1/knowledge/{scope}                       GET    — scope metadata
/api/v1/knowledge/{scope}                       DELETE — remove scope and all docs
/api/v1/knowledge/{scope}/ingest                POST   — operator-initiated doc ingest
/api/v1/knowledge/{scope}/reindex               POST   — re-embed all docs in scope
/api/v1/knowledge/{scope}/documents             GET    — list documents in scope
/api/v1/knowledge/{scope}/documents/{source}    DELETE — remove specific document by source path
/internal/health                                GET    — internal health: Qdrant + binding cache + proxy config
/internal/binding-sync                          POST   — webhook: bot-config-api pushes binding updates
/internal/binding-sync/{bot_name}               GET    — query single bot binding state
/internal/binding-sync/{bot_name}               DELETE — webhook: bot deleted
/health                                         GET    — liveness probe (no dependency checks)
```

**Search scoping note:** `/api/v1/knowledge/search` resolves scope access server-side. Bot callers (bearer token) get scopes from the binding cache. Operator callers (internal token) pass `?scopes=` query parameters explicitly. There is no per-scope search path — scope isolation is enforced inside the handler, not by the URL.

---

## Interactions with Other Services

### agentopia-protocol (bot-config-api)

- **Operator path:** Operator ingests documents via bot-config-api UI/API, which proxies to knowledge-api with `X-Internal-Token`. bot-config-api controls which scopes an operator can write to.
- **Binding lifecycle:** When a bot is created or deleted, bot-config-api calls `/internal/binding-sync` to push the updated scope subscription to the binding cache.
- **No direct Temporal integration:** Temporal workflows in bot-config-api call bot gateway pods, which call knowledge-api directly with bot bearer tokens. Knowledge retrieval is not a Temporal activity.

### agentopia-protocol (gateway pods)

- **Bot runtime path:** Gateway pods call `GET /api/v1/knowledge/search` with a per-bot bearer token (`Authorization: Bearer {token}` + `X-Bot-Name: {name}`). There is no per-scope search URL — the bot does not specify a scope in the path. Scope resolution is server-side: the service looks up which scopes the bot is subscribed to via the binding cache, and searches only those scopes. Auth is verified against the K8s Secret `agentopia-gateway-env-{bot_name}`.

### agentopia-infra

- Deployment manifests live in `agentopia-infra/charts/agentopia-base/`. The service is deployed as part of the base platform Helm chart, not as a standalone ArgoCD Application.
- Image tag is tracked by ArgoCD Image Updater (pattern: `dev-{sha}` on `dev` image builds).

### agentopia-graph-executor

- No direct interaction currently. In the planned evolution (see below), `graph-executor` may call knowledge-api during planner graph execution to provide domain context for planning. This is not yet implemented.

---

## Auth and Scope Model

### Dual-path auth

| Path | Header | Verified against | Scope resolution |
|---|---|---|---|
| Operator / bot-config-api | `X-Internal-Token` | `KNOWLEDGE_API_INTERNAL_TOKEN` env | Explicit `client_id/scope_name` in request |
| Bot gateway | `Authorization: Bearer {token}` + `X-Bot-Name: {name}` | K8s Secret `agentopia-gateway-env-{bot_name}` | Server-side binding cache lookup |

### Scope identity

A scope is identified by `{client_id}/{scope_name}` (e.g., `acme-corp/api-docs`). This canonical path is SHA-256 hashed (truncated to 16 hex chars) to produce the Qdrant collection name: `kb-{sha256_hex[:16]}`. The hash prevents cross-tenant namespace collisions and avoids Qdrant's restriction on `/` in collection names. The canonical form with `/` is preserved in all API responses, logs, and metadata — only Qdrant sees the physical hashed name.

### Binding cache

On startup, the service rebuilds a full binding cache by reading all ArgoCD Application CRDs in the configured K8s namespace. It reconciles every 300 seconds and also accepts push updates from bot-config-api via `/internal/binding-sync`. On cache miss, it falls back to a direct K8s API call before returning 403.

A bot can only search scopes present in its binding. Scope access cannot be elevated by the bot at query time.

---

## Retrieval Pipeline (Current Production Baseline)

**Dense-only vector search.** Hybrid retrieval (BM25 + dense) is planned but not yet shipped.

### Ingest path

```
document input (PDF / HTML / Markdown / code)
    │
    ├── parse     (pdf → text, html → cleaned text, md → text, code → text)
    │
    ├── chunk     (fixed-size 512 tokens / 50 overlap, or paragraph, or code-aware)
    │             Strategy is per-ingest configurable. Default: fixed-size.
    │
    ├── embed     (POST to EMBEDDING_API_URL, model: text-embedding-3-small, 1536d)
    │             Circuit breaker: 5 consecutive failures → open for 300s (5 min)
    │
    ├── upsert    → Qdrant collection (kb-{sha256_hex[:16]} of scope identity)
    │
    └── record    → PostgreSQL document_store (active state, source hash, version)
```

**Document update** uses two-phase atomic replace: new chunks are upserted with a new version tag, then old chunks are deleted. A partial failure leaves old chunks in place — no partial state is visible to callers.

### Search path

```
query string
    │
    ├── embed     (same model, same API)
    │
    ├── vector search → Qdrant (top-k by cosine similarity, k=5 default)
    │
    ├── build citations (source, page, chunk offset, score)
    │
    └── return    SearchResult[] with citations
```

**Dimension validation:** On startup, the service checks all existing `kb-*` Qdrant collections against the configured embedding dimension. A mismatch logs a `DIMENSION_MISMATCH` warning — it is **not** a hard startup error. The service continues to run, but search results on the mismatched collection will be degraded or incorrect. Operator action required: `POST /api/v1/knowledge/{scope}/reindex` to rebuild the collection with the correct dimension.

---

## Planned Evolution

The following are planned retrieval quality improvements, labeled explicitly as future. None are currently shipped.

### Markdown-aware chunking (W1 — shipped, opt-in)

`ChunkingStrategy.MARKDOWN_AWARE` — splits on structural boundaries: heading > code fence > paragraph > list > newline. Headings always create strong chunk boundaries (never merged across). Code fences are preserved intact. Falls back to fixed-size splitting for content without markdown structure or for oversized blocks.

**Status:** Implemented. Opt-in only — must be explicitly requested via `chunking_strategy: "markdown_aware"` in IngestConfig. Default remains `fixed_size`.

**Evaluation:** Pilot comparison completed on a synthetic 3-doc corpus (see `evaluation/results/w1_chunking_comparison.json`). Full promotion gate (Phase 1b golden question set on a production pilot scope) has not been run yet. Do not enable on production scopes until the full gate is executed per `docs/evaluation.md` promotion rules.

### Sparse index / hybrid retrieval (BM25 + dense)

BM25 sparse encoding alongside dense vectors. Score fusion via RRF. Issue tracked in `agentopia-protocol` as #319. Will require Qdrant sparse vector support and per-scope evaluation.

### Query expansion

LLM-generated alternative phrasings before vector search. Each phrasing is embedded independently; results are merged via RRF. Adds one llm-proxy round-trip per query (estimated +300–800ms latency). Requires latency budget approval and per-scope evaluation before enabling.

### HyDE (Hypothetical Document Embedding)

Embed a generated hypothetical answer rather than the raw query. Improves recall on long-tail questions in documentation corpora. Same latency profile as query expansion. Requires separate golden-question evaluation format.

### LLM cross-encoder reranking

Re-score top-k candidates with a cross-encoder model after initial retrieval. Highest quality lift potential, highest per-query cost. Sequenced after query expansion and HyDE in the roadmap.

All evolution items must pass the evaluation gate (see [evaluation.md](evaluation.md)) before enabling on any production scope.
