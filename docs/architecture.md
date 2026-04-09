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
/internal/health                                GET    — deep health: Qdrant + binding cache
/internal/binding-sync                          POST   — webhook: bot-config-api pushes binding updates
/internal/binding-sync/{bot_name}               GET    — query single bot binding state
/internal/binding-sync/{bot_name}               DELETE — webhook: bot deleted
/health                                         GET    — shallow health: Postgres + Redis
```

**Search scoping note:** `/api/v1/knowledge/search` resolves scope access server-side. Bot callers (bearer token) get scopes from the binding cache. Operator callers (internal token) pass `?scopes=` query parameters explicitly. There is no per-scope search path — scope isolation is enforced inside the handler, not by the URL.

---

## Interactions with Other Services

### agentopia-protocol (bot-config-api)

- **Operator path:** Operator ingests documents via bot-config-api UI/API, which proxies to knowledge-api with `X-Internal-Token`. bot-config-api controls which scopes an operator can write to.
- **Binding lifecycle:** When a bot is created or deleted, bot-config-api calls `/internal/binding-sync` to push the updated scope subscription to the binding cache.
- **No direct Temporal integration:** Temporal workflows in bot-config-api call bot gateway pods, which call knowledge-api directly with bot bearer tokens. Knowledge retrieval is not a Temporal activity.

### agentopia-protocol (gateway pods)

- **Bot runtime path:** Gateway pods call `/api/v1/knowledge/{scope}/search` with a per-bot bearer token. Auth is verified against the K8s Secret `agentopia-gateway-env-{bot_name}`. Scope resolution is server-side (the service looks up which scopes the bot is subscribed to, not the bot).

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

A scope is identified by `{client_id}/{scope_name}` (e.g., `acme-corp/api-docs`). This canonical path is SHA-256 hashed to produce the Qdrant collection name. The hash prevents cross-tenant namespace collisions even if client IDs share prefixes.

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
    │             Circuit breaker: 5 failures → open for 30s
    │
    ├── upsert    → Qdrant collection (scope SHA-256 name)
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

**Dimension validation:** On startup, if the collection already exists, the service validates that the stored vector dimension matches the configured model's dimension. A mismatch is a hard error — it prevents silent quality degradation after a model change.

---

## Planned Evolution

The following are planned retrieval quality improvements, labeled explicitly as future. None are currently shipped.

### Markdown-aware chunking

A chunking strategy that scores break-points by structural element (heading > code fence > paragraph > list > newline) rather than character count. Preserves semantic units for documentation-heavy corpora. Will be added as `ChunkingStrategy.MARKDOWN_AWARE`. Requires evaluation re-run on affected scopes before enabling.

### Sparse index / hybrid retrieval (BM25 + dense)

BM25 sparse encoding alongside dense vectors. Score fusion via RRF. Issue tracked in `agentopia-protocol` as #319. Will require Qdrant sparse vector support and per-scope evaluation.

### Query expansion

LLM-generated alternative phrasings before vector search. Each phrasing is embedded independently; results are merged via RRF. Adds one llm-proxy round-trip per query (estimated +300–800ms latency). Requires latency budget approval and per-scope evaluation before enabling.

### HyDE (Hypothetical Document Embedding)

Embed a generated hypothetical answer rather than the raw query. Improves recall on long-tail questions in documentation corpora. Same latency profile as query expansion. Requires separate golden-question evaluation format.

### LLM cross-encoder reranking

Re-score top-k candidates with a cross-encoder model after initial retrieval. Highest quality lift potential, highest per-query cost. Sequenced after query expansion and HyDE in the roadmap.

All evolution items must pass the evaluation gate (see [evaluation.md](evaluation.md)) before enabling on any production scope.
