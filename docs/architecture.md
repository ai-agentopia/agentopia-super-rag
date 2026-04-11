# Architecture

## Full System Overview

The Agentopia knowledge system spans two repos and three planes:

- **Ingest plane** (`agentopia-knowledge-ingest`): source connectors, raw document store, normalizer, extractor, orchestrator, operator UI
- **Retrieval plane** (`agentopia-super-rag`): chunking, embedding, Qdrant indexing, scoped retrieval, evaluation
- **Operator control plane** (shared): upload API, job status, document management, quality dashboard, retrieval debugger

```
Sources (S3, upload API, connectors)
    |
    v
Raw Document Store (S3, versioned immutable)
    |---> Document Registry (PostgreSQL: scope, version, state, provenance)
    |
    v [Async Queue — phase 2; day-1: synchronous]
Normalizer Service (PDF/DOCX/HTML/MD -> text + structure hints)
    |
    v [Async Queue — phase 2; day-1: synchronous]
Metadata Extractor Service (title, author, date, section hierarchy)
    |
    v [Sync boundary: Ingest Orchestrator]
Super RAG /ingest API
    |
    v
Super RAG Indexing (chunk -> embed -> Qdrant + PostgreSQL lifecycle)
    |
    v
Runtime Retrieval (scope-filtered, only active documents)
    |
    v
Evaluation (per-scope baselines, regression gates, quality trends)

Operator Control Plane:
    Upload UI -> Job Status -> Document Management -> Quality Dashboard -> Retrieval Debugger
```

### Service Map

| Service | Repo | Owns |
|---|---|---|
| Document Ingest Service | knowledge-ingest | Upload API, job tracking, document registry (PostgreSQL), raw S3 store |
| Normalizer Service | knowledge-ingest | Format parsing: PDF/DOCX/HTML/MD → text + structure hints; S3 normalized artifact |
| Metadata Extractor Service | knowledge-ingest | Structural extraction: title, author, date, hierarchy; S3 extracted artifact |
| Ingest Orchestrator | knowledge-ingest | Coordination: validate → call Super RAG → confirm → mark active |
| Operator UI | knowledge-ingest | Upload, job status, document management, scope management, quality dashboard |
| Super RAG | agentopia-super-rag | Chunking, embedding, Qdrant, scoped retrieval, evaluation framework |

### Source of Truth Per Layer

| Layer | Source of Truth | Storage | Mutable? |
|---|---|---|---|
| Raw document | Original file | S3 `original.{ext}` | Immutable per version |
| Document metadata + state | Document registry | PostgreSQL `documents` (knowledge-ingest) | Mutable (state evolves) |
| Normalized text | Parsed artifact | S3 `normalized.json` | Immutable per version |
| Extracted metadata | Structural artifact | S3 `extracted.json` | Immutable per version |
| Chunk vectors | Embeddings | Qdrant per-scope collection | Immutable per version |
| Document visibility state | active/superseded/deleted | Super RAG PostgreSQL `document_records` | Mutable |
| Scope-to-bot binding | Bot authorization | K8s CRD annotations | Changed via K8s only |
| Per-scope evaluation baselines | Quality anchor | PostgreSQL `evaluation_baselines` (super-rag) | Human-curated, append-only |
| Evaluation history | Quality audit | PostgreSQL `evaluation_results` (super-rag) | Append-only |

---

## Super RAG Integration Boundary

### What Super RAG owns

- Chunking documents into retrieval-sized segments
- Embedding chunks via OpenRouter (`text-embedding-3-small`, 1536d)
- Storing chunk vectors and payload metadata in Qdrant (one collection per scope)
- Maintaining document lifecycle state in PostgreSQL (`document_records` table): active / superseded / deleted
- Enforcing scope isolation at query time: `scope IN [resolved_scopes] AND status = "active"`
- Resolving bot-to-scope bindings via K8s CRD binding cache
- Evaluation framework: golden question sets, per-scope baselines, regression gates

### What Super RAG does NOT own

- Format parsing (PDF → text, DOCX → text, HTML → text)
- Metadata extraction (title, author, date, section hierarchy)
- Raw document storage (S3 originals)
- Upstream document registry (document_id, version, scope assignment)
- Compiled or curated knowledge representations
- Source connectors (S3 polling, GitHub, external APIs)

### /ingest API Contract

The Document Ingest Orchestrator calls Super RAG's ingest endpoint after normalization and extraction are complete.

**Endpoint:** `POST /api/v1/knowledge/{scope}/ingest`

**Authentication:** `X-Internal-Token` header (shared secret between Orchestrator and Super RAG)

**Request body:**

```json
{
  "document_id": "uuid",
  "version":     1,
  "text":        "full normalized document text",
  "metadata": {
    "title":        "API Reference",
    "section_path": ["Authentication", "Token Format"],
    "hierarchy": [
      {
        "level":    1,
        "heading":  "Authentication",
        "children": [
          { "level": 2, "heading": "Token Format", "children": [] }
        ]
      }
    ],
    "author":   "Engineering Team",
    "date":     "2026-01-15",
    "language": "en",
    "format":   "pdf"
  },
  "chunking_strategy": "markdown_aware"
}
```

**Fields:**

| Field | Required | Description |
|---|---|---|
| `document_id` | Yes | Stable UUID from the upstream document registry |
| `version` | Yes | Version number; used to tag all Qdrant chunks for this ingest |
| `text` | Yes | Full normalized plain text from `normalized.json` |
| `metadata.title` | No | From `extracted.json`; empty string if not extracted |
| `metadata.section_path` | No | Ordered heading path to the document's root context; used per-chunk |
| `metadata.hierarchy` | No | Full section tree from `extracted.json`; used to populate `section_path` per chunk |
| `metadata.author` | No | From `extracted.json` |
| `metadata.date` | No | From `extracted.json`; ISO date string |
| `metadata.language` | No | ISO 639-1 code; defaults to `"en"` |
| `metadata.format` | No | Source format: `pdf`, `docx`, `html`, `markdown`, `txt` |
| `chunking_strategy` | No | `fixed_size` (default) or `markdown_aware` (W1, opt-in) |

**Response:**

```json
{
  "document_id": "uuid",
  "scope":       "joblogic-kb/api-docs",
  "version":     1,
  "chunk_count": 42,
  "status":      "indexed"
}
```

The Orchestrator waits for this response before marking the document as `active` in the upstream registry. The sync boundary is intentional: atomicity of "indexed in Qdrant" and "visible to retrieval" is guaranteed.

**Idempotency:** If `(document_id, version)` already exists in Qdrant with `status = "active"`, the call is a no-op and returns the existing chunk count. Safe to retry.

---

## Super RAG Service Boundary (Retrieval Plane)

`agentopia-super-rag` is a standalone FastAPI service. It owns one well-defined responsibility: **scoped, governed knowledge retrieval for Agentopia bots**.

It does not make workflow decisions, does not call LLMs for reasoning, and does not hold any bot state beyond the scope binding cache (which is derived from K8s CRD annotations maintained by `bot-config-api`).

### Endpoints

```
/api/v1/knowledge/search                        GET    — scoped semantic search
/api/v1/knowledge/scopes                        GET    — list scopes accessible to caller
/api/v1/knowledge/stale                         GET    — scopes with outdated embeddings
/api/v1/knowledge/{scope}                       GET    — scope metadata
/api/v1/knowledge/{scope}                       DELETE — remove scope and all docs
/api/v1/knowledge/{scope}/ingest                POST   — ingest document (called by Orchestrator)
/api/v1/knowledge/{scope}/reindex               POST   — re-embed all docs in scope
/api/v1/knowledge/{scope}/documents             GET    — list documents in scope
/api/v1/knowledge/{scope}/documents/{source}    DELETE — remove specific document
/internal/health                                GET    — internal health: Qdrant + binding cache
/internal/binding-sync                          POST   — webhook: bot-config-api pushes binding updates
/internal/binding-sync/{bot_name}               GET    — query single bot binding state
/internal/binding-sync/{bot_name}               DELETE — webhook: bot deleted
/health                                         GET    — liveness probe
```

---

## Auth and Scope Model

### Dual-path auth

| Path | Header | Verified against | Scope resolution |
|---|---|---|---|
| Operator / Orchestrator | `X-Internal-Token` | `KNOWLEDGE_API_INTERNAL_TOKEN` env | Explicit `{scope}` in URL path |
| Bot gateway | `Authorization: Bearer {token}` + `X-Bot-Name: {name}` | K8s Secret `agentopia-gateway-env-{bot_name}` | Server-side binding cache lookup |

### Scope identity

A scope is identified by `{client_id}/{scope_name}` (e.g., `joblogic-kb/api-docs`). This canonical path is SHA-256 hashed (truncated to 16 hex chars) to produce the Qdrant collection name: `kb-{sha256_hex[:16]}`. The hash prevents cross-tenant namespace collisions and avoids Qdrant's restriction on `/` in collection names. The canonical form is preserved in all API responses, logs, and metadata.

### Binding cache

On startup, the service rebuilds a full binding cache by reading all ArgoCD Application CRDs in the configured K8s namespace. It reconciles every `BINDING_RECONCILE_INTERVAL_SECS` seconds (default 300) and accepts push updates via `/internal/binding-sync`. On cache miss, it falls back to a direct K8s API call before returning 403.

A bot can only search scopes present in its binding. Scope access cannot be elevated by the bot at query time.

---

## Retrieval Pipeline

### Production baseline

Dense-only vector search, `text-embedding-3-small`, 1536d, top-5.

```
query string
    |
    v
embed (POST EMBEDDING_BASE_URL, model: text-embedding-3-small)
    |
    v
vector search -> Qdrant (filter: scope IN [resolved] AND status = "active", top-k=5)
    |
    v
build citations (source, section_path, score)
    |
    v
return SearchResult[] with citations
```

### Ingest path

```
/ingest API call (from Orchestrator)
    |
    v
chunk (fixed_size 512t/50 overlap, or markdown_aware)
    |
    v
embed (POST EMBEDDING_BASE_URL)
    |
    v
upsert -> Qdrant (collection: kb-{scope_hash}, payload: document_id, scope, version, section_path, status=active)
    |
    v
update document_records (PostgreSQL: status=active, chunk_count)
    |
    v
supersede prior version chunks (Qdrant payload update: status=superseded)
```

Document update is atomic: new chunks are confirmed before prior version is superseded. A partial failure leaves the prior version active.

---

## Retrieval Quality Improvements

### Accepted (production-eligible)

**W1 — Markdown-aware chunking (opt-in)**
Splits on heading > code fence > paragraph > list boundaries. Opt-in via `chunking_strategy: "markdown_aware"` in ingest call. Default remains `fixed_size`.
Evaluation: nDCG@5 +0.0477 on `joblogic-kb/api-docs` pilot. Gate: PASSED.

**W1.5 — Section/path-aware retrieval context**
Each chunk carries `section_path` (heading hierarchy from document root to chunk). Populated automatically for `markdown_aware` chunks. Additive field; backward compatible.

### Implemented, not approved

**W3a — Query expansion:** Disabled by default. Per-scope opt-in via `QUERY_EXPANSION_SCOPES`. Live evaluation did not clear the +0.02 gate on real pilot corpus. Dormant.

**W3b — HyDE:** Disabled by default. Per-scope opt-in via `HYDE_SCOPES`. Live evaluation: nDCG@5 -0.0026, latency +2736ms. Not approved.

**W4 — LLM listwise reranking:** Disabled by default. Per-scope opt-in via `RERANK_SCOPES`. Live evaluation: nDCG@5 -0.1238. Not approved.

### Frozen

**W2 — Hybrid sparse+dense retrieval:** Frozen. Tracked in [#15](https://github.com/ai-agentopia/agentopia-super-rag/issues/15). Reopen requires new evaluation evidence showing ≥3% nDCG@5 improvement.

See [evaluation.md](evaluation.md) for full evidence behind each decision.

---

## Interactions with Other Services

### agentopia-knowledge-ingest (new)

- Ingest Orchestrator calls `POST /api/v1/knowledge/{scope}/ingest` with normalized text and extracted metadata
- Authentication: `X-Internal-Token`
- Orchestrator waits for response before marking document `active` in its own registry

### agentopia-protocol (bot-config-api)

- **Operator path:** bot-config-api proxies operator ingest calls to Super RAG with `X-Internal-Token`
- **Binding lifecycle:** bot-config-api calls `/internal/binding-sync` on bot create/delete to push scope binding updates
- **No direct Temporal integration:** Temporal workflows call bot gateway pods, which call Super RAG with bot bearer tokens

### agentopia-protocol (gateway pods)

- Bot runtime: gateway pods call `GET /api/v1/knowledge/search` with per-bot bearer token
- Scope resolution is server-side; bots do not specify scopes in the request

### agentopia-infra

- Deployment manifests in `agentopia-infra/charts/agentopia-base/`
- Image tag tracked by ArgoCD Image Updater: `dev-{sha}`
