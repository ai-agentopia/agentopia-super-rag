# Operations

## Retrieval Debugger

Operators can inspect ranked retrieval results with full chunk metadata:

```
GET /api/v1/knowledge/debug/query?scope={scope}&q={query}&limit={N}
```

Auth: `X-Internal-Token` (operator only — not accessible to bots).

Returns: rank, score, source, section_path, section, page, chunk_index, document_hash, ingested_at, text.

Does not apply experimental features (W3a/W3b/W4). Always runs dense-only baseline retrieval.

---

## Super RAG Integration Boundary

Super RAG's `/ingest` endpoint is called by the Ingest Orchestrator in `agentopia-knowledge-ingest` after normalization and extraction complete. Super RAG does not own upstream document parsing, metadata extraction, or raw document storage.

For the full ingest contract, see [architecture.md](architecture.md#super-rag-integration-boundary).

---

## Runtime Configuration

All configuration is injected via environment variables. No config file is read at runtime.

### Required

| Variable | Description |
|---|---|
| `QDRANT_URL` | Qdrant service URL (e.g. `http://qdrant.agentopia-dev.svc.cluster.local:6333`). Unset → in-memory vector fallback (no persistence). |
| `DATABASE_URL` | PostgreSQL connection string (e.g. `postgresql://user:pass@host:5432/agentopia`). Unset → `InMemoryDocumentStore` (no document lifecycle persistence). |
| `EMBEDDING_API_KEY` | API key for OpenRouter (embedding and LLM calls) |
| `KNOWLEDGE_API_INTERNAL_TOKEN` | Shared secret for `X-Internal-Token` auth (bot-config-api → knowledge-api) |
| `K8S_NAMESPACE` | K8s namespace to read ArgoCD Application CRDs from (binding cache) |

### Optional

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_BASE_URL` | `https://openrouter.ai/api/v1/embeddings` | OpenRouter API base URL. Used for embedding calls and (when `LLM_PROXY_URL` unset) for W3a/W3b/W4 LLM calls. |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model name. In production via agentopia-llm-proxy/OpenRouter, must include provider prefix: `openrouter/openai/text-embedding-3-small`. |
| `BINDING_RECONCILE_INTERVAL_SECS` | `300` | How often to reconcile K8s binding cache (seconds) |
| `LOG_LEVEL` | `INFO` | Structured log level |
| `LOG_FORMAT` | `text` | Set to `json` for structured JSON log output |
| `PORT` | `8002` | Container port for uvicorn |

### Optional — dormant retrieval features (all default-off)

These env vars activate W-series retrieval features that are **implemented but not approved for production**. All are no-ops unless a scope is explicitly added to the allowlist. See `docs/evaluation.md` for gate status.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROXY_URL` | `""` | Base URL for LLM proxy (used by W3a, W3b, W4). Falls back to OpenRouter if unset. |
| `QUERY_EXPANSION_SCOPES` | `""` | Comma-separated scope names allowed to use W3a query expansion |
| `HYDE_SCOPES` | `""` | Comma-separated scope names allowed to use W3b HyDE retrieval |
| `RERANK_SCOPES` | `""` | Comma-separated scope names allowed to use W4 LLM listwise reranking |
| `RERANK_MODEL` | `openai/gpt-4o-mini` | LLM model for W4 reranking (without provider prefix) |
| `RERANK_TIMEOUT_MS` | `5000` | Timeout for W4 reranking LLM call (milliseconds) |
| `RERANK_CANDIDATE_K` | `20` | Candidate pool size for W4 reranking (retrieved before LLM reorders) |

---

## Health Checks

### Liveness probe

```
GET /health
→ {"status": "ok", "service": "knowledge-api", "version": "1.0.0"}
```

Returns 200 unconditionally (no dependency checks). Pure liveness probe — indicates the process is alive. Used by K8s liveness probe. Does **not** check Postgres, Qdrant, Redis, or any upstream dependency.

### Internal health (operator use)

```
GET /internal/health
→ {
    "service": "knowledge-api",
    "status": "ok" | "degraded",
    "binding_cache": {
      "bot_count": N,
      "last_reconcile": 1234567890.12,   ← Unix timestamp (float), not ISO8601
      "reconcile_interval_secs": 300,
      "bots": ["bot-a", "bot-b", ...]
    },
    "qdrant": "ok" | "error: <message>",  ← flat string, not nested object
    "proxy_mode": {
      "knowledge_api_url": "",
      "internal_token_configured": true
    }
  }
```

Returns 200 always (check `status` field for `"degraded"`). Status is `"degraded"` if Qdrant health check throws. Binding cache staleness is not checked in this response — use `bot_count` and `last_reconcile` to assess cache state manually.

This endpoint requires `X-Internal-Token`.

---

## Database Schema

Postgres requires two migrations applied in order:

```bash
psql -d <database> -f db/022_document_records.sql   # document lifecycle table
psql -d agentopia -f db/023_source_type.sql         # source_type column
```

These are idempotent (`CREATE TABLE IF NOT EXISTS`, `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`).

---

## Deployment Flow

1. Push to `main` branch in `agentopia-super-rag` (this repo — main-only, no dev/uat branches)
2. GitHub Actions: fast test gate → Docker build → push `ghcr.io/ai-agentopia/knowledge-api:dev-{sha}`
3. ArgoCD Image Updater detects new `dev-{sha}` tag → updates Helm values in `agentopia-infra`
4. ArgoCD reconciles `agentopia-base` app → rolling deployment in `agentopia-dev`
5. K8s liveness probe verifies `/health` after rollout
6. Binding cache rebuilds on startup (reads K8s CRDs for all bots)

---

## Troubleshooting

### Binding cache is stale / bots returning 403

1. Check `/internal/health` — look at `binding_cache.bot_count` and `binding_cache.last_reconcile` (Unix timestamp). If `last_reconcile` is stale, the cache has not rebuilt recently.
2. Force a reconcile: `POST /internal/binding-sync` from bot-config-api, or restart the pod
3. If K8s API is unreachable, the cache cannot rebuild. Check in-cluster service account RBAC (`list` verb on `applications.argoproj.io` in the configured namespace)

### Qdrant connection errors

1. Check `/internal/health` — look at the `qdrant` field (`"ok"` or `"error: <message>"`)
2. Verify `QDRANT_URL` env var points to correct endpoint
3. Check Qdrant pod status: `kubectl get pod -n agentopia-dev -l app=qdrant`
4. Embedding circuit breaker opens after 5 consecutive failures and resets after 300s (5 minutes). Check logs for `EmbeddingCircuitBreaker: OPEN` messages.

### Dimension mismatch (DIMENSION_MISMATCH warning in logs)

```
DIMENSION_MISMATCH: collection 'kb-<hash>' has dim=4096, config expects dim=1536. Reindex required.
```

This is a **warning, not an error**. The service continues to run. Search results on the mismatched collection will be incorrect. Reindex is required:

1. Reindex the scope: `POST /api/v1/knowledge/{scope}/reindex` — drops and rebuilds the Qdrant collection with the correct dimension
2. Do not change `EMBEDDING_MODEL` to match the old dimension — that masks the problem

### Ingest silently succeeds but search returns no results

1. Check that the ingest returned a document hash (not a parsing error)
2. Check Qdrant directly: the collection should have points after ingest
3. Verify the bot's scope subscription includes the ingested scope (`GET /api/v1/knowledge/scopes`)
4. Check embedding API is reachable — a circuit-breaker open state will cause ingest to skip embedding

---

## Operator-Safe Validation Steps

After a deployment, an operator can verify the service is functioning without risking data:

```bash
# 1. Shallow health
curl -s https://<your-ingress-host>/health  # Agentopia production: https://dev.agentopia.vn/health

# 2. List scopes (uses internal token)
curl -s -H "X-Internal-Token: $KNOWLEDGE_API_INTERNAL_TOKEN" \
  http://knowledge-api.agentopia-dev.svc.cluster.local:8080/api/v1/knowledge/scopes

# 3. Deep health (check Qdrant + binding cache)
curl -s -H "X-Internal-Token: $KNOWLEDGE_API_INTERNAL_TOKEN" \
  http://knowledge-api.agentopia-dev.svc.cluster.local:8080/internal/health | python3 -m json.tool

# 4. Check for stale scopes (embeddings out of date)
curl -s -H "X-Internal-Token: $KNOWLEDGE_API_INTERNAL_TOKEN" \
  http://knowledge-api.agentopia-dev.svc.cluster.local:8080/api/v1/knowledge/stale
```

None of the above modify any data. They are read-only except for `/internal/binding-sync`.
