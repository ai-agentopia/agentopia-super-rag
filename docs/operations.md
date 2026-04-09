# Operations

## Runtime Configuration

All configuration is injected via environment variables. No config file is read at runtime.

### Required

| Variable | Description |
|---|---|
| `QDRANT_URL` | Qdrant service URL (e.g. `http://qdrant.agentopia-dev.svc.cluster.local:6333`) |
| `POSTGRES_DSN` | PostgreSQL connection string |
| `EMBEDDING_API_URL` | Embedding API base URL (e.g. `https://openrouter.ai/api/v1/embeddings`) |
| `EMBEDDING_API_KEY` | API key for embedding service |
| `KNOWLEDGE_API_INTERNAL_TOKEN` | Shared secret for `X-Internal-Token` auth (bot-config-api → knowledge-api) |
| `K8S_NAMESPACE` | K8s namespace to read ArgoCD Application CRDs from |

### Optional

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model name |
| `EMBEDDING_BASE_URL` | (same as `EMBEDDING_API_URL` base) | Override if different from embedding URL |
| `BINDING_RECONCILE_INTERVAL_SECONDS` | `300` | How often to reconcile K8s binding cache |
| `LOG_LEVEL` | `INFO` | Structured log level |

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

## Deployment Flow

1. Push to `dev` branch in `agentopia-protocol` (current source)
2. GitHub Actions: fast test gate → Docker build → push `ghcr.io/ai-agentopia/knowledge-api:dev-{sha}`
3. ArgoCD Image Updater detects new `dev-{sha}` tag → updates Helm values in `agentopia-infra`
4. ArgoCD reconciles `agentopia-base` app → rolling deployment in `agentopia-dev`
5. K8s liveness probe verifies `/health` after rollout
6. Binding cache rebuilds on startup (reads K8s CRDs for all bots)

After extraction to this repo, step 1 will be a push to this repo's `dev` branch. Steps 2–6 are unchanged.

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
curl -s https://dev.agentopia.vn/api/v1/knowledge/health  # or via internal ingress

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
