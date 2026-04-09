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

### Shallow health (liveness probe)

```
GET /health
→ {"status": "ok", "version": "2.0.0", "checks": {"postgres": {"status": "ok"}, "redis": {"status": "ok"}}}
```

Returns 200 if Postgres is reachable. Used by K8s liveness probe.

**Note:** Redis is checked here but not yet used for any runtime path. Check will pass even if Redis is unreachable in current code.

### Deep health (internal, operator use)

```
GET /internal/health
→ {
    "qdrant": {"status": "ok" | "error", "message": "..."},
    "binding_cache": {
      "status": "ok" | "stale",
      "bot_count": N,
      "last_reconcile": "ISO8601",
      "stale_bots": [...]
    }
  }
```

Returns 200 only if both Qdrant and the binding cache are healthy. Binding cache is considered stale if last reconcile was more than 2× the configured interval ago.

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

1. Check `/internal/health` — look at `binding_cache.stale_bots` and `last_reconcile`
2. Force a reconcile: `POST /internal/binding-sync` from bot-config-api, or restart the pod
3. If K8s API is unreachable, the cache cannot rebuild. Check in-cluster service account RBAC (`list` verb on `applications.argoproj.io` in the configured namespace)

### Qdrant connection errors

1. Check `/internal/health` for Qdrant status
2. Verify `QDRANT_URL` env var points to correct endpoint
3. Check Qdrant pod status: `kubectl get pod -n agentopia-dev -l app=qdrant`
4. Embedding circuit breaker opens after 5 consecutive failures and resets after 30s. Check logs for `circuit_breaker: open` messages

### Dimension mismatch on startup

```
ValueError: Collection {scope} exists with dimension 4096, configured model returns 1536
```

This means the Qdrant collection was created with a different embedding model. Options:
1. Reindex the scope: `POST /api/v1/knowledge/{scope}/reindex` — this drops and rebuilds the collection
2. Change `EMBEDDING_MODEL` back to match the stored dimension (not recommended long-term)

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
