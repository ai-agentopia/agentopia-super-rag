# E2E Integration Testing

## Real-System Validation (primary)

The primary E2E path runs against the actual Agentopia dev cluster, testing the real runtime: bot auth, binding cache, scope resolution, and Qdrant search.

### What is proven

| Test | What it exercises |
|---|---|
| Service liveness | `/health` returns 200 on real pod |
| Real Qdrant + binding cache | `/internal/health` shows `qdrant: ok`, 3 bots in cache |
| Bot auth success | Real relay token for `tim-joblogic-sa` → 5 results from both bound scopes |
| Auth isolation (no token) | Request without auth → 401 |
| Auth isolation (wrong token) | Fake token for real bot → 401 |
| Auth isolation (nonexistent bot) | Fake bot name → 401 |
| Scope isolation | `dan-joblogic-qa` (bound to `api-docs` only) gets 0 results from `debate-docs` — while `tim-joblogic-sa` (bound to both) does |
| Internal token search | Operator explicit-scope search returns results |
| Nonexistent scope | Search against fake scope → 0 results |

### How to run

```bash
bash e2e/real_system_validation.sh
```

Prerequisites:
- SSH access to `server36` (configurable via `SSH_HOST` env var)
- `kubectl` access via `KUBECONFIG=/etc/rancher/k3s/k3s.yaml`
- knowledge-api pod running in `agentopia-dev`
- At least one bot with knowledge-scope bindings deployed

### Key evidence: scope isolation

The scope isolation test is the strongest assertion. Same query, same cluster:

- **Tim** (`tim-joblogic-sa`): bound to `api-docs` + `debate-docs` → results from **both** scopes
- **Dan** (`dan-joblogic-qa`): bound to `api-docs` only → results from `api-docs` **only**, zero `debate-docs`

This proves the binding cache + scope resolution + Qdrant collection-level isolation path end-to-end.

### What is NOT covered

| Gap | Why |
|---|---|
| Dedicated test bot/scope | Would require bot-config-api deploy — uses existing bots instead |
| Gateway Telegram → plugin → knowledge-api | Plugin calls knowledge-api directly; full chain is manual Telegram validation |
| Postgres document lifecycle | Postgres auth failure on dev cluster (`$(POSTGRES_PASSWORD)` env expansion issue) — search works via Qdrant only |
| Automated CI/CD pipeline E2E | No scheduled/triggered real-system tests — manual `bash e2e/real_system_validation.sh` |

### Environment

- Cluster: k3s on `server36`, namespace `agentopia-dev`
- knowledge-api: real pod, real Qdrant at `qdrant.agentopia-dev.svc.cluster.local:6333`
- Embedding: via `agentopia-llm-proxy` → OpenRouter (`openrouter/openai/text-embedding-3-small`)
- Bots tested: `tim-joblogic-sa` (2 scopes), `dan-joblogic-qa` (1 scope)

---

## Local Compose E2E (secondary)

A local compose stack (`compose.yaml` + `compose.e2e.yaml`) can run a standalone E2E harness with a deterministic embedding stub. This does NOT test the real Agentopia runtime — it tests the knowledge-api service boundary in isolation.

See `compose.e2e.yaml` and `tests/e2e/` for the local harness. This path is secondary to the real-system validation above.
