# Migration: Extraction from agentopia-protocol

## Current Phase: Phase 0 — Repo Bootstrap

The service code has not yet been extracted. Production continues to run from:

```
agentopia-protocol/knowledge-api/
├── src/
│   ├── main.py
│   ├── models/knowledge.py
│   ├── services/knowledge.py       ← QdrantBackend, KnowledgeService, chunking
│   ├── services/document_store.py  ← PostgresDocumentStore, document lifecycle
│   ├── services/binding_cache.py   ← K8s CRD binding cache
│   ├── auth/guards.py              ← dual-path auth
│   ├── routers/knowledge.py        ← API routes
│   ├── evaluation/                 ← golden question sets, metrics runner
│   └── tests/
└── Dockerfile
```

This repo (`agentopia-super-rag`) currently contains only documentation. Code extraction has not started.

---

## Extraction Phases

### Phase 0 (current): Repo bootstrap
- [x] GitHub repo created
- [x] Documentation baseline written
- [ ] Code not yet moved

### Phase 1: Code mirror
Move the `agentopia-protocol/knowledge-api/` subtree into this repo without functional changes:
- Copy `src/`, `Dockerfile`, `pyproject.toml`, `requirements.txt`
- Copy `evaluation/` with all datasets and runners
- Set up CI in this repo (test gate + Docker build/push)
- The monorepo path (`agentopia-protocol/knowledge-api/`) becomes a stub that imports from this service or is deleted

### Phase 2: Infrastructure handover
- Move Helm templates from `agentopia-infra/charts/agentopia-base/templates/knowledge-api.yaml` to a chart owned by this repo (or referenced chart)
- Update ArgoCD Image Updater to track this repo's image tag pattern
- Update `agentopia-protocol` bot-config-api to reference the new image/service DNS (no change expected — service DNS in cluster stays the same)

### Phase 3: Stable extracted service
- This repo is the single source of truth for the retrieval service
- `agentopia-protocol` no longer contains knowledge-api source
- CI/CD fully owned here

---

## Compatibility Assumptions

- **Service DNS does not change.** The in-cluster DNS name (`knowledge-api.agentopia-dev.svc.cluster.local`) is set by the Helm chart and is independent of the source repo. No callers need updating.
- **Internal token is compatible.** `KNOWLEDGE_API_INTERNAL_TOKEN` is injected via K8s Secret and will remain the same value during migration.
- **Image tag format is compatible.** ArgoCD Image Updater tracks `dev-{sha}` tags. The new repo will use the same tag format.
- **Qdrant collections are not migrated.** Existing collections remain in place. The service reconnects to the same Qdrant instance with the same collection names (SHA-256 of scope identities).
- **Postgres document store is not migrated.** Same database, same schema, same connection string.

---

## Rollback Strategy

During Phase 1 (code mirror), rollback is trivial: the old image from `agentopia-protocol` builds is still available in GHCR. ArgoCD Image Updater can be pointed back to the old image tag pattern.

During Phase 2 (infra handover), rollback requires reverting the Helm chart changes in `agentopia-infra`. Since ArgoCD tracks Git, a revert commit restores the previous state.

There is no data migration. Qdrant and Postgres are shared services; no data changes during the repo extraction. Rollback does not affect stored knowledge.

---

## What Is Not Changing

- Auth model (internal token + bot bearer)
- Scope isolation enforcement
- Document lifecycle (two-phase atomic replace)
- Embedding model (`text-embedding-3-small`, 1536d)
- Evaluation dataset format and metric definitions
- K8s namespace (`agentopia-dev`, `agentopia-uat`)
