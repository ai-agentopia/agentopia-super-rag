# Migration: Extraction from agentopia-protocol

## Current Phase: Phase 0 — Contract / Ownership Freeze

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
│   ├── routers/internal.py         ← internal binding-sync, deep health
│   ├── evaluation/                 ← golden question sets, metrics runner
│   └── tests/
└── Dockerfile
```

---

## Phase 0 Exit Checklist

Phase 1 (repo bootstrap) must not begin until all items below are closed.

- [ ] **A1** — [agentopia-super-rag#1](https://github.com/ai-agentopia/agentopia-super-rag/issues/1): API contract frozen — `docs/architecture.md` verified against source at freeze commit and accepted by CTO
- [ ] **A2** — [agentopia-protocol#387](https://github.com/ai-agentopia/agentopia-protocol/issues/387): Freeze notice committed to `agentopia-protocol/knowledge-api/` — no new features after this date
- [ ] **A3** — [agentopia-super-rag#2](https://github.com/ai-agentopia/agentopia-super-rag/issues/2): Artifact ownership cutover rule accepted — single publisher, explicit sequence, no overlap
- [ ] **A4** — [agentopia-super-rag#3](https://github.com/ai-agentopia/agentopia-super-rag/issues/3): This file updated with Phase 0 exit checklist and cutover rule (this doc)

---

## Image Ownership Cutover Rule

**Single publisher rule:** `ghcr.io/ai-agentopia/knowledge-api` has exactly one publisher at any given time.

Current publisher: `agentopia-protocol` CI.
Future publisher (after Phase 3): `agentopia-super-rag` CI.

**Cutover sequence (must be followed in order):**

1. `agentopia-super-rag/.github/workflows/build-image.yml` switched from `workflow_dispatch` to `push` trigger on `dev`
2. First `dev-{sha}` image pushed from this repo — GHCR URL captured as evidence
3. `agentopia-infra` Image Updater annotation updated to track new publisher (agentopia-infra#113)
4. `agentopia-protocol` CI image push step disabled in same deploy window (agentopia-protocol#389)
5. Deployed pod image SHA verified against GHCR push from this repo

**No-overlap constraint:** Steps 3 and 4 must be completed in the same deploy window. There is no period where both repos push to the same image tag. If step 2 fails, abort — do not proceed to steps 3/4.

The atomic cutover transaction is owned by **agentopia-super-rag#24**.

---

## Extraction Phases

### Phase 0: Contract / Ownership Freeze (pending sign-off)
- [x] GitHub repo created
- [x] Documentation baseline written and route table verified against source (PR #26)
- [x] Phase 0 exit checklist added to this doc
- [x] Image ownership cutover rule documented
- [ ] A1: CTO sign-off on docs/architecture.md as authoritative contract
- [ ] A2: Freeze notice committed to agentopia-protocol/knowledge-api/ (PR #390)
- [ ] A3: Artifact ownership issue accepted
- [ ] A4: This doc accepted as complete

### Phase 1 (current): Repo Bootstrap
- [x] Directory structure: `src/`, `tests/`, `evaluation/`, `.github/workflows/`
- [x] `pyproject.toml` with Python 3.12 constraint and pytest config
- [x] `requirements.txt` pinned from monorepo freeze commit
- [x] `Dockerfile` placeholder (structurally correct, source COPY commented until Phase 2)
- [x] CI: `.github/workflows/ci.yml` (fast gate on dev push, passes with 0 tests in Phase 1)
- [x] CI: `.github/workflows/build-image.yml` (workflow_dispatch only, push gated by cutover)
- [x] `.gitignore`
- [ ] B3: Docs verified and signed off against current source (pending Phase 0 merge)

### Phase 2: Code Extraction
Copy monorepo source without functional changes (using freeze commit SHA as copy source):
- `src/` (all modules), `Dockerfile`, `pyproject.toml`, `requirements.txt`
- `evaluation/` with all datasets and runners
- `tests/`
- Zero behavior drift: `diff -r` between copied source and monorepo freeze commit must show path changes only
- Contract preservation audit (C4) required before Phase 3 begins

### Phase 3: Atomic Cutover
Transfer image ownership per the cutover rule above (agentopia-super-rag#24):
- Enable image push in this repo's CI
- Switch ArgoCD Image Updater (agentopia-infra#113)
- Disable monorepo image push (agentopia-protocol#389)
- Post-cutover validation in `agentopia-dev` (agentopia-super-rag#12)
- Rollback drill mandatory (agentopia-super-rag#13)
- Monorepo source retired only after validation + rollback drill confirmed (agentopia-protocol#388)

### Phase 4: Wiki-RAG Evolution
Incremental retrieval improvements, each isolated and independently evaluated:
- **W1**: Markdown-aware chunking (`ChunkingStrategy.MARKDOWN_AWARE`)
- **W1.5**: Context/path-aware retrieval — preserve document hierarchy/path in chunk metadata to improve docs/wiki corpora
- **W3a**: Query expansion via LLM alternative phrasings (per-scope opt-in, latency budget required)
- **W3b**: HyDE — hypothetical document embedding (separate eval run from W3a)
- **W4**: Cross-encoder reranking (highest cost, CTO approval per scope)
- **W2**: BM25 hybrid — **FROZEN**, conditional reopen only (agentopia-super-rag#15)

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
