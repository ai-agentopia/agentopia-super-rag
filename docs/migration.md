# Migration: Extraction from agentopia-protocol

> **Status: Complete.** All extraction phases executed. `agentopia-super-rag` is the production source as of 2026-04-09. `agentopia-protocol/knowledge-api/` is retired and no longer receives changes.

This document is a historical record of the extraction. It is not a guide to any pending work.

---

## Final State

- `agentopia-super-rag` main branch is the only production source for the knowledge-api service.
- `ghcr.io/ai-agentopia/knowledge-api` is published exclusively from this repo's CI.
- `agentopia-protocol/knowledge-api/` is frozen at the last extraction commit. No new changes.
- All routes, auth model, and behavior are documented in `docs/architecture.md`.

---

## Phase Summary

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Contract freeze: `docs/architecture.md` declared authoritative; freeze notice added to monorepo | Complete |
| Phase 1 | Repo bootstrap: directory structure, CI skeleton, docs baseline | Complete |
| Phase 2 | Code extraction: `src/`, `tests/`, `evaluation/` copied from monorepo freeze commit | Complete |
| Phase 3 | Atomic cutover: image ownership transferred from `agentopia-protocol` to this repo | Complete (2026-04-09) |
| Phase 4 | Wiki-RAG evolution: W1, W1.5, W3a, W3b, W4 implemented and evaluated | Complete |

---

## Cutover Record

**Cutover date:** 2026-04-09

Atomic cutover executed per the single-publisher rule:

1. `.github/workflows/build-image.yml` push trigger enabled on `main` in this repo (main-only repo — no dev/uat branches)
2. `ghcr.io/ai-agentopia/knowledge-api:dev-{sha}` first published from this repo
3. `agentopia-infra` ArgoCD Image Updater annotation updated to track this repo's tags
4. `agentopia-protocol` CI image push step disabled (`agentopia-protocol#389`)
5. Deployed pod verified against GHCR push from this repo

No overlap period: steps 3 and 4 were completed in the same deploy window.

---

## W-Series Outcomes (Phase 4)

| Item | Status | Evidence |
|---|---|---|
| W1 — Markdown-aware chunking | **Accepted (opt-in)** | Gate PASSED: nDCG@5 +0.0477 on real pilot scope |
| W1.5 — Section path context | **Accepted** | `section_path` field added to `Citation`. Backward compatible, additive. |
| W2 — BM25/hybrid retrieval | **Frozen** | No implementation. Conditional reopen: Qdrant sparse support + explicit CTO reopen. |
| W3a — Query expansion | **Not approved** | Live eval: nDCG@5 delta below +0.02 gate on both models tested |
| W3b — HyDE | **Not approved** | Live eval: nDCG@5 −0.0026, latency +2736ms |
| W4 — LLM listwise reranking | **Not approved** | Live eval: nDCG@5 −0.1238 (gpt-4o-mini misranks domain-specific docs) |

Full evaluation evidence: `docs/evaluation.md`, `evaluation/results/`.

---

## Compatibility Preserved

No compatibility changes were made during extraction or cutover:

- **Service DNS** — `knowledge-api.agentopia-dev.svc.cluster.local` unchanged (set by Helm chart)
- **Internal token** — `KNOWLEDGE_API_INTERNAL_TOKEN` same K8s Secret value
- **Image tag format** — `dev-{sha}` — ArgoCD Image Updater continues tracking same pattern
- **Qdrant collections** — same collections, no data migration
- **Postgres document store** — same database and schema, no migration
- **Auth model** — internal token + bot bearer, unchanged
- **Embedding model** — `openrouter/openai/text-embedding-3-small`, 1536d, unchanged

---

## Rollback

If a rollback is ever required (unlikely since the service is now standalone):

- ArgoCD tracks Git. A revert commit in `agentopia-infra` restores the previous Helm state.
- There is no data migration to undo. Qdrant and Postgres are shared services and were not touched.
- Old images from `agentopia-protocol` CI builds remain in GHCR with `dev-{sha}` tags if needed.
