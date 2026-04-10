# Contributing to agentopia-super-rag

This is the production knowledge-api service for the Agentopia platform. It is **not** a general-purpose RAG library. Contributions must be consistent with that scope.

---

## Repo model

**Main-only.** No `dev`, `uat`, or feature branches on the remote. All work goes through pull requests against `main`. CI runs on `push` to `main` and on PRs targeting `main`.

---

## Before you open a PR

### For bug fixes
- Reproduce the bug with a failing test
- Fix the root cause, not the symptom
- All 421+ fast-gate tests must pass: `python -m pytest tests/ -m "not integration and not e2e" -x -q`

### For retrieval pipeline changes (chunking, retrieval mode, embedding model)
These are gated. Read `docs/evaluation.md` before opening a PR.

- Chunking changes: nDCG@5 must not regress more than 0.01 on the evaluation scope
- Retrieval mode changes (query expansion, HyDE, reranking): nDCG@5 must improve ≥ 0.02 AND latency within approved budget
- Embedding model changes: require full reindex + evaluation across all scopes

**Do not open a PR for a retrieval change without evaluation evidence.** A PR without evaluation results will not be reviewed.

### For new features
- Must align with the service boundary (`docs/architecture.md`)
- Must not break existing API contracts
- Must have tests
- Must have doc updates (`README.md`, `docs/architecture.md`, `docs/operations.md` as applicable)

---

## Pull request process

1. Open a PR with the template filled in
2. The fast gate must pass (CI runs automatically)
3. At least one CTO review required before merge
4. No merge without resolving all review comments
5. Squash or merge commit — no history rewriting on `main`

---

## Doc sync rule

After any code change:
- Update affected docs in `docs/`
- Update `README.md` if the local workflow, env vars, or feature roadmap changed
- Update `docs/evaluation.md` if any retrieval pipeline outcome changed

---

## What is not in scope for external PRs

- W-series retrieval features that have already been evaluated and not approved (W3a, W3b, W4) — these require new evaluation evidence on a new corpus, not a code-only change
- W2 hybrid retrieval — frozen, conditional reopen only (see [#15](https://github.com/ai-agentopia/agentopia-super-rag/issues/15))
- Architecture changes to service boundaries — those originate from the Agentopia platform roadmap
- Changes to the embedding model without a full reindex + evaluation plan

---

## Local dev

See `README.md` for setup and `docs/operations.md` for env var reference.

For questions, open a GitHub Discussion or an issue tagged `question`.
