# Evaluation

Retrieval quality is measured against labeled golden question sets. No retrieval pipeline change is promoted to production without passing the evaluation gate.

## Current Production Baseline

Measured on dense-only retrieval (`text-embedding-3-small`, 1536d, top-5 results):

| Metric | Value | Meaning |
|---|---|---|
| nDCG@5 | 0.925 | Normalized Discounted Cumulative Gain at rank 5 — quality of ranked result ordering |
| MRR | 0.96 | Mean Reciprocal Rank — how quickly the first relevant result appears |
| P@5 | 0.84 | Precision at rank 5 — fraction of top-5 results that are relevant |
| R@5 | 1.0 | Recall at rank 5 — all relevant docs found within top-5 |

These metrics were established during the Super RAG production milestone (Phase 1a/1b). The evaluation datasets and runner scripts are in `evaluation/` in the source tree.

---

## Evaluation Framework

### Location (current source)

```
evaluation/
├── datasets/                       ← labeled question sets per scope
├── retrieval_metrics.py            ← nDCG, MRR, P@k, R@k computation
├── phase1a_runner.py               ← Phase 1a evaluation runner
├── phase1b_baseline.py             ← Phase 1b dense-only baseline
├── e2e_baseline_test.py            ← end-to-end baseline smoke test
├── run_phase2a_comparison.py       ← Phase 2a sparse index comparison
├── run_phase2a_full_comparison.py  ← Phase 2a full scope comparison
├── w1_promotion_gate.py            ← W1 markdown-aware promotion gate
├── w1_real_pilot_gate.py           ← W1 real pilot scope gate
├── w3a_expansion_comparison.py     ← W3a query expansion evaluation
└── results/                        ← stored evaluation outputs
```

### Golden question format

Each question set consists of:
- A natural-language query
- One or more labeled relevant document IDs (source + scope)
- Optionally, a relevance grade (1–3) for graded nDCG computation

### How evaluation runs

```bash
# From repo root
PYTHONPATH=src python evaluation/phase1b_baseline.py --scope acme-corp/api-docs --qdrant-url http://localhost:6333
```

Output: per-question scores, aggregate nDCG/MRR/P@k/R@k, comparison delta if baseline file provided.

---

## Promotion Gates

### For chunking strategy changes

1. Enable new strategy on a single pilot scope (test environment)
2. Run the full golden question set for that scope
3. Gate: nDCG@5 must not regress more than 0.01 from baseline
4. If gate passes: enable as opt-in `ChunkingStrategy` option for operator-selected scopes
5. Do not make new strategy the default until it has passed on at least 3 distinct scopes

### For retrieval mode changes (query expansion, HyDE, reranking)

1. Implement behind a per-scope feature flag (disabled by default)
2. Run latency benchmarks: measure P50/P95 query latency delta vs baseline
3. Present cost/latency budget to CTO before enabling on any production scope
4. Run full golden question set with new mode
5. Gate: nDCG@5 must improve by ≥ 0.02 AND latency increase must be within approved budget
6. If gate passes: enable as opt-in per-scope configuration

### For embedding model changes

Embedding model changes affect the entire vector space. They require:
1. Re-embedding all documents in all scopes (full reindex)
2. Dimension validation: `QdrantBackend._validate_collection_dimensions()` logs a warning on mismatch but does NOT block startup — this allows graceful migration where new scopes use the new dimension while old scopes are reindexed incrementally
3. Full evaluation run across all scopes
4. Staged rollout: new model on new scopes first, existing scopes migrated scope-by-scope
5. Gate: all scopes must maintain nDCG@5 ≥ 0.90 (current minus 0.025 margin)

**Dimension mismatch is warning-only, not a hard error.** The operator must monitor for `DIMENSION_MISMATCH` log warnings and schedule reindexing of affected scopes. Queries against mismatched collections will return degraded results until reindexed.

---

## W1 Evidence — Markdown-Aware Chunking

**Date:** 2026-04-09  
**Script:** `evaluation/w1_promotion_gate.py`  
**Results:** `evaluation/results/w1_promotion_gate.json`

### Evidence 1: Regression check (phase1b_labeled_seed.json, 5 queries)

Seed documents are plain text — both strategies produce identical chunks (markdown-aware falls back to fixed-size when no headings or code fences are present). Proves no regression on existing behavior.

| Metric | FIXED_SIZE | MARKDOWN_AWARE | Delta |
|---|---|---|---|
| nDCG@5 | 0.8774 | 0.8774 | +0.0000 |
| MRR | 0.8400 | 0.8400 | +0.0000 |

**Status:** Complete. No regression.

### Evidence 2: Synthetic markdown benchmark (w1_markdown_pilot.json, 5 queries)

Synthetic corpus with markdown-structured documents (headings, sections). Demonstrates that heading-aware chunking improves retrieval when documents have markdown structure. This is a **synthetic benchmark**, not a real production pilot scope.

| Metric | FIXED_SIZE | MARKDOWN_AWARE | Delta |
|---|---|---|---|
| nDCG@5 | 0.5861 | **0.9956** | +0.4095 |
| MRR | 0.5167 | **1.0000** | +0.4833 |
| P@5 | 0.2000 | **0.3200** | +0.1200 |
| R@5 | 0.5000 | **0.8333** | +0.3333 |

**Status:** Complete. Strong directional signal that markdown-aware improves retrieval on structured docs.

### Evidence 3: Real pilot-scope gate (w1_real_pilot.json, 10 queries, joblogic-kb/api-docs)

Real production documents from `ai-agentopia/docs` repo. 9 documents loaded (corpus explicitly defined — `ui-knowledge-workflow-trigger.md` excluded as it was ingested directly, not from the docs repo, and no labeled queries reference it).

**Corpus:** 9 markdown documents, 244 chunks (fixed-size) / 349 chunks (markdown-aware after remediation).
**Labels:** REVIEWED — each query-source mapping verified against document heading structure. CTO sign-off complete.
**Metric model:** nDCG@5 and MRR are authoritative. P@5 is directional. Recall@5 excluded (invalid with source-level grading).

| Metric | FIXED_SIZE (244 chunks) | MARKDOWN_AWARE (349 chunks) | Delta |
|---|---|---|---|
| nDCG@5 | 0.7440 | **0.7917** | **+0.0477** |
| MRR | 0.7500 | **0.7667** | +0.0167 |
| P@5 | 0.4200 | **0.4400** | +0.0200 |

**Gate result: PASSED** — nDCG@5 improved by 0.0477 (no regression).

**Remediation applied:** Original W1 implementation produced 388 chunks (82 under 100 chars) causing score dilution. Remediation: (1) filter horizontal-rule-only blocks (`---`), (2) force-attach heading-only blocks to following content block — headings are never emitted as standalone chunks. Post-remediation: 349 chunks, 42 under 100 chars.

**Remaining limitation:** In-memory cosine search, not production Qdrant embedding search.

**Script:** `evaluation/w1_real_pilot_gate.py`
**Results:** `evaluation/results/w1_real_pilot_gate.json`

### Current W1 status

- Implementation: complete, merged, opt-in only
- Regression check on seed scope: complete, no regression
- Synthetic benchmark: complete, positive signal (nDCG@5 +0.4095)
- Real pilot-scope gate: **PASSED** (nDCG@5 +0.0477)
- W1 is safe for opt-in on documentation-heavy scopes. Default remains `fixed_size`.

---

## W3a Evidence — Query Expansion

**Script / artifact in repo:** `evaluation/w3a_expansion_comparison.py`, `evaluation/results/w3a_expansion_comparison.json`

### Evidence 1: Simulated benchmark

Manual/simulated alternative phrasings on `joblogic-kb/api-docs` showed positive directional signal:

| Metric | Baseline | Expanded | Delta |
|---|---|---|---|
| nDCG@5 | 0.7917 | 0.8568 | +0.0652 |
| MRR | 0.7667 | 0.8250 | +0.0583 |

**Status:** Complete. Useful mechanism check only — not production acceptance evidence.

### Evidence 2: Live evaluation + bounded model sweep

Live LLM evaluation on the real pilot corpus did **not** clear the retrieval-mode gate (`nDCG@5` must improve by `>= +0.02`):

| Expansion model | nDCG@5 delta vs baseline | Result |
|---|---|---|
| `openai/gpt-4o-mini` | -0.0050 | Fail |
| `openai/gpt-4.1-mini` | +0.0050 | Fail |

**Current W3a status:** Implemented, default-off, **not approved** for production rollout on the current corpus. The code remains dormant and per-scope gated for future reconsideration only.

---

## W3b Evidence — HyDE

**Results:** `evaluation/results/w3b_hyde_live_eval.json`

Live HyDE evaluation on `joblogic-kb/api-docs`:

| Metric | Baseline | HyDE | Delta |
|---|---|---|---|
| nDCG@5 | 0.9201 | 0.9175 | -0.0026 |
| MRR | 0.9000 | 0.9000 | +0.0000 |
| Avg latency | 577ms | 3313ms | +2736ms |

**Gate result:** Fail. HyDE did not improve `nDCG@5` and materially increased latency.

**Current W3b status:** Implemented, default-off, **not approved** for production rollout on the current corpus. Future reconsideration requires a different corpus/evidence set, not a silent enablement.

---

## W4 Evidence — LLM Listwise Reranking

**Results:** `evaluation/results/w4_reranking_live_eval.json`

Live reranking evaluation on `joblogic-kb/api-docs` (18 chunks, 258 Qdrant points at eval time):

| Metric | Baseline | Reranked | Delta |
|---|---|---|---|
| nDCG@5 | 0.5262 | 0.4024 | -0.1238 |
| MRR | 0.5000 | 0.3333 | -0.1667 |
| Avg rerank latency | — | 908ms | — |

**Mechanism:** LLM listwise reranking — K=20 candidates retrieved from Qdrant, all sent to `openai/gpt-4o-mini` in one call for relevance ranking, top-5 returned from reranked order.

**Cost:** ~$0.0009/query at gpt-4o-mini pricing (K=20 × ~300 chars ≈ 6000 input tokens + ~50 output tokens).

**Gate result:** Fail. The reranker actively regressed retrieval quality (−0.1238 nDCG@5). gpt-4o-mini consistently misranked domain-specific documents (e.g., promoted `production-super-rag.md` over `chatbot-architecture.md` for queries about monitoring and bot authentication).

**Note:** Baseline nDCG@5 differs from W3a/W3b baseline (0.9201) because the corpus expanded — new documents ingested between evaluations. The lower baseline (0.5262) would normally make improvement *easier* to demonstrate, making the reranker regression result even more negative.

**Current W4 status:** Implemented, default-off, **not approved** for production rollout on the current corpus with the current model. Future reconsideration requires: a domain-tuned cross-encoder (e.g., Cohere Rerank) or a different corpus where generic LLM ranking adds signal rather than noise.

---

## What Is Not Evaluated Here

- Reasoning quality (planner, reviewer-shadow) — owned by `agentopia-graph-executor`
- End-to-end workflow quality — owned by Temporal workflow evaluation in `agentopia-protocol`
- Bot response quality — out of scope for the retrieval layer

Retrieval evaluation answers only: given a query and a corpus, are the right documents returned in the right order?

---

## Runtime Evaluation and Governance

The system includes a runtime evaluation layer that governs document replacements. This is separate from the research evaluation scripts in `evaluation/` — it runs as part of the service on every document replacement.

### Per-Scope Baselines

Every scope has an independently established baseline. No global nDCG threshold applies across all scopes — a scope with structured markdown docs and a scope with mixed PDFs will have different baseline values.

Baselines are stored in the `evaluation_baselines` table (PostgreSQL). They are human-curated (established by running `POST /api/v1/evaluation/baselines/{scope}`) and rarely changed.

### Golden Questions

Golden questions are stored in the `golden_questions` table. Each question has:
- A natural-language query
- `expected_sources`: list of `{source, relevance}` where source is the document filename/path and relevance is 0 (not relevant), 1 (partially relevant), or 2 (fully relevant)
- An optional weight

Manage via:
```
GET    /api/v1/evaluation/questions/{scope}   — list questions
POST   /api/v1/evaluation/questions/{scope}   — add question
DELETE /api/v1/evaluation/questions/{id}      — remove question
```

### Regression Gate on Document Replacement

When a document replacement completes (new version reaches `active` state), the evaluation service automatically:

1. Runs all golden questions for the affected scope against live retrieval
2. Computes nDCG@5, MRR, P@5, R@5
3. Compares nDCG@5 against the per-scope baseline
4. Applies the gate:

| Delta (nDCG@5 vs baseline) | Verdict | Action |
|---|---|---|
| >= 0 | `passed` | No action; logged at INFO |
| >= -0.02 and < 0 | `warning` | Document stays active; operator informed via log |
| < -0.02 | `blocked` | Document stays active (already committed); operator notified; override required |

**Important:** The document is **already active** when the gate fires. The gate is a governance notification, not a pre-commit check. If the verdict is `blocked`, the operator can either:
- Roll back to the prior version via `POST /api/v1/knowledge/{scope}/documents/{source}` (re-ingest prior version)
- Accept the regression via `POST /api/v1/evaluation/results/{result_id}/override`

### Operator Notification

Regression blocks are surfaced as:
- Structured log at WARNING level with fields: scope, document_id, version, ndcg_5, delta, result_id
- Record in `evaluation_results` table with `verdict = 'blocked'`
- Visible via `GET /api/v1/evaluation/results?scope={scope}`

### API Endpoints

```
GET  /api/v1/evaluation/baselines                    — list all baselines
GET  /api/v1/evaluation/baselines/{scope}            — get baseline for scope
POST /api/v1/evaluation/baselines/{scope}            — establish/refresh baseline
GET  /api/v1/evaluation/results?scope={scope}        — evaluation run history
POST /api/v1/evaluation/results/{result_id}/override — accept regression
GET  /api/v1/evaluation/questions/{scope}            — list golden questions
POST /api/v1/evaluation/questions/{scope}            — add golden question
DELETE /api/v1/evaluation/questions/{id}             — remove golden question
POST /api/v1/evaluation/run/{scope}                  — manual benchmark trigger
```

All endpoints require `X-Internal-Token` auth.

### Evaluation Results Table

`evaluation_results` is **append-only** — rows are never deleted. This provides a full quality audit trail across all document replacements for every scope.

Fields: id, scope, document_id, document_version, run_at, trigger, ndcg_5, mrr, p_5, r_5, delta_ndcg_5, verdict, operator_override, operator_note.

### Gate Failure Behavior

- If the evaluation service fails to connect to the database: gate is skipped, document stays active, WARNING logged
- If golden question search fails: gate is skipped, verdict = `eval_error`, WARNING logged
- Evaluation failures **never roll back** a completed ingest — the document remains active and the pipeline continues
