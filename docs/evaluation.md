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
agentopia-protocol/knowledge-api/src/evaluation/
├── datasets/              ← labeled question sets per scope
├── retrieval_metrics.py   ← nDCG, MRR, P@k, R@k computation
├── phase1a_runner.py      ← Phase 1a evaluation runner
├── phase1b_baseline.py    ← Phase 1b dense-only baseline
├── e2e_baseline_test.py   ← end-to-end baseline smoke test
├── run_phase2a_comparison.py      ← Phase 2a sparse index comparison
├── run_phase2a_full_comparison.py ← Phase 2a full scope comparison
└── results/               ← stored evaluation outputs
```

### Golden question format

Each question set consists of:
- A natural-language query
- One or more labeled relevant document IDs (source + scope)
- Optionally, a relevance grade (1–3) for graded nDCG computation

### How evaluation runs

```bash
# From knowledge-api/src/evaluation/
python phase1b_baseline.py --scope acme-corp/api-docs --qdrant-url http://localhost:6333
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

### Evidence 3: Real pilot-scope provisional run (w1_real_pilot.json, 10 queries, joblogic-kb/api-docs)

Real production documents from `ai-agentopia/docs` repo. **Provisional evidence only** — not an authoritative gate decision. See limitations below.

| Metric | FIXED_SIZE (244 chunks) | MARKDOWN_AWARE (371 chunks) | Delta |
|---|---|---|---|
| nDCG@5 | 0.7440 | 0.6309 | -0.1131 |
| MRR | 0.7500 | 0.6333 | -0.1167 |

**Directional signal:** nDCG@5 and MRR both show regression. This is a negative directional signal for W1 on this corpus, but is NOT an authoritative gate decision.

**Limitations of this evidence:**
1. **DRAFT labels:** Dataset labels created from document heading analysis, not CTO-reviewed. Authoritative gate requires label review.
2. **Partial corpus:** 9 of 10 source documents loaded (1 not found locally: `ui-knowledge-workflow-trigger.md`). Not a fully faithful scope replay.
3. **Invalid Recall@5:** Source-level grading produces Recall@5 > 1.0 (denominator is source count, numerator is chunk count). Recall values in this run are not meaningful. nDCG@5, MRR, and Precision@5 are valid.
4. **In-memory search only:** Comparison uses in-memory cosine similarity, not the production Qdrant retrieval path. Embedding-based search may rank differently.

**Observation:** Markdown-aware produces 52% more chunks (371 vs 244) from the same documents. With in-memory cosine similarity, more smaller chunks appear to dilute scoring. Whether this holds with embedding-based Qdrant search is unknown.

**Script:** `evaluation/w1_real_pilot_gate.py`
**Results:** `evaluation/results/w1_real_pilot_gate.json`

### Current W1 status

- Implementation: complete, merged, opt-in only
- Regression check on seed scope: complete, no regression
- Synthetic benchmark: complete, positive signal (nDCG@5 +0.4095)
- Real pilot-scope run: provisional negative signal (nDCG@5 -0.1131), NOT authoritative
- **Authoritative gate decision: pending** — requires label review, full corpus, and ideally embedding-based evaluation
- W1 remains opt-in. NOT promoted for production use.

---

## What Is Not Evaluated Here

- Reasoning quality (planner, reviewer-shadow) — owned by `agentopia-graph-executor`
- End-to-end workflow quality — owned by Temporal workflow evaluation in `agentopia-protocol`
- Bot response quality — out of scope for the retrieval layer

Retrieval evaluation answers only: given a query and a corpus, are the right documents returned in the right order?
