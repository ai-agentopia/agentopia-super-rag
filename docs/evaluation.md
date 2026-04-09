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

## W1 Promotion Gate Results — Markdown-Aware Chunking

**Date:** 2026-04-09  
**Script:** `evaluation/w1_promotion_gate.py`  
**Results:** `evaluation/results/w1_promotion_gate.json`

### Part 1: Regression check (phase1b_labeled_seed.json, 5 queries)

Seed documents are plain text — both strategies produce identical chunks (fallback). Proves no regression.

| Metric | FIXED_SIZE | MARKDOWN_AWARE | Delta |
|---|---|---|---|
| nDCG@5 | 0.8774 | 0.8774 | **+0.0000** |
| MRR | 0.8400 | 0.8400 | +0.0000 |

### Part 2: W1 pilot (w1_markdown_pilot.json, 5 queries, markdown-structured docs)

Documents with headings, sections, and structured content — exercises the actual strategy.

| Metric | FIXED_SIZE | MARKDOWN_AWARE | Delta |
|---|---|---|---|
| nDCG@5 | 0.5861 | **0.9956** | **+0.4095** |
| MRR | 0.5167 | **1.0000** | +0.4833 |
| P@5 | 0.2000 | **0.3200** | +0.1200 |
| R@5 | 0.5000 | **0.8333** | +0.3333 |

### Gate decision

- Regression check: PASS (nDCG@5 delta = +0.0000)
- W1 pilot: PASS (nDCG@5 delta = +0.4095)
- **Gate: PASSED**

### Rollout recommendation

Safe as opt-in for documentation-heavy scopes. NOT recommended as default until validated on at least 3 production scopes per promotion gate rules above.

**Rollout recommendation:** Implemented and available as opt-in. Do NOT enable on production scopes until the full promotion gate above is executed.

---

## What Is Not Evaluated Here

- Reasoning quality (planner, reviewer-shadow) — owned by `agentopia-graph-executor`
- End-to-end workflow quality — owned by Temporal workflow evaluation in `agentopia-protocol`
- Bot response quality — out of scope for the retrieval layer

Retrieval evaluation answers only: given a query and a corpus, are the right documents returned in the right order?
