# Evaluation

## Phase 1 Status: Placeholder

The evaluation harness and golden question datasets will be copied here in **Phase 2** (code extraction).

Current source location:
```
agentopia-protocol/knowledge-api/src/evaluation/
├── datasets/              ← golden question sets per scope
├── retrieval_metrics.py   ← nDCG, MRR, P@k, R@k computation
├── phase1a_runner.py
├── phase1b_baseline.py
├── e2e_baseline_test.py
├── run_phase2a_comparison.py
├── run_phase2a_full_comparison.py
└── results/               ← stored baseline outputs
```

## Production Baseline (from Phase 0 docs)

These metrics will be reproduced from this repo's evaluation harness after Phase 2:

| Metric | Value |
|---|---|
| nDCG@5 | 0.925 |
| MRR | 0.96 |
| P@5 | 0.84 |
| R@5 | 1.0 |

Baseline measured on dense-only retrieval (`text-embedding-3-small`, 1536d, top-5 results).

See [docs/evaluation.md](../docs/evaluation.md) for promotion gates and evaluation framework details.
