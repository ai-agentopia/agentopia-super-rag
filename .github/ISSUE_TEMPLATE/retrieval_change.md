---
name: Retrieval pipeline change
about: Propose a change to chunking, retrieval mode, or embedding model
labels: retrieval
---

## What you want to change

<!-- Describe the retrieval pipeline change. Be specific about which stage:
     chunking / embedding / search / reranking / query expansion / HyDE -->

## Motivation

<!-- Why do you believe this will improve retrieval quality? -->

## Evaluation plan

<!-- Retrieval pipeline changes require evaluation evidence before any PR is reviewed.
     Describe your evaluation plan before writing any code.
     See docs/evaluation.md for gate criteria. -->

Gate type:
- [ ] Chunking change → nDCG@5 must not regress > 0.01
- [ ] Retrieval mode change → nDCG@5 must improve ≥ 0.02 AND latency within budget
- [ ] Embedding model change → full reindex + all scopes at nDCG@5 ≥ 0.90

Pilot scope for evaluation:
Evaluation dataset:
Expected delta:

## W-series status

If this overlaps with a previous W-series item:
- W3a query expansion: not approved on current corpus — new corpus evidence required
- W3b HyDE: not approved on current corpus — new corpus evidence required
- W4 LLM reranking: not approved on current corpus — new corpus/model required
- W2 hybrid retrieval: frozen — conditional reopen only ([#15](https://github.com/ai-agentopia/agentopia-super-rag/issues/15))

**Do not open a PR for any of the above without first establishing that the conditions for reopen are met.**
