"""Phase 1a evaluation — RAGAS early signal for SA-KB / Super RAG.

This is a DIRECTIONAL SIGNAL ONLY. It does NOT produce:
- nDCG, MRR, or Precision@K (those require Phase 1b labeled data, #318)
- Authoritative retrieval-ranking gates

Metrics: Faithfulness, Answer Relevancy, Context Utilization (all reference-free).
"""
