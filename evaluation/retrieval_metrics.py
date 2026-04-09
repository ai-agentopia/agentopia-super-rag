"""Labeled retrieval metrics for Phase 1b (#318).

Computes nDCG@K, MRR, Precision@K, Recall@K from graded relevance judgments.
These are the AUTHORITATIVE retrieval quality metrics — not reference-free.

Metric definitions:
- nDCG@K: Normalized Discounted Cumulative Gain. Measures ranking quality
  accounting for both relevance grade and position. Range [0, 1].
  DCG@K = sum(rel_i / log2(i+1) for i in 1..K)
  IDCG@K = DCG of ideal ranking (sorted by relevance desc)
  nDCG@K = DCG@K / IDCG@K

- MRR: Mean Reciprocal Rank. 1/rank of the first relevant result.
  Range [0, 1]. Relevant = relevance >= 1.

- Precision@K: Fraction of top-K results that are relevant (relevance >= 1).
  Range [0, 1].

- Recall@K: Fraction of all relevant items that appear in top-K results.
  Range [0, 1].
"""

import math


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at position K."""
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        # Position is 1-indexed: discount = log2(i + 2) since i starts at 0
        score += rel / math.log2(i + 2)
    return score


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized DCG@K. Returns 0.0 if no relevant items exist."""
    dcg = dcg_at_k(relevances, k)
    # Ideal: sort by relevance descending
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(relevances: list[float], threshold: float = 1.0) -> float:
    """Mean Reciprocal Rank. 1/rank of first result with relevance >= threshold."""
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevances: list[float], k: int, threshold: float = 1.0) -> float:
    """Fraction of top-K results with relevance >= threshold."""
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if r >= threshold)
    return relevant / len(top_k)


def recall_at_k(relevances: list[float], k: int, total_relevant: int, threshold: float = 1.0) -> float:
    """Fraction of all relevant items that appear in top-K."""
    if total_relevant == 0:
        return 0.0
    top_k = relevances[:k]
    found = sum(1 for r in top_k if r >= threshold)
    return found / total_relevant


def compute_query_metrics(relevances: list[float], total_relevant: int, k: int = 5) -> dict:
    """Compute all Phase 1b metrics for a single query's ranked results.

    Args:
        relevances: Graded relevance scores in retrieval rank order.
                    Each value is the relevance grade (0, 1, 2) of the result
                    at that rank position. Unjudged results default to 0.
        total_relevant: Total number of relevant items in the labeled set
                       (for recall computation).
        k: Cutoff for @K metrics.

    Returns:
        Dict with ndcg, mrr, precision, recall — all at K.
    """
    return {
        "ndcg": round(ndcg_at_k(relevances, k), 4),
        "mrr": round(mrr(relevances), 4),
        "precision": round(precision_at_k(relevances, k), 4),
        "recall": round(recall_at_k(relevances, k, total_relevant), 4),
    }
