"""Phase 1b labeled baseline evaluation runner (#318).

Runs actual retrieval against labeled relevance judgments to produce
authoritative nDCG@5, MRR, Precision@5, Recall@5 baseline.

This is the AUTHORITATIVE retrieval quality baseline.
- Phase 1a (RAGAS) = directional reference-free signal
- Phase 1b (this) = labeled metrics from judged data — the real gate

Usage:
    cd bot-config-api/src
    QDRANT_URL=http://localhost:6333 python -m evaluation.phase1b_baseline \
        --dataset evaluation/datasets/phase1b_labeled_seed.json \
        --output-dir evaluation/results

Wave gates:
    Wave 1: harness + seed dataset + successful baseline run
    Wave 2: sufficient pilot/client judgments (#307) for authoritative coverage
    Close gate: Wave 2 complete — not before
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from evaluation.retrieval_metrics import compute_query_metrics

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "evaluation/datasets/phase1b_labeled_seed.json"
DEFAULT_OUTPUT_DIR = "evaluation/results"
DEFAULT_K = 5


def load_labeled_dataset(path: str) -> dict:
    """Load and validate Phase 1b labeled dataset."""
    with open(path) as f:
        data = json.load(f)

    assert data.get("phase") == "1b", f"Dataset phase must be '1b', got '{data.get('phase')}'"
    assert "grading_scale" in data, "Dataset must define grading_scale"
    assert "samples" in data and len(data["samples"]) > 0, "Dataset must have samples"

    for sample in data["samples"]:
        assert "query" in sample, f"Sample {sample.get('id', '?')} missing 'query'"
        assert "scope" in sample, f"Sample {sample.get('id', '?')} missing 'scope'"
        assert "relevant_chunks" in sample, f"Sample {sample.get('id', '?')} missing 'relevant_chunks'"
        for chunk in sample["relevant_chunks"]:
            assert "source" in chunk, f"Chunk in {sample.get('id', '?')} missing 'source'"
            assert "chunk_index" in chunk, f"Chunk in {sample.get('id', '?')} missing 'chunk_index'"
            assert "relevance" in chunk and chunk["relevance"] in (0, 1, 2), \
                f"Chunk in {sample.get('id', '?')} has invalid relevance (must be 0, 1, or 2)"

    logger.info("Loaded %d labeled samples from %s (wave: %s)",
                len(data["samples"]), path, data.get("wave", "unknown"))
    return data


def _make_chunk_key(source: str, chunk_index: int) -> str:
    """Create a unique key for a chunk (source + chunk_index)."""
    return f"{source}::{chunk_index}"


def grade_retrieval_results(
    retrieved: list[dict[str, Any]],
    labeled_chunks: list[dict],
    k: int,
) -> dict:
    """Grade retrieved results against labeled relevance judgments.

    Args:
        retrieved: List of retrieval results, each with:
            - citation.source
            - citation.chunk_index
            - score
        labeled_chunks: List of labeled relevant chunks from dataset.
        k: Cutoff for @K metrics.

    Returns:
        Dict with relevances list, metrics, and grading details.
    """
    # Build lookup: chunk_key → relevance grade
    label_map = {}
    for chunk in labeled_chunks:
        key = _make_chunk_key(chunk["source"], chunk["chunk_index"])
        label_map[key] = chunk["relevance"]

    # Grade each retrieved result in rank order
    relevances = []
    grading_detail = []
    for rank, result in enumerate(retrieved[:k]):
        source = result.get("citation", {}).get("source", "")
        chunk_idx = result.get("citation", {}).get("chunk_index", -1)
        key = _make_chunk_key(source, chunk_idx)
        grade = label_map.get(key, 0)  # unjudged defaults to 0 (irrelevant)
        relevances.append(float(grade))
        grading_detail.append({
            "rank": rank + 1,
            "source": source,
            "chunk_index": chunk_idx,
            "retrieval_score": result.get("score", 0.0),
            "relevance_grade": grade,
            "judged": key in label_map,
        })

    # Total relevant items (grade >= 1) in labeled set — for recall
    total_relevant = sum(1 for c in labeled_chunks if c["relevance"] >= 1)

    metrics = compute_query_metrics(relevances, total_relevant, k)

    return {
        "relevances": relevances,
        "metrics": metrics,
        "grading_detail": grading_detail,
        "total_relevant_in_labels": total_relevant,
        "retrieved_count": len(retrieved),
    }


def run_baseline(
    dataset: dict,
    retrieval_adapter: Any,
    k: int = 5,
) -> dict:
    """Run Phase 1b baseline evaluation.

    For each labeled query:
    1. Execute actual retrieval (system under test)
    2. Grade results against labeled relevance judgments
    3. Compute nDCG@K, MRR, Precision@K, Recall@K

    Returns full baseline summary.
    """
    samples = dataset["samples"]
    per_query = []

    for i, sample in enumerate(samples):
        sample_id = sample.get("id", f"lb{i:02d}")
        query = sample["query"]
        scope = sample["scope"]

        logger.info("[%d/%d] %s — retrieving from '%s'...", i + 1, len(samples), sample_id, scope)

        # Step 1: ACTUAL RETRIEVAL — system under test
        try:
            results = retrieval_adapter.retrieve_raw(query, scope, k)
        except Exception as exc:
            logger.warning("Retrieval failed for %s: %s", sample_id, exc)
            results = []

        # Step 2: GRADE against labeled judgments
        grading = grade_retrieval_results(results, sample["relevant_chunks"], k)

        per_query.append({
            "id": sample_id,
            "query": query[:80],
            "scope": scope,
            "scenario": sample.get("scenario", ""),
            **grading,
        })

    return _build_baseline_summary(per_query, k, dataset)


def _build_baseline_summary(per_query: list[dict], k: int, dataset: dict) -> dict:
    """Aggregate per-query metrics into baseline summary."""
    metric_names = ["ndcg", "mrr", "precision", "recall"]
    aggregates = {}

    for metric in metric_names:
        values = [q["metrics"][metric] for q in per_query]
        if values:
            aggregates[metric] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }
        else:
            aggregates[metric] = {"mean": None, "count": 0}

    return {
        "per_query": per_query,
        "aggregates": aggregates,
        "k": k,
        "dataset_version": dataset.get("version", "unknown"),
        "dataset_wave": dataset.get("wave", "unknown"),
        "sample_count": len(per_query),
    }


def write_baseline_artifact(
    summary: dict,
    output_dir: str,
    dataset_path: str,
    adapter_type: str,
) -> tuple[str, str]:
    """Write baseline JSON + Markdown artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    artifact = {
        "phase": "1b",
        "type": "baseline",
        "timestamp": ts,
        "adapter": adapter_type,
        "dataset": dataset_path,
        "note": (
            "Phase 1b AUTHORITATIVE BASELINE. Metrics computed from graded relevance judgments. "
            "This baseline is the reference for Phase 2a improvement claims. "
            "Phase 1a (RAGAS) is directional only — this is the real gate."
        ),
        **summary,
    }

    json_path = os.path.join(output_dir, f"phase1b_baseline_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2)

    # Markdown
    agg = summary["aggregates"]
    md_lines = [
        f"# Phase 1b Baseline — {ts}",
        "",
        "> **AUTHORITATIVE BASELINE** — labeled retrieval metrics from graded relevance judgments.",
        "> Phase 2a (#319) must beat these numbers to pass its eval gate.",
        "> Phase 1a (RAGAS) is a separate directional signal — NOT this gate.",
        "",
        f"- **Dataset**: `{dataset_path}` (wave: {summary.get('dataset_wave', '?')})",
        f"- **Adapter**: `{adapter_type}`",
        f"- **Samples**: {summary['sample_count']}",
        f"- **K**: {summary['k']}",
        "",
        "## Baseline Metrics (aggregate)",
        "",
        f"| Metric | Mean | Min | Max |",
        f"|---|---|---|---|",
    ]
    for name in ["ndcg", "mrr", "precision", "recall"]:
        a = agg.get(name, {})
        mean = f"{a['mean']:.4f}" if a.get("mean") is not None else "N/A"
        mn = f"{a.get('min', 'N/A')}" if a.get("min") is not None else "N/A"
        mx = f"{a.get('max', 'N/A')}" if a.get("max") is not None else "N/A"
        label = {"ndcg": "nDCG@5", "mrr": "MRR", "precision": "Precision@5", "recall": "Recall@5"}[name]
        md_lines.append(f"| {label} | {mean} | {mn} | {mx} |")

    md_lines.extend(["", "## Per-Query Breakdown", "",
        "| ID | Scenario | nDCG@5 | MRR | P@5 | R@5 | Retrieved | Judged Relevant |",
        "|---|---|---|---|---|---|---|---|"])
    for q in summary["per_query"]:
        m = q["metrics"]
        md_lines.append(
            f"| {q['id']} | {q['scenario']} | {m['ndcg']:.4f} | {m['mrr']:.4f} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {q['retrieved_count']} | {q['total_relevant_in_labels']} |"
        )

    md_lines.extend(["", "---", "",
        "## Wave Gate Status", "",
        "- **Wave 1**: Harness + seed dataset + baseline run — COMPLETE",
        "- **Wave 2**: Pilot/client judged data (#307) — PENDING",
        "- **Close gate**: NOT PASSED — requires Wave 2",
        "",
        "*Generated by Phase 1b baseline runner (#318).*"])

    md_path = os.path.join(output_dir, f"phase1b_baseline_{ts}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path


# ── Retrieval adapter that returns raw results (with citation objects) ────────


class RawKnowledgeServiceAdapter:
    """Returns raw SearchResult dicts (not just text) for metric grading."""

    def __init__(self):
        from services.knowledge import get_knowledge_service
        self._svc = get_knowledge_service()
        if not self._svc._qdrant:
            raise RuntimeError("Requires QDRANT_URL and indexed data.")
        logger.info("RawKnowledgeServiceAdapter: using live KnowledgeService")

    def retrieve_raw(self, query: str, scope: str, limit: int) -> list[dict]:
        results = self._svc.search(query, scopes=[scope], limit=limit)
        return [
            {
                "text": r.text,
                "score": r.score,
                "citation": {
                    "source": r.citation.source,
                    "chunk_index": r.citation.chunk_index,
                    "section": r.citation.section,
                },
                "scope": r.scope,
            }
            for r in results
        ]


class InMemoryLabeledAdapter:
    """Returns results from in-memory KnowledgeService for e2e testing.

    Seed documents must be ingested before calling retrieve_raw.
    """

    def __init__(self, knowledge_service):
        self._svc = knowledge_service

    def retrieve_raw(self, query: str, scope: str, limit: int) -> list[dict]:
        results = self._svc.search(query, scopes=[scope], limit=limit)
        return [
            {
                "text": r.text,
                "score": r.score,
                "citation": {
                    "source": r.citation.source,
                    "chunk_index": r.citation.chunk_index,
                    "section": r.citation.section,
                },
                "scope": r.scope,
            }
            for r in results
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1b labeled baseline evaluation (#318)"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset = load_labeled_dataset(args.dataset)
    adapter = RawKnowledgeServiceAdapter()

    t0 = time.monotonic()
    summary = run_baseline(dataset, adapter, args.k)
    elapsed = time.monotonic() - t0

    json_path, md_path = write_baseline_artifact(
        summary, args.output_dir, args.dataset, "knowledge-service (in-process)",
    )

    logger.info("Baseline complete in %.1fs", elapsed)
    for name in ["ndcg", "mrr", "precision", "recall"]:
        a = summary["aggregates"].get(name, {})
        label = {"ndcg": "nDCG@5", "mrr": "MRR", "precision": "P@5", "recall": "R@5"}[name]
        logger.info("  %s: %s", label, f"{a['mean']:.4f}" if a.get("mean") is not None else "N/A")
    logger.info("Artifacts: %s, %s", json_path, md_path)
    logger.info("Wave 1: COMPLETE. Wave 2: PENDING (#307).")


if __name__ == "__main__":
    main()
