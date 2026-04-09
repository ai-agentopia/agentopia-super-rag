"""W1 Real Pilot Gate: evaluate markdown-aware chunking on joblogic-kb/api-docs (#14).

Uses REAL documents from the production scope joblogic-kb/api-docs.
Documents are read from the local docs repo (same source used for ingestion).
Evaluation uses the Phase 1b grading harness with source-level relevance labels.

The comparison runs locally in-memory — it does NOT modify the live cluster.

Pilot scope: joblogic-kb/api-docs (10 documents, 258 chunks in production)
Dataset: evaluation/datasets/w1_real_pilot.json (10 labeled queries)

Gate criteria (from docs/evaluation.md):
  nDCG@5 must not regress more than 0.01 from baseline.

Usage:
    PYTHONPATH=src:. python evaluation/w1_real_pilot_gate.py --docs-root /path/to/docs/repo

    Default docs-root: /Users/thtn/Documents/Working/Codebase/personal/ai-agentopia/docs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.knowledge import ChunkingStrategy, DocumentFormat, IngestConfig
from services.knowledge import KnowledgeService, chunk_document
from evaluation.phase1b_baseline import (
    InMemoryLabeledAdapter,
    run_baseline,
)
from evaluation.retrieval_metrics import compute_query_metrics

DATASET_PATH = str(Path(__file__).parent / "datasets" / "w1_real_pilot.json")
SCOPE = "joblogic-kb/api-docs"
GATE_NDCG_MAX_REGRESSION = 0.01

# Map of source filename → path relative to docs repo root.
# These are the 10 documents currently ingested in the production scope.
SOURCE_MAP = {
    "chatbot-architecture.md": "architecture/chatbot-architecture.md",
    "overview.md": "architecture/overview.md",
    "a2a-full-design.md": "architecture/a2a-full-design.md",
    "architecture-multiple-bot.md": "architecture/architecture-multiple-bot.md",
    "governed-pr-review-workflow.md": "architecture/governed-pr-review-workflow.md",
    "reviewer-coexistence.md": "architecture/reviewer-coexistence.md",
    "production-sa-knowledge-base.md": "milestones/production-sa-knowledge-base.md",
    "production-super-rag.md": "milestones/production-super-rag.md",
    "p1-web-app-primary-dual-lane-mvp.md": "milestones/p1-web-app-primary-dual-lane-mvp.md",
    # ui-knowledge-workflow-trigger.md excluded — ingested directly, not in docs repo.
    # No labeled queries reference it. Pilot corpus is 9 documents.
}


def load_dataset():
    """Load and adapt the real pilot dataset for source-level grading."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    assert data.get("phase") == "1b"
    assert "samples" in data and len(data["samples"]) > 0

    # Convert source-level labels to chunk-level format expected by Phase 1b grader.
    # We use chunk_index=-1 as a sentinel meaning "any chunk from this source".
    # The grading adapter below handles this.
    return data


def load_documents(docs_root: str) -> list[tuple[str, str]]:
    """Load real documents from the docs repo."""
    docs = []
    missing = []
    for source, rel_path in SOURCE_MAP.items():
        if rel_path is None:
            # Try common locations
            for try_path in [
                os.path.join(docs_root, "architecture", source),
                os.path.join(docs_root, "milestones", source),
                os.path.join(docs_root, "operations", source),
                os.path.join(docs_root, "product", source),
            ]:
                if os.path.exists(try_path):
                    rel_path = os.path.relpath(try_path, docs_root)
                    break

        if rel_path:
            full_path = os.path.join(docs_root, rel_path)
            if os.path.exists(full_path):
                with open(full_path) as f:
                    content = f.read()
                docs.append((source, content))
            else:
                missing.append(source)
        else:
            missing.append(source)

    if missing:
        print(f"WARNING: {len(missing)} source documents not found locally: {missing}")
        print("  These documents are in Qdrant but not in the local docs repo.")
        print("  Evaluation will use the documents that ARE available.")

    return docs


class SourceLevelAdapter:
    """Adapter that grades retrieval results by source document match.

    Since chunk_index differs between chunking strategies, we grade at
    the source level: if the retrieved chunk comes from a relevant source,
    it gets that source's relevance grade.
    """

    def __init__(self, svc: KnowledgeService):
        self._svc = svc

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


def grade_source_level(retrieved, relevant_sources, k):
    """Grade retrieved results using source-level relevance labels."""
    source_grades = {}
    for rs in relevant_sources:
        source_grades[rs["source"]] = rs["relevance"]

    relevances = []
    detail = []
    for rank, result in enumerate(retrieved[:k]):
        source = result.get("citation", {}).get("source", "")
        grade = source_grades.get(source, 0)
        relevances.append(float(grade))
        detail.append({
            "rank": rank + 1,
            "source": source,
            "chunk_index": result.get("citation", {}).get("chunk_index", -1),
            "section": result.get("citation", {}).get("section", ""),
            "retrieval_score": result.get("score", 0.0),
            "relevance_grade": grade,
        })

    # Compute only valid metrics for source-level grading.
    # Recall@5 is invalid: denominator = source count, numerator = chunk count → can exceed 1.0.
    # nDCG@5, MRR, P@5 are valid because they only use the relevance list.
    from evaluation.retrieval_metrics import ndcg_at_k, mrr as compute_mrr, precision_at_k
    metrics = {
        "ndcg": round(ndcg_at_k(relevances, k), 4),
        "mrr": round(compute_mrr(relevances), 4),
        "precision": round(precision_at_k(relevances, k), 4),
    }

    return {
        "relevances": relevances,
        "metrics": metrics,
        "grading_detail": detail,
        "retrieved_count": len(retrieved),
    }


def run_strategy(docs, dataset, strategy, chunk_size=512):
    """Ingest docs with strategy, run evaluation, return summary."""
    svc = KnowledgeService()
    config = IngestConfig(chunking_strategy=strategy, chunk_size=chunk_size)

    total_chunks = 0
    for source, content in docs:
        result = svc.ingest(SCOPE, content, source, DocumentFormat.MARKDOWN, config=config)
        total_chunks += result.chunks_created

    adapter = SourceLevelAdapter(svc)
    k = 5
    per_query = []

    for sample in dataset["samples"]:
        results = adapter.retrieve_raw(sample["query"], sample["scope"], k)
        grading = grade_source_level(results, sample["relevant_sources"], k)
        per_query.append({
            "id": sample["id"],
            "query": sample["query"][:80],
            "scenario": sample.get("scenario", ""),
            **grading,
        })

    # Aggregate (only valid metrics — no recall)
    aggregates = {}
    for metric in ["ndcg", "mrr", "precision"]:
        values = [q["metrics"][metric] for q in per_query]
        aggregates[metric] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
            "count": len(values),
        }

    return {
        "strategy": strategy.value,
        "total_chunks": total_chunks,
        "total_documents": len(docs),
        "per_query": per_query,
        "aggregates": aggregates,
    }


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="W1 Real Pilot Gate (#14)")
    parser.add_argument(
        "--docs-root",
        default="/Users/thtn/Documents/Working/Codebase/personal/ai-agentopia/docs",
        help="Path to the docs repo root",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("W1 REAL PILOT GATE — joblogic-kb/api-docs")
    print("=" * 72)
    print(f"Scope: {SCOPE}")
    print(f"Docs root: {args.docs_root}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Gate: nDCG@5 regression <= {GATE_NDCG_MAX_REGRESSION}")
    print()

    # Load real documents
    docs = load_documents(args.docs_root)
    print(f"Loaded {len(docs)} real documents from docs repo")
    for source, content in docs:
        headings = sum(1 for line in content.split("\n") if line.strip().startswith("#"))
        print(f"  {source}: {len(content)} chars, {headings} headings")
    print()

    # Load dataset
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset['samples'])} labeled queries")
    print()

    # Run both strategies
    t0 = time.time()
    fs_result = run_strategy(docs, dataset, ChunkingStrategy.FIXED_SIZE)
    md_result = run_strategy(docs, dataset, ChunkingStrategy.MARKDOWN_AWARE)
    elapsed = time.time() - t0

    # Print per-query breakdown
    for label, result in [("FIXED_SIZE (baseline)", fs_result), ("MARKDOWN_AWARE", md_result)]:
        print(f"--- {label} ({result['total_chunks']} chunks from {result['total_documents']} docs) ---")
        for q in result["per_query"]:
            m = q["metrics"]
            top_src = q["grading_detail"][0]["source"] if q["grading_detail"] else "?"
            top_sec = q["grading_detail"][0].get("section", "") if q["grading_detail"] else ""
            print(
                f"  {q['id']} ({q['scenario']:<25s}) "
                f"nDCG={m['ndcg']:.4f}  MRR={m['mrr']:.4f}  "
                f"P@5={m['precision']:.4f}  "
                f"top1={top_src}:{top_sec[:20]}"
            )
        print()

    # Aggregate comparison
    print("-" * 72)
    print(f"{'Metric':<15} {'FIXED_SIZE':>12} {'MD_AWARE':>12} {'Delta':>10}")
    print("-" * 72)

    deltas = {}
    for metric in ["ndcg", "mrr", "precision"]:
        fs_val = fs_result["aggregates"][metric]["mean"]
        md_val = md_result["aggregates"][metric]["mean"]
        delta = md_val - fs_val if fs_val is not None and md_val is not None else None
        deltas[metric] = delta
        label = {"ndcg": "nDCG@5", "mrr": "MRR", "precision": "P@5"}[metric]
        d = f"{delta:+.4f}" if delta is not None else "N/A"
        print(f"{label:<15} {fs_val:>12.4f} {md_val:>12.4f} {d:>10}")

    print(f"\nChunks: fixed_size={fs_result['total_chunks']}, markdown_aware={md_result['total_chunks']}")
    print(f"Elapsed: {elapsed*1000:.0f}ms")

    # Gate decision
    ndcg_delta = deltas.get("ndcg")
    if ndcg_delta is None:
        gate_passed = False
        reason = "nDCG@5 could not be computed"
    elif ndcg_delta >= -GATE_NDCG_MAX_REGRESSION:
        gate_passed = True
        if ndcg_delta > 0:
            reason = f"nDCG@5 improved by {ndcg_delta:+.4f} (no regression)"
        elif ndcg_delta == 0:
            reason = "nDCG@5 identical (no regression)"
        else:
            reason = f"nDCG@5 regressed {abs(ndcg_delta):.4f} — within tolerance ({GATE_NDCG_MAX_REGRESSION})"
    else:
        gate_passed = False
        reason = f"nDCG@5 regressed {abs(ndcg_delta):.4f} — exceeds tolerance ({GATE_NDCG_MAX_REGRESSION})"

    print(f"\nGATE: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"Reason: {reason}")

    # Save results
    output = {
        "evaluation": "w1_real_pilot_gate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pilot_scope": SCOPE,
        "dataset": "w1_real_pilot.json",
        "dataset_queries": len(dataset["samples"]),
        "documents_loaded": len(docs),
        "label_status": dataset.get("labeling_status", "unknown"),
        "gate_criteria": f"nDCG@5 regression <= {GATE_NDCG_MAX_REGRESSION}",
        "gate_passed": gate_passed,
        "gate_reason": reason,
        "baseline": {
            "strategy": "fixed_size",
            "total_chunks": fs_result["total_chunks"],
            "aggregates": fs_result["aggregates"],
            "per_query": [{"id": q["id"], "scenario": q["scenario"], "metrics": q["metrics"]} for q in fs_result["per_query"]],
        },
        "candidate": {
            "strategy": "markdown_aware",
            "total_chunks": md_result["total_chunks"],
            "aggregates": md_result["aggregates"],
            "per_query": [{"id": q["id"], "scenario": q["scenario"], "metrics": q["metrics"]} for q in md_result["per_query"]],
        },
        "metric_model": "nDCG@5 and MRR authoritative. P@5 directional. Recall@5 excluded (invalid with source-level grading).",
        "deltas": {
            "nDCG@5": round(deltas.get("ndcg", 0), 4),
            "MRR": round(deltas.get("mrr", 0), 4),
            "P@5": round(deltas.get("precision", 0), 4),
        },
    }

    output_path = Path(__file__).parent / "results" / "w1_real_pilot_gate.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
