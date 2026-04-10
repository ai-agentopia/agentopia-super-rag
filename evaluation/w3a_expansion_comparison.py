"""W3a Evaluation: Query expansion impact on retrieval quality (#16).

Compares baseline (dense-only) vs expansion-augmented search on the
real pilot scope (joblogic-kb/api-docs) using simulated expansions.

Since this evaluation runs without a live LLM, expansions are generated
by simple synonym/rephrasing rules. This measures the RRF merge mechanism
and multi-query retrieval benefit, not the LLM phrasing quality.

For live LLM quality measurement, run with OPENROUTER_API_KEY set.

Usage:
    PYTHONPATH=src:. python evaluation/w3a_expansion_comparison.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.knowledge import ChunkingStrategy, DocumentFormat, IngestConfig
from services.knowledge import KnowledgeService
from services.query_expansion import rrf_merge
from evaluation.w1_real_pilot_gate import load_documents, SCOPE

DATASET_PATH = str(Path(__file__).parent / "datasets" / "w1_real_pilot.json")

# Simulated expansions: manual alternative phrasings for each pilot query
SIMULATED_EXPANSIONS = {
    "What are the core services in the Agentopia platform architecture?":
        ["List the main services in Agentopia", "What components make up the platform", "Agentopia service architecture"],
    "How does bot provisioning work and how are credentials secured?":
        ["Bot creation pipeline and token security", "How are new bots deployed and authenticated", "Credential management for bot provisioning"],
    "What is the memory architecture and how does scope-based isolation work?":
        ["Memory layers and scope isolation design", "How do bots share or isolate memory", "Scope-based memory separation architecture"],
    "How does the governed PR review workflow operate end-to-end?":
        ["PR review governance workflow steps", "How does the automated code review process work", "End-to-end governed review pipeline"],
    "What is the dual-lane interaction model and what are the execution boundaries?":
        ["Communication lane vs workflow lane separation", "Execution boundary model for agent interaction", "Dual-lane architecture boundaries"],
    "How do agents communicate using the A2A thread model?":
        ["Agent-to-agent thread communication protocol", "A2A thread model for multi-agent conversation", "How does the thread state machine work"],
    "What is the SA Knowledge Base and how does it inject context at inference time?":
        ["Knowledge retrieval injection at inference", "How does the gateway plugin inject domain knowledge", "SA bot knowledge context injection pipeline"],
    "What are the production acceptance criteria for the SA Knowledge Base?":
        ["SA Knowledge Base go-live criteria", "Production readiness checklist for knowledge system", "Acceptance criteria for knowledge base deployment"],
    "What human-in-the-loop controls exist in the A2A collaboration system?":
        ["Checkpoint and approval mechanisms in agent collaboration", "Human approval flow in A2A threads", "Human-in-the-loop governance for multi-agent"],
    "What is the reviewer bot's scope of permissions and what can it not do?":
        ["Reviewer bot permission boundaries and limitations", "What the review bot cannot do", "Scoped permissions for automated reviewer"],
}


def run_comparison():
    """Run baseline vs expansion-augmented comparison."""
    docs = load_documents("/Users/thtn/Documents/Working/Codebase/personal/ai-agentopia/docs")
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Ingest with markdown-aware (current production config)
    svc = KnowledgeService()
    config = IngestConfig(chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE, chunk_size=512)
    for source, content in docs:
        svc.ingest(SCOPE, content, source, DocumentFormat.MARKDOWN, config=config)

    from evaluation.w1_real_pilot_gate import grade_source_level
    from evaluation.retrieval_metrics import ndcg_at_k, mrr as compute_mrr, precision_at_k

    baseline_metrics = []
    expanded_metrics = []

    for sample in dataset["samples"]:
        query = sample["query"]
        scope = sample["scope"]

        # Baseline: single query
        baseline_results = svc.search(query, [scope], limit=5)
        baseline_raw = [
            {"text": r.text, "score": r.score, "scope": r.scope,
             "citation": {"source": r.citation.source, "chunk_index": r.citation.chunk_index, "section": r.citation.section}}
            for r in baseline_results
        ]
        baseline_grading = grade_source_level(baseline_raw, sample["relevant_sources"], 5)
        baseline_metrics.append(baseline_grading["metrics"])

        # Expanded: original + 3 simulated expansions, merged via RRF
        expansions = SIMULATED_EXPANSIONS.get(query, [])
        all_queries = [query] + expansions[:3]
        ranked_lists = []
        for q in all_queries:
            results_q = svc.search(q, [scope], limit=5)
            ranked_lists.append([
                {"text": r.text, "score": r.score, "scope": r.scope,
                 "citation": {"source": r.citation.source, "chunk_index": r.citation.chunk_index, "section": r.citation.section}}
                for r in results_q
            ])

        merged = rrf_merge(ranked_lists, limit=5)
        expanded_grading = grade_source_level(merged, sample["relevant_sources"], 5)
        expanded_metrics.append(expanded_grading["metrics"])

    return baseline_metrics, expanded_metrics, dataset


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("=" * 72)
    print("W3a EVALUATION — Query Expansion (simulated expansions)")
    print("=" * 72)
    print(f"Scope: {SCOPE}")
    print(f"Dataset: {DATASET_PATH}")
    print()

    t0 = time.time()
    baseline, expanded, dataset = run_comparison()
    elapsed = time.time() - t0

    # Per-query comparison
    print(f"{'ID':<8} {'Scenario':<28} {'BL nDCG':>8} {'EX nDCG':>8} {'Delta':>8}")
    print("-" * 65)
    for i, sample in enumerate(dataset["samples"]):
        bl = baseline[i]
        ex = expanded[i]
        delta = ex["ndcg"] - bl["ndcg"]
        marker = " <<<" if abs(delta) > 0.01 else ""
        print(f"{sample['id']:<8} {sample['scenario']:<28} {bl['ndcg']:>8.4f} {ex['ndcg']:>8.4f} {delta:>+8.4f}{marker}")

    # Aggregate
    def avg(metrics, key):
        vals = [m[key] for m in metrics]
        return sum(vals) / len(vals) if vals else 0

    print()
    print("-" * 65)
    print(f"{'Metric':<15} {'Baseline':>12} {'Expanded':>12} {'Delta':>10}")
    print("-" * 65)
    for metric in ["ndcg", "mrr", "precision"]:
        bl_avg = avg(baseline, metric)
        ex_avg = avg(expanded, metric)
        delta = ex_avg - bl_avg
        label = {"ndcg": "nDCG@5", "mrr": "MRR", "precision": "P@5"}[metric]
        print(f"{label:<15} {bl_avg:>12.4f} {ex_avg:>12.4f} {delta:>+10.4f}")

    print(f"\nElapsed: {elapsed*1000:.0f}ms")

    # Cost/latency note
    print()
    print("=" * 72)
    print("COST / LATENCY IMPACT")
    print("=" * 72)
    print("Per query with expansion enabled:")
    print("  - 1 LLM call to generate 3 alternative phrasings (~300-800ms)")
    print("  - 4 retrieval queries instead of 1 (original + 3 expansions)")
    print("  - 4 embedding calls instead of 1 (if Qdrant embedding-at-query)")
    print("  - RRF merge: negligible (<1ms)")
    print("  - Estimated total latency addition: +300-1000ms per search")
    print("  - Estimated cost: 1 LLM completion (~0.1-0.5 cents) + 3x embedding")
    print()
    print("NOTE: This evaluation used simulated expansions (no LLM call).")
    print("Live latency measurement requires OPENROUTER_API_KEY.")

    # Save results
    output = {
        "evaluation": "w3a_expansion_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": SCOPE,
        "expansion_type": "simulated (manual phrasings)",
        "queries": len(dataset["samples"]),
        "baseline_ndcg_mean": round(avg(baseline, "ndcg"), 4),
        "expanded_ndcg_mean": round(avg(expanded, "ndcg"), 4),
        "ndcg_delta": round(avg(expanded, "ndcg") - avg(baseline, "ndcg"), 4),
        "baseline_mrr_mean": round(avg(baseline, "mrr"), 4),
        "expanded_mrr_mean": round(avg(expanded, "mrr"), 4),
        "cost_note": "1 LLM call + 3x embedding per query when enabled",
        "latency_note": "Estimated +300-1000ms per query (LLM + 3 extra retrievals)",
    }

    output_path = Path(__file__).parent / "results" / "w3a_expansion_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
