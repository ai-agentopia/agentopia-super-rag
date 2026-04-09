"""Phase 2a full Wave 1 comparison: dense-only vs hybrid.

Runs BOTH Phase 1a (RAGAS) AND Phase 1b (labeled metrics) for both modes.
Requires: OPENROUTER_API_KEY, QDRANT_URL env vars set.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY required"
assert os.getenv("QDRANT_URL"), "QDRANT_URL required"

from models.knowledge import DocumentFormat

DOCS = [
    ("The knowledge system supports four chunking strategies: FIXED_SIZE (default 512 tokens), PARAGRAPH, CODE_AWARE, and SEMANTIC (not yet implemented). IngestConfig controls chunk_size and chunk_overlap.", "knowledge-architecture.md"),
    ("Document ingestion uses SHA-256 hashing for deduplication. Two-phase atomic replace supersedes old records. ADR-012 defines the commit model.", "adr-012-ingestion.md"),
    ("Knowledge routes use dual-path authentication: operator session cookies and bot bearer tokens with X-Bot-Name header for bot relay auth.", "adr-010-runtime.md"),
    ("The gateway knowledge-retrieval plugin has maxContextTokens=3000 and confidenceThreshold=0.3 for filtering low-relevance results.", "gateway-plugin-config.md"),
    ("BotKnowledgeIndex is a singleton that rebuilds from K8s CRD annotations on startup for O(1) scope resolution via in-memory dict.", "bot-knowledge-index.md"),
]

SCOPE_DENSE = "eval_platform_docs_dense"
SCOPE_HYBRID = "eval_platform_docs_hybrid"

PHASE1A_QUERIES = [
    {"id": "s01", "user_input": "What chunking strategies does the knowledge system support?", "scenario": "factual_recall"},
    {"id": "s02", "user_input": "How does document deduplication work during ingestion?", "scenario": "process_explanation"},
    {"id": "s03", "user_input": "What authentication methods are used for knowledge routes?", "scenario": "security_architecture"},
]


class LiveRetrievalAdapter:
    def __init__(self, svc):
        self._svc = svc

    def retrieve(self, query, scope, limit):
        results = self._svc.search(query, scopes=[scope], limit=limit)
        return [r.text for r in results]

    def retrieve_raw(self, query, scope, limit):
        results = self._svc.search(query, scopes=[scope], limit=limit)
        return [{"text": r.text, "score": r.score, "citation": {"source": r.citation.source, "chunk_index": r.citation.chunk_index, "section": r.citation.section}, "scope": r.scope} for r in results]


def setup_mode(hybrid_enabled, scope):
    os.environ["HYBRID_SEARCH_ENABLED"] = "true" if hybrid_enabled else "false"
    for mod in list(sys.modules.keys()):
        if "services.knowledge" in mod:
            del sys.modules[mod]
    from services.knowledge import QdrantBackend, KnowledgeService
    backend = QdrantBackend(os.environ["QDRANT_URL"])
    try:
        backend.delete_collection(scope)
    except Exception:
        pass
    svc = KnowledgeService()
    svc._qdrant = backend
    for text, source in DOCS:
        svc.ingest(scope, text, source=source, format=DocumentFormat.MARKDOWN)
    return svc, backend


async def run_phase1a(svc, scope, mode_name, api_key, base_url):
    """Run Phase 1a RAGAS on 3 samples."""
    from evaluation.phase1a_runner import run_evaluation, write_artifacts
    adapter = LiveRetrievalAdapter(svc)
    samples = [{"id": q["id"], "user_input": q["user_input"], "scope": scope, "scenario": q["scenario"], "search_limit": 5} for q in PHASE1A_QUERIES]
    summary = await run_evaluation(
        samples, adapter, "google/gemini-2.0-flash-001", "google/gemini-2.0-flash-001",
        base_url, api_key,
    )
    json_path, md_path = write_artifacts(
        summary, "evaluation/results",
        "google/gemini-2.0-flash-001", "google/gemini-2.0-flash-001",
        f"phase1a_comparison_{mode_name}", len(samples),
        f"qdrant-live {mode_name}",
    )
    return summary, json_path


def run_phase1b(svc, scope, mode_name):
    """Run Phase 1b labeled baseline."""
    from evaluation.phase1b_baseline import load_labeled_dataset, run_baseline, write_baseline_artifact
    dataset = load_labeled_dataset("evaluation/datasets/phase1b_labeled_seed.json")
    for s in dataset["samples"]:
        s["scope"] = scope
    adapter = LiveRetrievalAdapter(svc)
    summary = run_baseline(dataset, adapter, k=5)
    json_path, md_path = write_baseline_artifact(
        summary, "evaluation/results", "phase1b_labeled_seed.json",
        f"qdrant-live {mode_name}",
    )
    return summary, json_path


async def main():
    api_key = os.environ["OPENROUTER_API_KEY"]
    base_url = "https://openrouter.ai/api/v1"

    # Dense-only
    print("=" * 60)
    print("DENSE-ONLY")
    print("=" * 60)
    svc_d, backend_d = setup_mode(False, SCOPE_DENSE)
    print(f"Ingested {len(DOCS)} docs into '{SCOPE_DENSE}'")

    p1b_dense, p1b_dense_art = run_phase1b(svc_d, SCOPE_DENSE, "DENSE-ONLY")
    print("Phase 1b (dense):")
    for n in ["ndcg", "mrr", "precision", "recall"]:
        print(f"  {n}: {p1b_dense['aggregates'][n]['mean']:.4f}")

    p1a_dense, p1a_dense_art = await run_phase1a(svc_d, SCOPE_DENSE, "dense-only", api_key, base_url)
    print("Phase 1a (dense):")
    for n in ["faithfulness", "context_utilization"]:
        v = p1a_dense["aggregates"].get(n, {}).get("mean")
        print(f"  {n}: {v:.4f}" if v is not None else f"  {n}: N/A")

    # Hybrid
    print()
    print("=" * 60)
    print("HYBRID (sparse TF + RRF)")
    print("=" * 60)
    svc_h, backend_h = setup_mode(True, SCOPE_HYBRID)
    print(f"Ingested {len(DOCS)} docs into '{SCOPE_HYBRID}'")

    p1b_hybrid, p1b_hybrid_art = run_phase1b(svc_h, SCOPE_HYBRID, "HYBRID")
    print("Phase 1b (hybrid):")
    for n in ["ndcg", "mrr", "precision", "recall"]:
        print(f"  {n}: {p1b_hybrid['aggregates'][n]['mean']:.4f}")

    p1a_hybrid, p1a_hybrid_art = await run_phase1a(svc_h, SCOPE_HYBRID, "hybrid", api_key, base_url)
    print("Phase 1a (hybrid):")
    for n in ["faithfulness", "context_utilization"]:
        v = p1a_hybrid["aggregates"].get(n, {}).get("mean")
        print(f"  {n}: {v:.4f}" if v is not None else f"  {n}: N/A")

    # Comparison
    print()
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print("\nPhase 1b (labeled, authoritative):")
    print(f"{'Metric':<15} {'Dense':<10} {'Hybrid':<10} {'Delta':<15}")
    for n in ["ndcg", "mrr", "precision", "recall"]:
        d = p1b_dense["aggregates"][n]["mean"]
        h = p1b_hybrid["aggregates"][n]["mean"]
        delta = h - d
        pct = (delta / d * 100) if d > 0 else 0
        label = {"ndcg": "nDCG@5", "mrr": "MRR", "precision": "P@5", "recall": "R@5"}[n]
        print(f"{label:<15} {d:<10.4f} {h:<10.4f} {delta:+.4f} ({pct:+.1f}%)")

    print("\nPhase 1a (RAGAS, directional only):")
    print(f"{'Metric':<25} {'Dense':<10} {'Hybrid':<10} {'Delta':<15}")
    for n in ["faithfulness", "context_utilization"]:
        d = p1a_dense["aggregates"].get(n, {}).get("mean")
        h = p1a_hybrid["aggregates"].get(n, {}).get("mean")
        if d is not None and h is not None:
            delta = h - d
            print(f"{n:<25} {d:<10.4f} {h:<10.4f} {delta:+.4f}")
        else:
            print(f"{n:<25} {'N/A':<10} {'N/A':<10}")

    # Cleanup
    print("\nCleanup:")
    try: backend_d.delete_collection(SCOPE_DENSE); print(f"  Deleted {SCOPE_DENSE}")
    except Exception as e: print(f"  {SCOPE_DENSE}: {e}")
    try: backend_h.delete_collection(SCOPE_HYBRID); print(f"  Deleted {SCOPE_HYBRID}")
    except Exception as e: print(f"  {SCOPE_HYBRID}: {e}")

    print(f"\nArtifacts:")
    print(f"  Phase 1b Dense:  {p1b_dense_art}")
    print(f"  Phase 1b Hybrid: {p1b_hybrid_art}")
    print(f"  Phase 1a Dense:  {p1a_dense_art}")
    print(f"  Phase 1a Hybrid: {p1a_hybrid_art}")


if __name__ == "__main__":
    asyncio.run(main())
