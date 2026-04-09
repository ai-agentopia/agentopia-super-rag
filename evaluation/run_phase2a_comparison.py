"""Phase 2a real comparison: dense-only vs hybrid on live Qdrant.

Requires: OPENROUTER_API_KEY, QDRANT_URL env vars set.
Creates isolated eval collections, ingests, runs Phase 1b baseline on both modes.
"""

import json
import os
import sys
import time

# Force env before imports
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


class LiveAdapter:
    def __init__(self, svc):
        self._svc = svc

    def retrieve_raw(self, query, scope, limit):
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


def run_mode(mode_name, hybrid_enabled, scope):
    os.environ["HYBRID_SEARCH_ENABLED"] = "true" if hybrid_enabled else "false"

    # Fresh imports to pick up env change
    for mod in list(sys.modules.keys()):
        if "services.knowledge" in mod:
            del sys.modules[mod]

    from services.knowledge import QdrantBackend, KnowledgeService

    print(f"\n{'='*60}")
    print(f"MODE: {mode_name}")
    print(f"{'='*60}")

    backend = QdrantBackend(os.environ["QDRANT_URL"])

    # Clean + create
    try:
        backend.delete_collection(scope)
    except Exception:
        pass

    svc = KnowledgeService()
    svc._qdrant = backend

    # Ingest
    for text, source in DOCS:
        svc.ingest(scope, text, source=source, format=DocumentFormat.MARKDOWN)
    print(f"Ingested {len(DOCS)} docs into '{scope}'")

    # Load dataset + patch scopes
    from evaluation.phase1b_baseline import load_labeled_dataset, run_baseline, write_baseline_artifact

    dataset = load_labeled_dataset("evaluation/datasets/phase1b_labeled_seed.json")
    for s in dataset["samples"]:
        s["scope"] = scope

    # Run baseline
    t0 = time.monotonic()
    summary = run_baseline(dataset, LiveAdapter(svc), k=5)
    elapsed = time.monotonic() - t0

    # Write artifact
    json_path, md_path = write_baseline_artifact(
        summary, "evaluation/results", "phase1b_labeled_seed.json",
        f"qdrant-live {mode_name}",
    )

    print(f"Completed in {elapsed:.1f}s")
    for name in ["ndcg", "mrr", "precision", "recall"]:
        a = summary["aggregates"][name]
        label = {"ndcg": "nDCG@5", "mrr": "MRR", "precision": "P@5", "recall": "R@5"}[name]
        print(f"  {label}: {a['mean']:.4f}")
    print(f"Artifact: {json_path}")

    return summary, json_path, backend


if __name__ == "__main__":
    summary_d, art_d, backend_d = run_mode("DENSE-ONLY", False, SCOPE_DENSE)
    summary_h, art_h, backend_h = run_mode("HYBRID (sparse TF + RRF)", True, SCOPE_HYBRID)

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: Dense-Only vs Hybrid (Sparse TF + RRF)")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Dense':<10} {'Hybrid':<10} {'Delta':<15}")
    for name in ["ndcg", "mrr", "precision", "recall"]:
        d_val = summary_d["aggregates"][name]["mean"]
        h_val = summary_h["aggregates"][name]["mean"]
        delta = h_val - d_val
        pct = (delta / d_val * 100) if d_val > 0 else 0
        label = {"ndcg": "nDCG@5", "mrr": "MRR", "precision": "P@5", "recall": "R@5"}[name]
        print(f"{label:<15} {d_val:<10.4f} {h_val:<10.4f} {delta:+.4f} ({pct:+.1f}%)")

    # Cleanup
    print("\nCleanup:")
    try:
        backend_d.delete_collection(SCOPE_DENSE)
        print(f"  Deleted {SCOPE_DENSE}")
    except Exception as e:
        print(f"  {SCOPE_DENSE}: {e}")
    try:
        backend_h.delete_collection(SCOPE_HYBRID)
        print(f"  Deleted {SCOPE_HYBRID}")
    except Exception as e:
        print(f"  {SCOPE_HYBRID}: {e}")

    print(f"\nArtifacts:")
    print(f"  Dense:  {art_d}")
    print(f"  Hybrid: {art_h}")
