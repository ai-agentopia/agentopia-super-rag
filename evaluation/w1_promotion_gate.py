"""W1 Promotion Gate: markdown-aware vs fixed-size (#14).

Runs TWO evaluations using the Phase 1b harness:

1. REGRESSION CHECK on phase1b_labeled_seed.json (test/platform-docs)
   - Plain-text seed docs. Both strategies produce identical chunks (fallback).
   - Proves markdown-aware does NOT regress existing behavior.

2. W1 PILOT on w1_markdown_pilot.json (w1-pilot/docs)
   - Markdown-structured docs with headings and sections.
   - Tests whether heading-aware chunking improves retrieval precision.
   - Labels graded per-chunk (chunk_index labels match markdown-aware output).

Gate criteria (from docs/evaluation.md):
  nDCG@5 must not regress more than 0.01 from baseline.

Usage:
    PYTHONPATH=src:. python evaluation/w1_promotion_gate.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.knowledge import ChunkingStrategy, DocumentFormat, IngestConfig
from services.knowledge import KnowledgeService
from evaluation.phase1b_baseline import (
    InMemoryLabeledAdapter,
    load_labeled_dataset,
    run_baseline,
)

SEED_DATASET = str(Path(__file__).parent / "datasets" / "phase1b_labeled_seed.json")
PILOT_DATASET = str(Path(__file__).parent / "datasets" / "w1_markdown_pilot.json")
GATE_NDCG_MAX_REGRESSION = 0.01


# ── Seed documents for test/platform-docs (Phase 1b seed) ─────────────

SEED_SCOPE = "test/platform-docs"
SEED_DOCS = [
    ("knowledge-architecture.md",
     "The knowledge system supports four chunking strategies: "
     "FIXED_SIZE (default, 512 tokens), PARAGRAPH, CODE_AWARE, and SEMANTIC (not yet implemented). "
     "IngestConfig controls chunk_size (64-4096) and chunk_overlap (0-512)."),
    ("adr-012-ingestion.md",
     "Document ingestion uses SHA-256 hashing for deduplication. "
     "Two-phase atomic replace supersedes old records. "
     "ADR-012 defines the two-phase replace model for document lifecycle."),
    ("adr-010-runtime.md",
     "Knowledge routes use dual-path authentication: operator session cookies "
     "and bot bearer tokens with X-Bot-Name header."),
    ("gateway-plugin-config.md",
     "The gateway knowledge-retrieval plugin has maxContextTokens=3000 "
     "and confidenceThreshold=0.3 for filtering low-relevance results."),
    ("bot-knowledge-index.md",
     "BotKnowledgeIndex is a singleton that rebuilds from K8s CRD annotations "
     "on startup, maintaining an in-memory dict for O(1) scope resolution."),
]

# ── Markdown-structured docs for w1-pilot/docs ────────────────────────

PILOT_SCOPE = "w1-pilot/docs"
PILOT_DOCS = [
    ("api-reference.md", """# Knowledge API Reference

The Knowledge API provides document ingestion and semantic search for Agentopia bots.

## Authentication

Two authentication paths are supported:

- **Operator path**: Uses X-Internal-Token header for bot-config-api proxy calls.
- **Bot path**: Uses Authorization Bearer token with X-Bot-Name header for direct gateway calls.

Operator tokens are configured via the KNOWLEDGE_API_INTERNAL_TOKEN environment variable.

## Ingestion Pipeline

Documents are ingested through a two-phase atomic pipeline:

Phase 1 (Prepare): Compute SHA-256 hash, check for duplicates, split into chunks, deduplicate within scope.
Phase 2 (Commit): Create lifecycle record in Postgres, delete old chunks, embed via OpenRouter, upsert to Qdrant.

## Search

Query the knowledge base with semantic search. Results include citations with source, section, and score.

## Error Handling

All endpoints return standard HTTP status codes: 200 success, 201 created, 400 bad request, 401 unauthorized, 403 forbidden, 404 not found, 422 validation error."""),

    ("deployment-guide.md", """# Deployment Guide

This guide covers deploying knowledge-api to Kubernetes via ArgoCD.

## Prerequisites

Kubernetes cluster, ArgoCD, Qdrant vector database, PostgreSQL 16+, and Vault for secret management.

## Helm Chart Configuration

The knowledge-api Helm chart is part of the agentopia-base chart. Configure image repository, tag, replicas, resources, and environment variables in values.yaml.

## ArgoCD Image Updater

Images are automatically updated by ArgoCD Image Updater using regexp tag patterns and newest-build strategy.

## Health Checks

The service exposes two health endpoints: GET /health for liveness and GET /internal/health for readiness (checks Qdrant and Postgres connectivity).

## Troubleshooting

Common issues include Qdrant connection refused, embedding timeout, OOM on large documents, and stale search results from ArgoCD sync delays."""),

    ("adr-011.md", """# ADR-011: Document Provenance Tracking

## Status

Accepted (2026-03-28).

## Decision

Every ingested document and its chunks carry two provenance fields: document_hash (SHA-256 of original content, computed before chunking) and ingested_at (Unix timestamp set once at ingestion time). If the same content is re-ingested, the hash matches and the operation is skipped.

## Consequences

Every chunk carries 64 extra bytes of metadata. Duplicate detection is O(1) hash comparison. Operators can query staleness via ingested_at comparison."""),
]


def run_eval(scope, docs, dataset_path, strategy):
    """Ingest docs with strategy, run Phase 1b evaluation, return summary."""
    svc = KnowledgeService()
    config = IngestConfig(chunking_strategy=strategy, chunk_size=512)
    for source, content in docs:
        svc.ingest(scope, content, source, DocumentFormat.MARKDOWN, config=config)

    dataset = load_labeled_dataset(dataset_path)
    adapter = InMemoryLabeledAdapter(svc)
    return run_baseline(dataset, adapter, k=5), dataset


def print_comparison(name, fs_summary, md_summary):
    """Print formatted comparison."""
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")

    for label, summary in [("FIXED_SIZE", fs_summary), ("MARKDOWN_AWARE", md_summary)]:
        print(f"\n--- {label} ---")
        for q in summary["per_query"]:
            m = q["metrics"]
            print(f"  {q['id']} ({q['scenario']:<25s}) "
                  f"nDCG={m['ndcg']:.4f}  MRR={m['mrr']:.4f}  "
                  f"P@5={m['precision']:.4f}  R@5={m['recall']:.4f}")

    print(f"\n{'-'*72}")
    print(f"{'Metric':<15} {'FIXED_SIZE':>12} {'MD_AWARE':>12} {'Delta':>10}")
    print(f"{'-'*72}")

    deltas = {}
    for metric in ["ndcg", "mrr", "precision", "recall"]:
        fs_val = fs_summary["aggregates"][metric]["mean"]
        md_val = md_summary["aggregates"][metric]["mean"]
        delta = md_val - fs_val if fs_val is not None and md_val is not None else None
        deltas[metric] = delta
        label = {"ndcg": "nDCG@5", "mrr": "MRR", "precision": "P@5", "recall": "R@5"}[metric]
        d = f"{delta:+.4f}" if delta is not None else "N/A"
        print(f"{label:<15} {fs_val:>12.4f} {md_val:>12.4f} {d:>10}")

    return deltas


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("W1 PROMOTION GATE — Full Phase 1b evaluation (#14)")
    print(f"Gate: nDCG@5 regression <= {GATE_NDCG_MAX_REGRESSION}")

    t0 = time.time()

    # ── Part 1: Regression check on seed dataset ────────────────────
    fs_seed, _ = run_eval(SEED_SCOPE, SEED_DOCS, SEED_DATASET, ChunkingStrategy.FIXED_SIZE)
    md_seed, _ = run_eval(SEED_SCOPE, SEED_DOCS, SEED_DATASET, ChunkingStrategy.MARKDOWN_AWARE)
    seed_deltas = print_comparison("Part 1: REGRESSION CHECK (phase1b_labeled_seed.json)", fs_seed, md_seed)

    # ── Part 2: W1 Pilot on markdown-structured docs ────────────────
    fs_pilot, _ = run_eval(PILOT_SCOPE, PILOT_DOCS, PILOT_DATASET, ChunkingStrategy.FIXED_SIZE)
    md_pilot, pilot_ds = run_eval(PILOT_SCOPE, PILOT_DOCS, PILOT_DATASET, ChunkingStrategy.MARKDOWN_AWARE)
    pilot_deltas = print_comparison("Part 2: W1 PILOT (w1_markdown_pilot.json)", fs_pilot, md_pilot)

    elapsed = time.time() - t0

    # ── Gate decision ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  GATE DECISION")
    print(f"{'='*72}")

    seed_ndcg = seed_deltas.get("ndcg", 0)
    pilot_ndcg = pilot_deltas.get("ndcg", 0)

    seed_pass = seed_ndcg is not None and seed_ndcg >= -GATE_NDCG_MAX_REGRESSION
    pilot_pass = pilot_ndcg is not None and pilot_ndcg >= -GATE_NDCG_MAX_REGRESSION

    print(f"\n  Regression check (seed):  nDCG@5 delta = {seed_ndcg:+.4f}  → {'PASS' if seed_pass else 'FAIL'}")
    print(f"  W1 pilot (markdown docs): nDCG@5 delta = {pilot_ndcg:+.4f}  → {'PASS' if pilot_pass else 'FAIL'}")

    gate_passed = seed_pass and pilot_pass
    if gate_passed:
        if pilot_ndcg > 0:
            reason = f"No regression on seed (delta={seed_ndcg:+.4f}). Improvement on markdown pilot (delta={pilot_ndcg:+.4f})."
        elif pilot_ndcg == 0:
            reason = f"No regression on either dataset. Parity on markdown pilot."
        else:
            reason = f"No regression on seed. Pilot regression {abs(pilot_ndcg):.4f} within tolerance."
    else:
        parts = []
        if not seed_pass:
            parts.append(f"Seed regression {abs(seed_ndcg):.4f} exceeds tolerance")
        if not pilot_pass:
            parts.append(f"Pilot regression {abs(pilot_ndcg):.4f} exceeds tolerance")
        reason = ". ".join(parts) + "."

    print(f"\n  GATE: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"  Reason: {reason}")
    print(f"  Elapsed: {elapsed*1000:.0f}ms")

    # ── Save results ────────────────────────────────────────────────
    output = {
        "evaluation": "w1_promotion_gate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_criteria": f"nDCG@5 regression <= {GATE_NDCG_MAX_REGRESSION}",
        "gate_passed": gate_passed,
        "gate_reason": reason,
        "regression_check": {
            "scope": SEED_SCOPE,
            "dataset": "phase1b_labeled_seed.json",
            "queries": fs_seed["sample_count"],
            "baseline_ndcg": fs_seed["aggregates"]["ndcg"]["mean"],
            "candidate_ndcg": md_seed["aggregates"]["ndcg"]["mean"],
            "delta": round(seed_ndcg, 4) if seed_ndcg is not None else None,
            "passed": seed_pass,
        },
        "w1_pilot": {
            "scope": PILOT_SCOPE,
            "dataset": "w1_markdown_pilot.json",
            "queries": fs_pilot["sample_count"],
            "baseline": {
                "strategy": "fixed_size",
                "aggregates": fs_pilot["aggregates"],
                "per_query": [{"id": q["id"], "scenario": q["scenario"], "metrics": q["metrics"]} for q in fs_pilot["per_query"]],
            },
            "candidate": {
                "strategy": "markdown_aware",
                "aggregates": md_pilot["aggregates"],
                "per_query": [{"id": q["id"], "scenario": q["scenario"], "metrics": q["metrics"]} for q in md_pilot["per_query"]],
            },
            "deltas": {k: round(v, 4) if v is not None else None for k, v in pilot_deltas.items()},
            "passed": pilot_pass,
        },
    }

    output_path = Path(__file__).parent / "results" / "w1_promotion_gate.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results: {output_path}")


if __name__ == "__main__":
    main()
