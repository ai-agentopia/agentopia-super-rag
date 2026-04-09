"""Phase 1b labeled baseline evaluation tests (#318).

Tests:
1. Metric computation correctness (known-answer tests)
2. Dataset format validation
3. Grading logic (retrieved results → labeled judgments → relevance list)
4. Artifact generation
5. E2E: in-memory retrieval → grade → metrics → artifact
6. Wave gate and Phase 1a boundary
"""

import json
import math
import os
import tempfile
from unittest.mock import MagicMock

import pytest

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "datasets", "phase1b_labeled_seed.json"
)


# ── Metric computation correctness ───────────────────────────────────────────


class TestNDCG:
    def test_perfect_ranking(self):
        """Perfect ranking: all relevant at top → nDCG = 1.0."""
        from evaluation.retrieval_metrics import ndcg_at_k
        # relevances in ideal order already
        assert ndcg_at_k([2, 1, 0, 0, 0], 5) == 1.0

    def test_worst_ranking(self):
        """All irrelevant → nDCG = 0.0."""
        from evaluation.retrieval_metrics import ndcg_at_k
        assert ndcg_at_k([0, 0, 0, 0, 0], 5) == 0.0

    def test_reversed_ranking(self):
        """Relevant items at bottom → nDCG < 1.0."""
        from evaluation.retrieval_metrics import ndcg_at_k
        score = ndcg_at_k([0, 0, 0, 1, 2], 5)
        assert 0 < score < 1.0

    def test_single_relevant_at_top(self):
        """One relevant item at rank 1."""
        from evaluation.retrieval_metrics import ndcg_at_k
        assert ndcg_at_k([2, 0, 0, 0, 0], 5) == 1.0

    def test_known_value(self):
        """Verify against hand-calculated value."""
        from evaluation.retrieval_metrics import dcg_at_k, ndcg_at_k
        # relevances: [2, 0, 1, 0, 0]
        # DCG = 2/log2(2) + 0/log2(3) + 1/log2(4) + 0 + 0
        #     = 2/1 + 0 + 1/2 + 0 + 0 = 2.5
        dcg = dcg_at_k([2, 0, 1, 0, 0], 5)
        assert abs(dcg - 2.5) < 0.001

        # IDCG (ideal: [2, 1, 0, 0, 0]): 2/1 + 1/log2(3) = 2 + 0.6309 = 2.6309
        idcg = dcg_at_k([2, 1, 0, 0, 0], 5)
        expected_ndcg = 2.5 / idcg
        actual_ndcg = ndcg_at_k([2, 0, 1, 0, 0], 5)
        assert abs(actual_ndcg - expected_ndcg) < 0.001


class TestMRR:
    def test_relevant_at_rank_1(self):
        from evaluation.retrieval_metrics import mrr
        assert mrr([2, 0, 0]) == 1.0

    def test_relevant_at_rank_3(self):
        from evaluation.retrieval_metrics import mrr
        assert abs(mrr([0, 0, 1]) - 1/3) < 0.001

    def test_no_relevant(self):
        from evaluation.retrieval_metrics import mrr
        assert mrr([0, 0, 0]) == 0.0


class TestPrecision:
    def test_all_relevant(self):
        from evaluation.retrieval_metrics import precision_at_k
        assert precision_at_k([2, 1, 2, 1, 1], 5) == 1.0

    def test_none_relevant(self):
        from evaluation.retrieval_metrics import precision_at_k
        assert precision_at_k([0, 0, 0, 0, 0], 5) == 0.0

    def test_partial(self):
        from evaluation.retrieval_metrics import precision_at_k
        assert precision_at_k([2, 0, 1, 0, 0], 5) == 0.4  # 2/5


class TestRecall:
    def test_all_found(self):
        from evaluation.retrieval_metrics import recall_at_k
        assert recall_at_k([2, 1], 5, total_relevant=2) == 1.0

    def test_none_found(self):
        from evaluation.retrieval_metrics import recall_at_k
        assert recall_at_k([0, 0, 0], 5, total_relevant=3) == 0.0

    def test_partial(self):
        from evaluation.retrieval_metrics import recall_at_k
        assert abs(recall_at_k([2, 0, 0], 5, total_relevant=2) - 0.5) < 0.001

    def test_zero_total_relevant(self):
        from evaluation.retrieval_metrics import recall_at_k
        assert recall_at_k([0, 0], 5, total_relevant=0) == 0.0


class TestComputeQueryMetrics:
    def test_returns_all_metrics(self):
        from evaluation.retrieval_metrics import compute_query_metrics
        m = compute_query_metrics([2, 1, 0, 0, 0], total_relevant=2, k=5)
        assert "ndcg" in m
        assert "mrr" in m
        assert "precision" in m
        assert "recall" in m
        assert m["ndcg"] == 1.0  # perfect ranking
        assert m["mrr"] == 1.0  # relevant at rank 1
        assert m["precision"] == 0.4  # 2/5
        assert m["recall"] == 1.0  # both found


# ── Dataset format validation ────────────────────────────────────────────────


def test_labeled_dataset_format():
    from evaluation.phase1b_baseline import load_labeled_dataset
    data = load_labeled_dataset(DATASET_PATH)
    assert data["phase"] == "1b"
    assert "grading_scale" in data
    assert len(data["samples"]) == 5


def test_labeled_dataset_has_graded_relevance():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    for sample in data["samples"]:
        for chunk in sample["relevant_chunks"]:
            assert chunk["relevance"] in (0, 1, 2), f"Invalid grade in {sample['id']}"
            assert "source" in chunk
            assert "chunk_index" in chunk


def test_labeled_dataset_has_no_ragas_metrics():
    """Phase 1b dataset must not contain RAGAS/reference-free metrics."""
    with open(DATASET_PATH) as f:
        data = json.load(f)
    for sample in data["samples"]:
        assert "faithfulness" not in sample
        assert "answer_relevancy" not in sample


# ── Grading logic ────────────────────────────────────────────────────────────


def test_grade_retrieval_results_matches_labels():
    from evaluation.phase1b_baseline import grade_retrieval_results

    retrieved = [
        {"text": "t1", "score": 0.9, "citation": {"source": "doc.md", "chunk_index": 3, "section": ""}},
        {"text": "t2", "score": 0.7, "citation": {"source": "doc.md", "chunk_index": 5, "section": ""}},
        {"text": "t3", "score": 0.5, "citation": {"source": "other.md", "chunk_index": 0, "section": ""}},
    ]
    labeled_chunks = [
        {"source": "doc.md", "chunk_index": 3, "relevance": 2, "reason": "exact match"},
        {"source": "doc.md", "chunk_index": 5, "relevance": 1, "reason": "partial"},
    ]

    result = grade_retrieval_results(retrieved, labeled_chunks, k=5)
    assert result["relevances"] == [2.0, 1.0, 0.0]  # third is unjudged → 0
    assert result["metrics"]["ndcg"] == 1.0  # perfect ranking
    assert result["metrics"]["mrr"] == 1.0
    assert result["total_relevant_in_labels"] == 2


def test_grade_unjudged_defaults_to_zero():
    from evaluation.phase1b_baseline import grade_retrieval_results

    retrieved = [
        {"text": "unknown", "score": 0.8, "citation": {"source": "mystery.md", "chunk_index": 99, "section": ""}},
    ]
    labeled_chunks = [
        {"source": "doc.md", "chunk_index": 1, "relevance": 2, "reason": "not this one"},
    ]

    result = grade_retrieval_results(retrieved, labeled_chunks, k=5)
    assert result["relevances"] == [0.0]
    assert result["grading_detail"][0]["judged"] is False


def test_grade_empty_retrieval():
    from evaluation.phase1b_baseline import grade_retrieval_results

    result = grade_retrieval_results([], [{"source": "d", "chunk_index": 0, "relevance": 2, "reason": "r"}], k=5)
    assert result["relevances"] == []
    assert result["metrics"]["ndcg"] == 0.0
    assert result["metrics"]["mrr"] == 0.0


# ── Artifact output ──────────────────────────────────────────────────────────


def test_write_baseline_artifact():
    from evaluation.phase1b_baseline import write_baseline_artifact

    summary = {
        "per_query": [{
            "id": "lb01", "query": "test", "scope": "s", "scenario": "factual",
            "relevances": [2.0, 0.0], "metrics": {"ndcg": 0.8, "mrr": 1.0, "precision": 0.5, "recall": 1.0},
            "grading_detail": [], "total_relevant_in_labels": 1, "retrieved_count": 2,
        }],
        "aggregates": {
            "ndcg": {"mean": 0.8, "min": 0.8, "max": 0.8, "count": 1},
            "mrr": {"mean": 1.0, "min": 1.0, "max": 1.0, "count": 1},
            "precision": {"mean": 0.5, "min": 0.5, "max": 0.5, "count": 1},
            "recall": {"mean": 1.0, "min": 1.0, "max": 1.0, "count": 1},
        },
        "k": 5, "dataset_version": "1.0", "dataset_wave": "test", "sample_count": 1,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, md_path = write_baseline_artifact(summary, tmpdir, "test.json", "test-adapter")

        with open(json_path) as f:
            artifact = json.load(f)
        assert artifact["phase"] == "1b"
        assert artifact["type"] == "baseline"
        assert "AUTHORITATIVE" in artifact["note"]
        assert "Phase 1a" in artifact["note"]

        with open(md_path) as f:
            md = f.read()
        assert "AUTHORITATIVE BASELINE" in md
        assert "Phase 2a" in md
        assert "Wave 1" in md
        assert "Wave 2" in md
        assert "PENDING" in md


# ── E2E: in-memory retrieval → grade → metrics → artifact ────────────────────


def test_e2e_baseline_with_in_memory_retrieval():
    """E2E: seed data into in-memory KnowledgeService, run baseline, verify metrics."""
    from services.knowledge import KnowledgeService
    from models.knowledge import DocumentFormat
    from evaluation.phase1b_baseline import (
        InMemoryLabeledAdapter,
        load_labeled_dataset,
        run_baseline,
        write_baseline_artifact,
    )

    # Create knowledge service with in-memory backend
    svc = KnowledgeService()
    scope = "test/platform-docs"

    # Seed documents via ingest() — in-memory backend uses text matching
    svc.ingest(scope, (
        "The knowledge system supports four chunking strategies: "
        "FIXED_SIZE (default, 512 tokens), PARAGRAPH, CODE_AWARE, and SEMANTIC (not yet implemented). "
        "IngestConfig controls chunk_size (64-4096) and chunk_overlap (0-512)."
    ), source="knowledge-architecture.md", format=DocumentFormat.MARKDOWN)

    svc.ingest(scope, (
        "Document ingestion uses SHA-256 hashing for deduplication. "
        "Two-phase atomic replace supersedes old records. "
        "ADR-012 defines the two-phase replace model for document lifecycle."
    ), source="adr-012-ingestion.md", format=DocumentFormat.MARKDOWN)

    svc.ingest(scope, (
        "Knowledge routes use dual-path authentication: operator session cookies "
        "and bot bearer tokens with X-Bot-Name header."
    ), source="adr-010-runtime.md", format=DocumentFormat.MARKDOWN)

    svc.ingest(scope, (
        "The gateway knowledge-retrieval plugin has maxContextTokens=3000 "
        "and confidenceThreshold=0.3 for filtering low-relevance results."
    ), source="gateway-plugin-config.md", format=DocumentFormat.MARKDOWN)

    svc.ingest(scope, (
        "BotKnowledgeIndex is a singleton that rebuilds from K8s CRD annotations "
        "on startup, maintaining an in-memory dict for O(1) scope resolution."
    ), source="bot-knowledge-index.md", format=DocumentFormat.MARKDOWN)

    # Load labeled dataset
    dataset = load_labeled_dataset(DATASET_PATH)

    # Run baseline with in-memory adapter
    adapter = InMemoryLabeledAdapter(svc)
    summary = run_baseline(dataset, adapter, k=5)

    # Verify structure
    assert summary["sample_count"] == 5
    assert summary["k"] == 5
    assert all(m in summary["aggregates"] for m in ["ndcg", "mrr", "precision", "recall"])

    # All metrics should be numeric (not None)
    for metric_name in ["ndcg", "mrr", "precision", "recall"]:
        assert summary["aggregates"][metric_name]["mean"] is not None
        assert summary["aggregates"][metric_name]["count"] == 5

    # Verify per-query structure
    for q in summary["per_query"]:
        assert "metrics" in q
        assert "relevances" in q
        assert "grading_detail" in q

    # Write artifacts and verify
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, md_path = write_baseline_artifact(
            summary, tmpdir, DATASET_PATH, "in-memory (e2e test)",
        )
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

        with open(json_path) as f:
            artifact = json.load(f)
        assert artifact["sample_count"] == 5
        assert artifact["phase"] == "1b"

    # Log metrics for visibility
    for name in ["ndcg", "mrr", "precision", "recall"]:
        print(f"  {name}: {summary['aggregates'][name]['mean']:.4f}")


# ── Wave gate and boundary tests ─────────────────────────────────────────────


def test_phase1b_boundary_with_phase1a():
    """Phase 1b module explicitly distinguishes from Phase 1a."""
    from evaluation import phase1b_baseline
    assert "AUTHORITATIVE" in phase1b_baseline.__doc__
    assert "Phase 1a" in phase1b_baseline.__doc__


def test_dataset_wave_is_explicit():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    assert "wave" in data
    assert "Wave 1" in data["wave"] or "seed" in data["wave"].lower()
    assert "#307" in data["wave"]
