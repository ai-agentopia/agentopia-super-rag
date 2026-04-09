"""Phase 1a evaluation harness tests (#317).

Tests validate:
1. Dataset format (queries + scopes, no pre-baked results)
2. Summary computation logic
3. Artifact generation
4. Integration: run_evaluation() with mocked retrieval + mocked RAGAS scorers
5. Phase boundary messaging
6. Cost control defaults

NOTE: Phase 1a is EARLY SIGNAL ONLY. No nDCG/MRR/Precision@K.
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "datasets", "phase1a_sample.json"
)


# ── Dataset format tests ─────────────────────────────────────────────────────


def test_load_dataset_reads_samples():
    from evaluation.phase1a_runner import load_dataset
    samples = load_dataset(DATASET_PATH, max_samples=0)
    assert len(samples) == 5


def test_load_dataset_respects_max_samples():
    from evaluation.phase1a_runner import load_dataset
    samples = load_dataset(DATASET_PATH, max_samples=2)
    assert len(samples) == 2


def test_dataset_v2_format():
    """v2 dataset has queries + scopes, NOT pre-baked responses or contexts."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    assert data["version"] == "2.0"
    assert "system under test" in data["note"]

    for sample in data["samples"]:
        assert "user_input" in sample
        assert "scope" in sample
        assert "scenario" in sample
        # v2: NO pre-baked results — runner produces these
        assert "response" not in sample, f"{sample['id']}: has pre-baked response (v1 leftover)"
        assert "retrieved_contexts" not in sample, f"{sample['id']}: has pre-baked contexts (v1 leftover)"


def test_dataset_has_no_labeled_relevance():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    for sample in data["samples"]:
        assert "relevance" not in sample
        assert "ndcg" not in sample


# ── Summary computation tests ────────────────────────────────────────────────


def test_build_summary_computes_aggregates():
    from evaluation.phase1a_runner import _build_summary
    results = [
        {"id": "s01", "query": "q1", "scenario": "a", "contexts_retrieved": 3,
         "response_preview": "...", "scope": "s",
         "scores": {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_utilization": 0.7}},
        {"id": "s02", "query": "q2", "scenario": "b", "contexts_retrieved": 2,
         "response_preview": "...", "scope": "s",
         "scores": {"faithfulness": 0.7, "answer_relevancy": 0.9, "context_utilization": 0.8}},
    ]
    summary = _build_summary(results)
    assert summary["aggregates"]["faithfulness"]["mean"] == 0.8
    assert summary["total_contexts_retrieved"] == 5
    assert summary["status"] == "OK"


def test_build_summary_warns_on_low_scores():
    from evaluation.phase1a_runner import _build_summary
    results = [
        {"id": "s01", "query": "q1", "scenario": "a", "contexts_retrieved": 0,
         "response_preview": "...", "scope": "s",
         "scores": {"faithfulness": 0.3, "answer_relevancy": 0.4, "context_utilization": 0.2}},
    ]
    summary = _build_summary(results)
    assert summary["status"] == "WARN"


def test_build_summary_handles_none_scores():
    from evaluation.phase1a_runner import _build_summary
    results = [
        {"id": "s01", "query": "q1", "scenario": "a", "contexts_retrieved": 1,
         "response_preview": "...", "scope": "s",
         "scores": {"faithfulness": None, "answer_relevancy": 0.8, "context_utilization": None}},
    ]
    summary = _build_summary(results)
    assert summary["aggregates"]["faithfulness"]["count"] == 0


# ── Artifact output tests ────────────────────────────────────────────────────


def test_write_artifacts_creates_json_and_markdown():
    from evaluation.phase1a_runner import write_artifacts
    summary = {
        "per_sample": [
            {"id": "s01", "query": "test", "scenario": "factual", "contexts_retrieved": 3,
             "response_preview": "answer...", "scope": "s",
             "scores": {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_utilization": 0.7}},
        ],
        "aggregates": {
            "faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9, "count": 1, "warn": False},
            "answer_relevancy": {"mean": 0.8, "min": 0.8, "max": 0.8, "count": 1, "warn": False},
            "context_utilization": {"mean": 0.7, "min": 0.7, "max": 0.7, "count": 1, "warn": False},
        },
        "total_contexts_retrieved": 3,
        "status": "OK",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, md_path = write_artifacts(
            summary, tmpdir, "judge/model", "resp/model", "test.json", 1, "test-adapter",
        )
        with open(json_path) as f:
            artifact = json.load(f)
        assert artifact["phase"] == "1a"
        assert artifact["adapter"] == "test-adapter"
        assert artifact["total_contexts_retrieved"] == 3
        assert "system under test" in artifact["note"]

        with open(md_path) as f:
            md = f.read()
        assert "EARLY SIGNAL ONLY" in md
        assert "system under test" in md
        assert "#318" in md


# ── Integration test: run_evaluation with mocked retrieval + RAGAS ───────────


def _setup_ragas_mocks():
    """Set up mock RAGAS modules in sys.modules so run_evaluation() can import them."""
    import sys

    mock_score_result = MagicMock()
    mock_score_result.value = 0.85

    mock_scorer_cls = MagicMock()
    mock_scorer_instance = MagicMock()
    mock_scorer_instance.ascore = AsyncMock(return_value=mock_score_result)
    mock_scorer_cls.return_value = mock_scorer_instance

    # Mock ragas modules
    ragas_mod = MagicMock()
    ragas_llms = MagicMock()
    ragas_llms.llm_factory = MagicMock(return_value=MagicMock())
    ragas_metrics = MagicMock()
    ragas_metrics.Faithfulness = mock_scorer_cls
    ragas_metrics.AnswerRelevancy = mock_scorer_cls
    ragas_metrics.ContextUtilization = mock_scorer_cls

    sys.modules["ragas"] = ragas_mod
    sys.modules["ragas.llms"] = ragas_llms
    sys.modules["ragas.metrics"] = ragas_metrics
    sys.modules["ragas.metrics.collections"] = ragas_metrics

    return mock_scorer_instance


def _mock_openai_response(text="The system supports three chunking strategies."):
    """Create a mock AsyncOpenAI client that returns a fixed response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = text

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.mark.asyncio
async def test_run_evaluation_calls_retrieval_adapter():
    """Integration: run_evaluation calls retrieval adapter per sample, generates response, scores."""
    mock_scorer = _setup_ragas_mocks()
    mock_openai_client = _mock_openai_response()

    mock_adapter = MagicMock()
    mock_adapter.retrieve.return_value = ["Context chunk about chunking strategies."]

    samples = [
        {"id": "t01", "user_input": "What chunking strategies?", "scope": "test/scope",
         "scenario": "test", "search_limit": 5},
    ]

    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        # Force re-import to pick up mocked modules
        import importlib
        import evaluation.phase1a_runner as runner_mod
        importlib.reload(runner_mod)

        summary = await runner_mod.run_evaluation(
            samples, mock_adapter, "judge/model", "resp/model",
            "http://test", "test-key",
        )

    # Verify retrieval adapter was called with correct args
    mock_adapter.retrieve.assert_called_once_with("What chunking strategies?", "test/scope", 5)

    # Verify response generation was called
    mock_openai_client.chat.completions.create.assert_called_once()

    # Verify RAGAS scorers were called (2 metrics × 1 sample = 2 calls)
    assert mock_scorer.ascore.call_count == 2

    # Verify results
    assert len(summary["per_sample"]) == 1
    assert summary["per_sample"][0]["contexts_retrieved"] == 1
    assert summary["per_sample"][0]["scores"]["faithfulness"] == 0.85
    assert summary["status"] == "OK"


@pytest.mark.asyncio
async def test_run_evaluation_handles_empty_retrieval():
    """Integration: gracefully handles scope with no results."""
    mock_score_result = MagicMock()
    mock_score_result.value = 0.0

    mock_scorer = _setup_ragas_mocks()
    mock_scorer.ascore = AsyncMock(return_value=mock_score_result)

    mock_openai_client = _mock_openai_response("I don't have relevant knowledge.")

    mock_adapter = MagicMock()
    mock_adapter.retrieve.return_value = []

    samples = [{"id": "t01", "user_input": "Unknown topic?", "scope": "empty/scope",
                "scenario": "test", "search_limit": 5}]

    with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
        import importlib
        import evaluation.phase1a_runner as runner_mod
        importlib.reload(runner_mod)

        summary = await runner_mod.run_evaluation(
            samples, mock_adapter, "m", "m", "http://test", "key",
        )

    assert summary["per_sample"][0]["contexts_retrieved"] == 0


# ── Phase boundary tests ─────────────────────────────────────────────────────


def test_phase1a_boundary_no_ndcg_mrr():
    import evaluation
    assert "nDCG" in evaluation.__doc__
    assert "MRR" in evaluation.__doc__
    assert "NOT" in evaluation.__doc__


def test_runner_module_docstring_boundary():
    from evaluation import phase1a_runner
    assert "EARLY SIGNAL ONLY" in phase1a_runner.__doc__
    assert "actual retrieval" in phase1a_runner.__doc__.lower()


# ── Cost control tests ───────────────────────────────────────────────────────


def test_default_max_samples_is_small():
    from evaluation.phase1a_runner import DEFAULT_MAX_SAMPLES
    assert DEFAULT_MAX_SAMPLES <= 10


def test_default_judge_model_is_cheap():
    from evaluation.phase1a_runner import DEFAULT_JUDGE_MODEL
    assert "gpt-4o" not in DEFAULT_JUDGE_MODEL
    assert "opus" not in DEFAULT_JUDGE_MODEL


def test_warn_thresholds_defined():
    from evaluation.phase1a_runner import WARN_THRESHOLDS
    assert all(k in WARN_THRESHOLDS for k in ["faithfulness", "context_utilization"])


# ── Adapter tests ────────────────────────────────────────────────────────────


def test_http_adapter_calls_correct_endpoint():
    """HttpAdapter calls GET /api/v1/knowledge/search with correct params."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"text": "chunk1"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response) as mock_get:
        from evaluation.phase1a_runner import HttpAdapter
        adapter = HttpAdapter("http://bot-config-api:8001", "test-token", "eval-bot")
        results = adapter.retrieve("test query", "test/scope", 5)

    assert results == ["chunk1"]
    call_args = mock_get.call_args
    assert "/api/v1/knowledge/search" in call_args[0][0]
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-token"
    assert call_args[1]["headers"]["X-Bot-Name"] == "eval-bot"


def test_http_adapter_does_not_send_scope_param():
    """HttpAdapter does NOT pass scope to the server — server resolves from bot subscription."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response) as mock_get:
        from evaluation.phase1a_runner import HttpAdapter
        adapter = HttpAdapter("http://test:8001", "token", "bot")
        adapter.retrieve("query", "some/specific/scope", 5)

    # Scope is NOT in the query params — server resolves from bot identity
    call_params = mock_get.call_args[1]["params"]
    assert "scope" not in call_params
    assert "scopes" not in call_params


def test_validate_dataset_rejects_multi_scope_for_http_adapter():
    """Multi-scope dataset + HttpAdapter raises ValueError."""
    from evaluation.phase1a_runner import HttpAdapter, validate_dataset_for_adapter

    with patch("httpx.get"):
        adapter = HttpAdapter("http://test", "token", "bot")

    samples = [
        {"id": "s1", "user_input": "q1", "scope": "client/docs"},
        {"id": "s2", "user_input": "q2", "scope": "client/code"},  # different scope
    ]

    with pytest.raises(ValueError, match="distinct scopes"):
        validate_dataset_for_adapter(samples, adapter)


def test_validate_dataset_allows_single_scope_for_http_adapter():
    """Single-scope dataset + HttpAdapter is accepted."""
    from evaluation.phase1a_runner import HttpAdapter, validate_dataset_for_adapter

    with patch("httpx.get"):
        adapter = HttpAdapter("http://test", "token", "bot")

    samples = [
        {"id": "s1", "user_input": "q1", "scope": "client/docs"},
        {"id": "s2", "user_input": "q2", "scope": "client/docs"},  # same scope
    ]

    # Should not raise
    validate_dataset_for_adapter(samples, adapter)


def test_validate_dataset_allows_any_scopes_for_knowledge_service_adapter():
    """KnowledgeServiceAdapter supports per-sample scopes — no validation needed."""
    from evaluation.phase1a_runner import validate_dataset_for_adapter

    # Use a mock that is NOT HttpAdapter
    mock_adapter = MagicMock()  # generic mock, not HttpAdapter instance

    samples = [
        {"id": "s1", "user_input": "q1", "scope": "client/docs"},
        {"id": "s2", "user_input": "q2", "scope": "client/code"},
    ]

    # Should not raise — KnowledgeServiceAdapter handles per-sample scopes
    validate_dataset_for_adapter(samples, mock_adapter)
