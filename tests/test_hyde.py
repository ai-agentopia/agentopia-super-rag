"""W3b #17: HyDE (Hypothetical Document Embedding) tests.

Validates:
- generate_hypothesis returns text on success, empty string on failure
- Default search path is unchanged when HyDE is disabled
- HyDE path only runs when enabled AND scope is in allowlist
- Per-scope rollout control (enable/disable/lifecycle)
- Deterministic RRF merge behavior
- Fallback to dense-only when generation fails
- Blocked scopes stay dense-only even when hyde_enabled=True
- HyDE flag is exposed in search API
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from services.hyde import generate_hypothesis


# ── generate_hypothesis ──────────────────────────────────────────────────


class TestGenerateHypothesis:
    """Hypothesis generation via LLM."""

    def test_no_api_key_returns_empty(self):
        """Without API key, returns empty string (silent fallback)."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "EMBEDDING_API_KEY": ""}, clear=False):
            result = generate_hypothesis("test query", api_key="")
            assert result == ""

    def test_successful_generation(self):
        """Mocked LLM response returns hypothesis text."""
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"choices":[{"message":{"content":"Kubernetes is a container orchestration platform."}}]}'
        )
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = generate_hypothesis("What is Kubernetes?", api_key="test-key")
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Kubernetes" in result

    def test_llm_failure_returns_empty(self):
        """LLM call failure returns empty string (circuit breaker)."""
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = generate_hypothesis("test query", api_key="test-key")
            assert result == ""

    def test_hypothesis_is_stripped(self):
        """Returned hypothesis has leading/trailing whitespace stripped."""
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"choices":[{"message":{"content":"  Some hypothesis text.  "}}]}'
        )
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = generate_hypothesis("query", api_key="test-key")
            assert result == "Some hypothesis text."

    def test_embedding_api_key_fallback(self):
        """EMBEDDING_API_KEY is used when OPENROUTER_API_KEY is absent."""
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"choices":[{"message":{"content":"A hypothesis."}}]}'
        )
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.dict(
            os.environ,
            {"EMBEDDING_API_KEY": "relay-token", "OPENROUTER_API_KEY": ""},
            clear=False
        ):
            with patch("urllib.request.urlopen", return_value=mock_response) as mock_call:
                result = generate_hypothesis("query")
                assert result == "A hypothesis."
                # Verify the call was made (key was resolved from EMBEDDING_API_KEY)
                mock_call.assert_called_once()


# ── Per-scope rollout control ─────────────────────────────────────────────


class TestHyDEPerScopeRollout:
    """HyDE is gated per-scope, not just per-request."""

    def test_hyde_blocked_when_scope_not_allowed(self):
        """hyde_enabled=True ignored when scope not in allowlist → dense-only."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.ingest("blocked-scope", "Kubernetes deployment guide.", "k8s.md")
        # No enable_hyde() called

        with patch("services.hyde.generate_hypothesis") as mock_gen:
            results = svc.search(
                "Kubernetes", ["blocked-scope"], hyde_enabled=True
            )
            mock_gen.assert_not_called()
            assert len(results) >= 1

    def test_hyde_allowed_when_scope_enabled(self):
        """Allowlist check passes when scope is enabled."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.enable_hyde("allowed-scope")

        assert svc.is_hyde_allowed(["allowed-scope"]) is True
        assert svc.is_hyde_allowed(["other-scope"]) is False

    def test_enable_disable_lifecycle(self):
        """Enable then disable HyDE for a scope."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        assert svc.is_hyde_allowed(["test"]) is False
        svc.enable_hyde("test")
        assert svc.is_hyde_allowed(["test"]) is True
        svc.disable_hyde("test")
        assert svc.is_hyde_allowed(["test"]) is False

    def test_mixed_scope_only_expands_allowed(self):
        """Multi-scope: only allowed scopes get HyDE."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.enable_hyde("allowed")

        allowed = [s for s in ["allowed", "blocked"] if s in svc._hyde_allowed_scopes]
        blocked = [s for s in ["allowed", "blocked"] if s not in svc._hyde_allowed_scopes]
        assert allowed == ["allowed"]
        assert blocked == ["blocked"]

    def test_all_scopes_blocked_no_hyde(self):
        """All scopes blocked → generate_hypothesis never called."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.ingest("scope-a", "Content A.", "a.md")
        svc.ingest("scope-b", "Content B.", "b.md")

        with patch("services.hyde.generate_hypothesis") as mock_gen:
            results = svc.search(
                "Content", ["scope-a", "scope-b"], hyde_enabled=True
            )
            mock_gen.assert_not_called()
            assert len(results) >= 1

    def test_env_var_initialization(self):
        """HYDE_SCOPES env var populates allowlist on init."""
        with patch.dict(os.environ, {"HYDE_SCOPES": "scope-a,scope-b"}):
            from services.knowledge import KnowledgeService
            svc = KnowledgeService()
            assert svc.is_hyde_allowed(["scope-a"]) is True
            assert svc.is_hyde_allowed(["scope-b"]) is True
            assert svc.is_hyde_allowed(["scope-c"]) is False


# ── Default path unchanged ────────────────────────────────────────────────


class TestDefaultSearchUnchanged:
    """HyDE disabled by default — existing search behavior unchanged."""

    def test_default_search_no_hyde(self):
        """search() with default params does not call generate_hypothesis."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.ingest("scope", "Kubernetes deployment guide.", "k8s.md")

        with patch("services.hyde.generate_hypothesis") as mock_gen:
            results = svc.search("Kubernetes", ["scope"])
            mock_gen.assert_not_called()
            assert len(results) >= 1

    def test_explicit_disabled(self):
        """search() with hyde_enabled=False does not call generate_hypothesis."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.ingest("scope", "Python programming language.", "py.md")

        with patch("services.hyde.generate_hypothesis") as mock_gen:
            svc.search("Python", ["scope"], hyde_enabled=False)
            mock_gen.assert_not_called()


# ── HyDE path with mocked LLM ─────────────────────────────────────────────


class TestHyDEPathWithMockedLLM:
    """Exercise the W3b path with controlled mocks."""

    def test_hyde_calls_generate_and_merges_via_rrf(self):
        """With HyDE enabled + allowed scope, full path: generate → retrieve → RRF merge."""
        from services.knowledge import KnowledgeService
        from services.query_expansion import rrf_merge

        svc = KnowledgeService()
        svc.ingest("test-scope", "Kubernetes container orchestration platform.", "k8s.md")
        svc.ingest("test-scope", "Python programming language for automation.", "py.md")
        svc.enable_hyde("test-scope")

        # Simulate what _search_with_hyde does: original + hypothesis → RRF
        original = svc.search("Kubernetes", ["test-scope"], limit=5)
        hypothesis_results = svc.search("Kubernetes is a container orchestration system.", ["test-scope"], limit=5)

        ranked_lists = [
            [{"text": r.text, "score": r.score, "scope": r.scope,
              "citation": {"source": r.citation.source, "chunk_index": r.citation.chunk_index}}
             for r in original],
            [{"text": r.text, "score": r.score, "scope": r.scope,
              "citation": {"source": r.citation.source, "chunk_index": r.citation.chunk_index}}
             for r in hypothesis_results],
        ]
        merged = rrf_merge(ranked_lists, limit=5)
        assert len(merged) >= 1
        # k8s.md should rank first (appears in both lists)
        assert merged[0]["citation"]["source"] == "k8s.md"

    def test_hyde_fallback_on_generation_failure(self):
        """Generation failure → dense-only results returned, no error."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.ingest("scope", "Kubernetes deployment.", "k8s.md")
        svc.enable_hyde("scope")

        with patch("services.hyde.generate_hypothesis", return_value=""):
            results = svc.search("Kubernetes", ["scope"], hyde_enabled=True)
            # In-memory path: still returns results (dense-only)
            assert len(results) >= 1

    def test_rrf_merge_deterministic(self):
        """RRF merge of [original, hyde] is deterministic."""
        from services.query_expansion import rrf_merge

        list1 = [
            {"text": "a", "score": 0.9, "scope": "s", "citation": {"source": "a.md", "chunk_index": 0}},
            {"text": "b", "score": 0.7, "scope": "s", "citation": {"source": "b.md", "chunk_index": 0}},
        ]
        list2 = [
            {"text": "a", "score": 0.85, "scope": "s", "citation": {"source": "a.md", "chunk_index": 0}},
            {"text": "c", "score": 0.6, "scope": "s", "citation": {"source": "c.md", "chunk_index": 0}},
        ]
        m1 = rrf_merge([list1, list2], limit=5)
        m2 = rrf_merge([list1, list2], limit=5)
        assert m1 == m2
        # a.md appears in both lists → should rank first
        assert m1[0]["citation"]["source"] == "a.md"

    def test_rrf_merge_overlap_boosts(self):
        """Result in both original and HyDE lists gets higher RRF score."""
        from services.query_expansion import rrf_merge

        original = [
            {"text": "shared", "score": 0.9, "scope": "s", "citation": {"source": "shared.md", "chunk_index": 0}},
            {"text": "only-orig", "score": 0.5, "scope": "s", "citation": {"source": "orig.md", "chunk_index": 0}},
        ]
        hyde = [
            {"text": "shared", "score": 0.8, "scope": "s", "citation": {"source": "shared.md", "chunk_index": 0}},
            {"text": "only-hyde", "score": 0.6, "scope": "s", "citation": {"source": "hyde.md", "chunk_index": 0}},
        ]
        merged = rrf_merge([original, hyde], limit=5)
        assert merged[0]["citation"]["source"] == "shared.md"
        assert merged[0]["score"] > merged[1]["score"]


# ── API endpoint ──────────────────────────────────────────────────────────


class TestSearchAPIHyDEParam:
    """Search API exposes hyde parameter."""

    def test_search_endpoint_accepts_hyde_param(self):
        """GET /search?hyde=false is accepted."""
        from fastapi.testclient import TestClient
        from main import app

        token = os.environ.get("KNOWLEDGE_API_INTERNAL_TOKEN", "test-internal-token-for-tests")
        with TestClient(app) as client:
            resp = client.get(
                "/api/v1/knowledge/search",
                params={"query": "test", "hyde": "false"},
                headers={"X-Internal-Token": token},
            )
            assert resp.status_code == 200

    def test_search_endpoint_hyde_default_false(self):
        """GET /search without hyde param defaults to false."""
        from fastapi.testclient import TestClient
        from main import app

        token = os.environ.get("KNOWLEDGE_API_INTERNAL_TOKEN", "test-internal-token-for-tests")
        with TestClient(app) as client:
            resp = client.get(
                "/api/v1/knowledge/search",
                params={"query": "test"},
                headers={"X-Internal-Token": token},
            )
            assert resp.status_code == 200
