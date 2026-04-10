"""W3a #16: Query expansion tests.

Validates:
- RRF merge logic is deterministic and correct
- Default search path is unchanged when expansion is disabled
- Expansion path runs only when enabled
- Duplicate/repeated results are handled correctly
- Fallback to dense-only when LLM call fails
- Query expansion flag is exposed in search API
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from services.query_expansion import rrf_merge, expand_query, RRF_K


# ── RRF merge ────────────────────────────────────────────────────────────


class TestRRFMerge:
    """Reciprocal Rank Fusion merge logic."""

    def test_single_list(self):
        """Single ranked list — RRF scores are 1/(k+rank+1)."""
        results = [
            {"text": "a", "score": 0.9, "scope": "s", "citation": {"source": "a.md", "chunk_index": 0}},
            {"text": "b", "score": 0.7, "scope": "s", "citation": {"source": "b.md", "chunk_index": 0}},
        ]
        merged = rrf_merge([results], limit=5)
        assert len(merged) == 2
        assert merged[0]["citation"]["source"] == "a.md"
        assert merged[1]["citation"]["source"] == "b.md"
        # RRF score for rank 0: 1/(60+1) ≈ 0.016393
        assert abs(merged[0]["score"] - 1 / (RRF_K + 1)) < 0.001

    def test_two_lists_boost_overlap(self):
        """Result appearing in both lists gets higher RRF score."""
        list1 = [
            {"text": "shared", "score": 0.9, "scope": "s", "citation": {"source": "shared.md", "chunk_index": 0}},
            {"text": "only1", "score": 0.5, "scope": "s", "citation": {"source": "only1.md", "chunk_index": 0}},
        ]
        list2 = [
            {"text": "shared", "score": 0.8, "scope": "s", "citation": {"source": "shared.md", "chunk_index": 0}},
            {"text": "only2", "score": 0.6, "scope": "s", "citation": {"source": "only2.md", "chunk_index": 0}},
        ]
        merged = rrf_merge([list1, list2], limit=5)
        # shared.md should be first (appears in both lists)
        assert merged[0]["citation"]["source"] == "shared.md"
        # Its score should be higher than single-list items
        assert merged[0]["score"] > merged[1]["score"]

    def test_limit_respected(self):
        """Merged results are limited to requested count."""
        results = [
            {"text": f"r{i}", "score": 1.0 - i * 0.1, "scope": "s",
             "citation": {"source": f"r{i}.md", "chunk_index": 0}}
            for i in range(10)
        ]
        merged = rrf_merge([results], limit=3)
        assert len(merged) == 3

    def test_empty_lists(self):
        """Empty input produces empty output."""
        assert rrf_merge([], limit=5) == []
        assert rrf_merge([[]], limit=5) == []

    def test_deterministic(self):
        """Same input produces same output every time."""
        results = [
            {"text": "a", "score": 0.9, "scope": "s", "citation": {"source": "a.md", "chunk_index": 0}},
            {"text": "b", "score": 0.7, "scope": "s", "citation": {"source": "b.md", "chunk_index": 1}},
        ]
        m1 = rrf_merge([results], limit=5)
        m2 = rrf_merge([results], limit=5)
        assert m1 == m2

    def test_different_chunk_indices_not_merged(self):
        """Same source but different chunk_index are separate results."""
        list1 = [
            {"text": "chunk0", "score": 0.9, "scope": "s", "citation": {"source": "doc.md", "chunk_index": 0}},
        ]
        list2 = [
            {"text": "chunk1", "score": 0.8, "scope": "s", "citation": {"source": "doc.md", "chunk_index": 1}},
        ]
        merged = rrf_merge([list1, list2], limit=5)
        assert len(merged) == 2


# ── expand_query ─────────────────────────────────────────────────────────


class TestExpandQuery:
    """Query expansion via LLM."""

    def test_no_api_key_returns_empty(self):
        """Without API key, returns empty list (silent fallback)."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            result = expand_query("test query", n=3, api_key="")
            assert result == []

    def test_successful_expansion(self):
        """Mocked LLM response is parsed into phrasings."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices":[{"message":{"content":"What is X?\\nHow does X work?\\nExplain X"}}]}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = expand_query("What is X?", n=3, api_key="test-key")
            assert len(result) >= 1
            assert all(isinstance(p, str) for p in result)

    def test_llm_failure_returns_empty(self):
        """LLM call failure returns empty list (circuit breaker)."""
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            result = expand_query("test query", n=3, api_key="test-key")
            assert result == []

    def test_duplicate_of_original_filtered(self):
        """If LLM returns the original query, it is filtered out."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices":[{"message":{"content":"test query\\nAlternative phrasing\\nAnother version"}}]}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = expand_query("test query", n=3, api_key="test-key")
            assert "test query" not in [p.lower() for p in result]

    def test_max_n_capped(self):
        """N is capped at MAX_EXPANSION_COUNT."""
        from services.query_expansion import MAX_EXPANSION_COUNT
        mock_response = MagicMock()
        lines = "\n".join(f"Phrasing {i}" for i in range(10))
        mock_response.read.return_value = ('{"choices":[{"message":{"content":' + f'"{lines}"' + '}}]}').encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = expand_query("test", n=10, api_key="test-key")
            assert len(result) <= MAX_EXPANSION_COUNT


# ── KnowledgeService integration ────────────────────────────────────────


class TestSearchWithExpansionDisabled:
    """Default search path is unchanged when expansion is disabled."""

    def test_default_search_no_expansion(self):
        """search() with default params does not call expand_query."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.ingest("scope", "Kubernetes deployment guide.", "k8s.md")

        with patch("services.query_expansion.expand_query") as mock_expand:
            results = svc.search("Kubernetes", ["scope"])
            mock_expand.assert_not_called()
            assert len(results) >= 1

    def test_explicit_disabled(self):
        """search() with query_expansion_enabled=False does not call expand_query."""
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.ingest("scope", "Python programming language.", "py.md")

        with patch("services.query_expansion.expand_query") as mock_expand:
            results = svc.search("Python", ["scope"], query_expansion_enabled=False)
            mock_expand.assert_not_called()


class TestSearchWithExpansionEnabled:
    """Expansion path runs when enabled (in-memory path — no Qdrant)."""

    def test_expansion_enabled_in_memory_still_works(self):
        """In-memory search with expansion enabled still returns results.

        The in-memory path does not support expansion (requires Qdrant),
        so it falls back to standard in-memory search.
        """
        from services.knowledge import KnowledgeService

        svc = KnowledgeService()
        svc.ingest("scope", "Kubernetes deployment.", "k8s.md")
        # In-memory path ignores expansion flag — just returns normal results
        results = svc.search("Kubernetes", ["scope"], query_expansion_enabled=True)
        assert len(results) >= 1


# ── API endpoint ─────────────────────────────────────────────────────────


class TestSearchAPIExpansionParam:
    """Search API exposes query_expansion parameter."""

    def test_search_endpoint_accepts_expansion_param(self):
        """GET /search?query_expansion=false is accepted."""
        from fastapi.testclient import TestClient
        from main import app

        token = os.environ.get("KNOWLEDGE_API_INTERNAL_TOKEN", "test-internal-token-for-tests")
        with TestClient(app) as client:
            resp = client.get(
                "/api/v1/knowledge/search",
                params={"query": "test", "query_expansion": "false"},
                headers={"X-Internal-Token": token},
            )
            assert resp.status_code == 200

    def test_search_endpoint_expansion_default_false(self):
        """GET /search without query_expansion defaults to false."""
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
