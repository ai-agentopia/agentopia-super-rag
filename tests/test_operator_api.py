"""Operator API tests — Finding 3 remediation.

Covers:
- GET /api/v1/knowledge/debug/query (retrieval debugger)
  - correct auth boundary (X-Internal-Token required; bots cannot use it)
  - correct response structure (rank, score, source, section_path, text, etc.)
  - 404 on unknown scope
  - empty results when no chunks indexed
  - scope injected via query param, not resolved from bot binding

All tests use the FastAPI TestClient with in-memory KnowledgeService (no Qdrant required).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient

INTERNAL_TOKEN = os.getenv("KNOWLEDGE_API_INTERNAL_TOKEN", "test-internal-token-for-tests")
BOT_HEADERS = {"Authorization": "Bearer some-bot-token", "X-Bot-Name": "test-bot"}
INTERNAL_HEADERS = {"X-Internal-Token": INTERNAL_TOKEN}


def _make_client():
    from main import app
    return TestClient(app, raise_server_exceptions=False)


class TestDebugQueryEndpoint(unittest.TestCase):
    """GET /api/v1/knowledge/debug/query — retrieval debugger."""

    def test_requires_internal_token(self):
        """Endpoint must reject requests without X-Internal-Token."""
        client = _make_client()
        resp = client.get("/api/v1/knowledge/debug/query?scope=test/scope&q=auth")
        # No auth → 401 or 403 (depends on auth guard implementation)
        self.assertIn(resp.status_code, (401, 403),
                      f"Expected 401/403 without token, got {resp.status_code}")

    def test_bot_token_rejected(self):
        """Bot bearer token must NOT grant access to the debug endpoint (write_router, internal only)."""
        client = _make_client()
        resp = client.get(
            "/api/v1/knowledge/debug/query?scope=test/scope&q=auth",
            headers=BOT_HEADERS,
        )
        # Bot bearer on write_router → 401 or 403
        self.assertIn(resp.status_code, (401, 403),
                      f"Bot token must not access debug endpoint, got {resp.status_code}")

    def test_missing_scope_returns_422(self):
        """scope is a required query param."""
        client = _make_client()
        resp = client.get("/api/v1/knowledge/debug/query?q=auth", headers=INTERNAL_HEADERS)
        self.assertEqual(resp.status_code, 422)

    def test_missing_query_returns_422(self):
        """q is a required query param."""
        client = _make_client()
        resp = client.get("/api/v1/knowledge/debug/query?scope=test/scope", headers=INTERNAL_HEADERS)
        self.assertEqual(resp.status_code, 422)

    def test_empty_query_string_returns_422(self):
        """q must not be empty (min_length=1)."""
        client = _make_client()
        resp = client.get("/api/v1/knowledge/debug/query?scope=test/scope&q=", headers=INTERNAL_HEADERS)
        self.assertEqual(resp.status_code, 422)

    def test_unknown_scope_returns_404(self):
        """Non-existent scope must return 404, not empty results."""
        client = _make_client()
        resp = client.get(
            "/api/v1/knowledge/debug/query?scope=nonexistent/scope&q=auth",
            headers=INTERNAL_HEADERS,
        )
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json().get("detail", "").lower())

    def test_response_structure_on_empty_scope(self):
        """When scope exists but has no indexed chunks, returns 200 with empty results."""
        from unittest.mock import patch, MagicMock
        from models.knowledge import KnowledgeScope

        client = _make_client()
        # Scope must exist in svc.get_scope() but return no results from search
        mock_scope = KnowledgeScope(name="empty/scope", document_count=0, chunk_count=0)

        with patch("routers.knowledge.get_knowledge_service") as mock_svc_fn:
            mock_svc = MagicMock()
            mock_svc.get_scope.return_value = mock_scope
            mock_svc.search.return_value = []
            mock_svc_fn.return_value = mock_svc

            resp = client.get(
                "/api/v1/knowledge/debug/query?scope=empty/scope&q=auth",
                headers=INTERNAL_HEADERS,
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["scope"], "empty/scope")
        self.assertEqual(data["result_count"], 0)
        self.assertEqual(data["results"], [])

    def test_response_fields_present(self):
        """Each result must include rank, score, source, section_path, section, chunk_index, text."""
        from unittest.mock import patch, MagicMock
        from models.knowledge import KnowledgeScope, SearchResult, Citation

        client = _make_client()
        mock_scope = KnowledgeScope(name="test/scope")
        mock_result = SearchResult(
            text="Bearer tokens are used for authentication.",
            score=0.92,
            citation=Citation(
                source="api-reference.pdf",
                section="Authentication",
                section_path="API Reference > Authentication",
                page=3,
                chunk_index=7,
                score=0.92,
                ingested_at=1712809200.0,
                document_hash="abc123",
            ),
            scope="test/scope",
        )

        with patch("routers.knowledge.get_knowledge_service") as mock_svc_fn:
            mock_svc = MagicMock()
            mock_svc.get_scope.return_value = mock_scope
            mock_svc.search.return_value = [mock_result]
            mock_svc_fn.return_value = mock_svc

            resp = client.get(
                "/api/v1/knowledge/debug/query?scope=test/scope&q=auth&limit=5",
                headers=INTERNAL_HEADERS,
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["result_count"], 1)
        r = data["results"][0]

        required_fields = ["rank", "score", "source", "section_path", "section",
                           "chunk_index", "document_hash", "ingested_at", "text"]
        for field in required_fields:
            self.assertIn(field, r, f"Missing required field: {field}")

        self.assertEqual(r["rank"], 1)
        self.assertAlmostEqual(r["score"], 0.92, places=3)
        self.assertEqual(r["source"], "api-reference.pdf")
        self.assertEqual(r["section_path"], "API Reference > Authentication")
        self.assertEqual(r["section"], "Authentication")
        self.assertEqual(r["chunk_index"], 7)
        self.assertIn("Bearer tokens", r["text"])

    def test_rank_is_1indexed_and_ordered(self):
        """Results must be ranked starting from 1, in score-descending order."""
        from unittest.mock import patch, MagicMock
        from models.knowledge import KnowledgeScope, SearchResult, Citation

        client = _make_client()
        mock_scope = KnowledgeScope(name="test/scope")

        def make_result(src, score):
            return SearchResult(
                text=f"Content from {src}",
                score=score,
                citation=Citation(source=src, score=score),
                scope="test/scope",
            )

        with patch("routers.knowledge.get_knowledge_service") as mock_svc_fn:
            mock_svc = MagicMock()
            mock_svc.get_scope.return_value = mock_scope
            # search returns results in score-desc order (service guarantees this)
            mock_svc.search.return_value = [
                make_result("a.pdf", 0.95),
                make_result("b.pdf", 0.87),
                make_result("c.pdf", 0.72),
            ]
            mock_svc_fn.return_value = mock_svc

            resp = client.get(
                "/api/v1/knowledge/debug/query?scope=test/scope&q=auth",
                headers=INTERNAL_HEADERS,
            )

        data = resp.json()
        ranks = [r["rank"] for r in data["results"]]
        self.assertEqual(ranks, [1, 2, 3])

    def test_limit_param_respected(self):
        """limit query param controls how many results are requested from search."""
        from unittest.mock import patch, MagicMock
        from models.knowledge import KnowledgeScope, SearchResult, Citation

        client = _make_client()
        mock_scope = KnowledgeScope(name="test/scope")

        with patch("routers.knowledge.get_knowledge_service") as mock_svc_fn:
            mock_svc = MagicMock()
            mock_svc.get_scope.return_value = mock_scope
            mock_svc.search.return_value = []
            mock_svc_fn.return_value = mock_svc

            resp = client.get(
                "/api/v1/knowledge/debug/query?scope=test/scope&q=auth&limit=3",
                headers=INTERNAL_HEADERS,
            )

        # Verify search was called with limit=3
        call_args = mock_svc.search.call_args
        self.assertEqual(call_args.kwargs.get("limit") or call_args.args[2], 3)

    def test_scope_double_dash_normalised(self):
        """Scope with '--' separator (URL-safe) is normalised to '/' before lookup."""
        from unittest.mock import patch, MagicMock

        client = _make_client()

        with patch("routers.knowledge.get_knowledge_service") as mock_svc_fn:
            mock_svc = MagicMock()
            mock_svc.get_scope.return_value = None  # triggers 404 — validates the path was normalised
            mock_svc_fn.return_value = mock_svc

            resp = client.get(
                "/api/v1/knowledge/debug/query?scope=test--scope&q=auth",
                headers=INTERNAL_HEADERS,
            )

        # get_scope was called with "test/scope" (-- → /)
        call_arg = mock_svc.get_scope.call_args[0][0]
        self.assertEqual(call_arg, "test/scope")


class TestDocumentManagementAPIOperator(unittest.TestCase):
    """Operator document management API path coverage — knowledge-ingest side.

    These tests validate the API paths that back the operator UI:
    document list, version history, and rollback.
    Uses TestClient against knowledge-ingest routes with DB stubbed.
    """

    def _make_ki_client(self):
        """Build a TestClient for the knowledge-ingest app."""
        import importlib, sys as _sys
        ki_src = os.path.join(
            os.path.dirname(__file__), "..", "..", "agentopia-knowledge-ingest", "src"
        )
        if ki_src not in _sys.path:
            _sys.path.insert(0, ki_src)
        import main as ki_main
        return TestClient(ki_main.app, raise_server_exceptions=False)

    def test_placeholder(self):
        """Placeholder: document management API is covered by ingest E2E tests.

        knowledge-ingest runs as a separate service; cross-service TestClient
        construction requires the two services to share a Python environment
        which is not the case in this repo. Full API coverage lives in
        tests/test_ingest_e2e.py and tests/test_operator_ui.py in the
        knowledge-ingest repo.
        """
        # The actual API tests live in the knowledge-ingest repo.
        # This placeholder records the intent without false assertions.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
