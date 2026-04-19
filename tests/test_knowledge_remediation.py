"""Wave 2 Track B — Knowledge base remediation tests.

Covers:
  #23  Qdrant runtime readiness — graceful fallback, auto-select logic
  #24  Webhook indexing — prove full flow + idempotency
  #100 Scope isolation — collection-per-scope, query isolation
  #101 Stale detection / reindex — is_stale(), reindex(), list_stale_scopes()
  #106 Knowledge lifecycle APIs — GET scope, list docs, delete doc, reindex endpoint
  #25  Graph RAG — BLOCKER to M3.1 (handoff artifact)
"""

import os
import time
import pytest
from unittest.mock import MagicMock, patch


def _auth_headers():
    token = os.getenv("KNOWLEDGE_API_INTERNAL_TOKEN", "")
    return {"X-Internal-Token": token} if token else {}

from models.knowledge import DocumentFormat, KnowledgeScope
from services.knowledge import KnowledgeService, get_knowledge_service


@pytest.fixture()
def svc():
    """Fresh KnowledgeService (in-memory, no Qdrant)."""
    return KnowledgeService()


# ── #23 Qdrant Runtime Readiness ──────────────────────────────────────────


class TestQdrantRuntimeReadiness:
    """Prove Qdrant graceful fallback path (#23)."""

    def test_no_qdrant_url_uses_in_memory(self, monkeypatch):
        """Without QDRANT_URL, singleton uses in-memory backend."""
        import services.knowledge as kmod

        kmod._knowledge = None  # reset singleton
        monkeypatch.delenv("QDRANT_URL", raising=False)
        svc = get_knowledge_service()
        assert svc._qdrant is None
        kmod._knowledge = None  # cleanup

    def test_qdrant_url_set_but_unreachable_falls_back(self, monkeypatch):
        """QDRANT_URL set but connection fails → in-memory fallback (no crash)."""
        import services.knowledge as kmod

        kmod._knowledge = None
        monkeypatch.setenv("QDRANT_URL", "http://localhost:29999")  # unreachable port

        with patch(
            "services.knowledge.QdrantBackend.__init__",
            side_effect=Exception("conn refused"),
        ):
            svc = get_knowledge_service()
        # Falls back to in-memory
        assert svc._qdrant is None
        kmod._knowledge = None  # cleanup

    def test_qdrant_backend_ingest_errors_dont_crash_service(self, svc):
        """If Qdrant.ingest_chunks() raises, service continues (in-memory still works)."""
        mock_qdrant = MagicMock()
        mock_qdrant.ingest_chunks.side_effect = Exception("Qdrant write error")
        svc._qdrant = mock_qdrant

        # Should not raise; in-memory path succeeds
        result = svc.ingest("test-scope", "A" * 100, source="test.py")
        assert result.chunks_created >= 1

    def test_qdrant_format_ingestion_code(self, svc):
        """CODE format uses code-aware chunking."""
        content = "def foo():\n    pass\n\ndef bar():\n    return 1"
        result = svc.ingest(
            "code-scope", content, source="app.py", format=DocumentFormat.CODE
        )
        assert result.chunks_created >= 1

    def test_qdrant_format_ingestion_markdown(self, svc):
        """MARKDOWN format uses paragraph chunking."""
        content = "# Title\n\nParagraph one.\n\nParagraph two."
        result = svc.ingest(
            "md-scope", content, source="README.md", format=DocumentFormat.MARKDOWN
        )
        assert result.chunks_created >= 1


# ── #24 Webhook Indexing Flow ─────────────────────────────────────────────


class TestWebhookIndexingFlow:
    """Prove webhook-triggered indexing flow and idempotency (#24)."""

    def test_webhook_ingest_creates_chunks(self, svc):
        """Ingest via service (simulating webhook call) creates chunks in scope."""
        content = "def authenticate(token):\n    return token == 'secret'" * 5
        result = svc.ingest(
            "auth-scope", content, source="auth.py", format=DocumentFormat.CODE
        )
        assert result.chunks_created >= 1
        assert result.scope == "auth-scope"
        assert result.source == "auth.py"

    def test_webhook_idempotency_same_content_zero_new_chunks(self, svc):
        """Re-ingesting same content → chunks_created=0 (dedup)."""
        content = "API_KEY = 'abc123'" * 10
        r1 = svc.ingest("cfg-scope", content, source="config.py")
        r2 = svc.ingest("cfg-scope", content, source="config.py")
        assert r1.chunks_created > 0
        assert r2.chunks_created == 0  # dedup: no new chunks

    def test_webhook_partial_change_indexes_only_new_chunks(self, svc):
        """Changed content → new chunks indexed; completely unchanged chunks deduped."""
        # Ingest large base content that produces multiple chunks
        base = (
            "stable line content that will not change\n" * 120
        )  # > 512 chars → 2+ chunks
        result1 = svc.ingest("patch-scope", base, source="file.txt")
        assert result1.chunks_created >= 2

        # Re-ingest same content — all deduped
        result_same = svc.ingest("patch-scope", base, source="file.txt")
        assert result_same.chunks_created == 0

        # Add small new content → produces at least 1 new chunk
        new_section = "\n\nnew unique content added here\n" * 20
        result2 = svc.ingest("patch-scope", new_section, source="new_section.txt")
        assert result2.chunks_created > 0  # new chunks added

    def test_webhook_endpoint_retired(self):
        """POST /api/v1/knowledge/webhook returns 410 Gone (#303 — retired)."""
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app, headers=_auth_headers()) as client:
            resp = client.post(
                "/api/v1/knowledge/webhook",
                params={
                    "scope": "webhook-test",
                    "source": "src/api.py",
                    "content": "def handler(): pass\n" * 20,
                    "format": "code",
                },
            )
        assert resp.status_code == 410
        assert "retired" in resp.json()["detail"].lower()


# ── #100 Scope Isolation ──────────────────────────────────────────────────


class TestScopeIsolation:
    """Prove collection-per-scope isolation (#100)."""

    def test_search_only_returns_results_from_requested_scope(self, svc):
        """Search in scope-a does not return chunks from scope-b."""
        svc.ingest("scope-a", "Python async programming patterns", source="a.txt")
        svc.ingest("scope-b", "Kubernetes deployment YAML examples", source="b.txt")

        results_a = svc.search("Python", scopes=["scope-a"])
        results_b = svc.search("Kubernetes", scopes=["scope-b"])

        for r in results_a:
            assert r.scope == "scope-a"
        for r in results_b:
            assert r.scope == "scope-b"

    def test_delete_scope_does_not_affect_other_scopes(self, svc):
        """Deleting scope-a leaves scope-b intact."""
        svc.ingest(
            "keep-scope", "Persistent content that must survive", source="keep.txt"
        )
        svc.ingest("del-scope", "Temporary content to delete", source="del.txt")

        svc.delete_scope("del-scope")

        # keep-scope still has content
        results = svc.search("Persistent content", scopes=["keep-scope"])
        assert len(results) > 0

        # del-scope is gone
        assert svc.get_scope("del-scope") is None

    def test_cross_scope_search_merges_results(self, svc):
        """Searching both scopes returns results from both."""
        svc.ingest("ns-x", "FastAPI async endpoint routing", source="api.py")
        svc.ingest("ns-y", "FastAPI middleware configuration", source="mid.py")

        results = svc.search("FastAPI", scopes=["ns-x", "ns-y"], limit=10)
        scopes_hit = {r.scope for r in results}
        assert "ns-x" in scopes_hit
        assert "ns-y" in scopes_hit

    def test_chunks_from_different_scopes_not_mixed(self, svc):
        """Unique content per scope → search returns only matching scope."""
        svc.ingest(
            "billing-scope",
            "invoice_number payment_method stripe_charge",
            source="billing.py",
        )
        svc.ingest(
            "auth-scope", "jwt_token user_session login_attempt", source="auth.py"
        )

        billing_results = svc.search("invoice", scopes=["billing-scope"])
        assert all(r.scope == "billing-scope" for r in billing_results)
        # auth-scope not polluted
        auth_results = svc.search("invoice", scopes=["auth-scope"])
        assert len(auth_results) == 0


# ── #101 Stale Detection and Reindex ──────────────────────────────────────


class TestStaleDetectionAndReindex:
    """Stale policy trigger path and reindex flow (#101)."""

    def test_never_indexed_scope_is_stale(self):
        scope = KnowledgeScope(name="fresh", last_indexed=0.0)
        assert scope.is_stale(max_age_secs=3600) is True

    def test_recently_indexed_scope_not_stale(self):
        scope = KnowledgeScope(name="recent", last_indexed=time.time())
        assert scope.is_stale(max_age_secs=3600) is False

    def test_old_scope_is_stale(self):
        two_days_ago = time.time() - 48 * 3600
        scope = KnowledgeScope(name="old", last_indexed=two_days_ago)
        assert scope.is_stale(max_age_secs=86400) is True

    def test_list_stale_scopes_identifies_old(self, svc):
        """list_stale_scopes() returns scopes past the threshold."""
        # Ingest then manually set last_indexed to old timestamp
        svc.ingest("stale-a", "A" * 100, source="a.txt")
        svc.ingest("fresh-b", "B" * 100, source="b.txt")

        svc._scopes["stale-a"].last_indexed = time.time() - 90000  # 25h ago
        svc._scopes["fresh-b"].last_indexed = time.time()

        stale = svc.list_stale_scopes(max_age_secs=86400)
        assert "stale-a" in stale
        assert "fresh-b" not in stale

    def test_reindex_clears_dedup_cache(self, svc):
        """After reindex(), same content is re-indexed (not deduped)."""
        content = "stable content to verify reindex\n" * 10
        r1 = svc.ingest("ri-scope", content, source="stable.txt")
        assert r1.chunks_created > 0

        r2 = svc.ingest("ri-scope", content, source="stable.txt")
        assert r2.chunks_created == 0  # deduped

        # Trigger reindex
        result = svc.reindex("ri-scope")
        assert result["status"] == "reindex_triggered"

        # Now same content should be re-indexed
        r3 = svc.ingest("ri-scope", content, source="stable.txt")
        assert r3.chunks_created > 0

    def test_reindex_nonexistent_scope_returns_not_found(self, svc):
        result = svc.reindex("nonexistent-scope-xyz")
        assert result["status"] == "not_found"

    def test_reindex_api_endpoint(self):
        """POST /api/v1/knowledge/{scope}/reindex returns reindex_triggered."""
        from fastapi.testclient import TestClient
        from main import app
        from services.knowledge import get_knowledge_service
        from models.knowledge import DocumentFormat

        # Seed via service layer (POST /ingest retired in P4.5 — use svc directly).
        get_knowledge_service().ingest(
            scope="ri-api-test", content="A" * 100, source="x.txt",
            format=DocumentFormat.TEXT,
        )
        with TestClient(app, headers=_auth_headers()) as client:
            resp = client.post("/api/v1/knowledge/ri-api-test/reindex")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reindex_triggered"

    def test_reindex_nonexistent_scope_api_returns_404(self):
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app, headers=_auth_headers()) as client:
            resp = client.post("/api/v1/knowledge/no-such-scope-xyz/reindex")
        assert resp.status_code == 404


# ── #106 Knowledge Lifecycle APIs ─────────────────────────────────────────


class TestKnowledgeLifecycleAPIs:
    """Full lifecycle: get-scope, list-docs, delete-doc, reindex (#106)."""

    def test_get_scope_endpoint(self):
        """GET /api/v1/knowledge/{scope} returns scope metadata."""
        from fastapi.testclient import TestClient
        from main import app
        from services.knowledge import get_knowledge_service
        from models.knowledge import DocumentFormat

        get_knowledge_service().ingest(
            scope="meta-scope", content="B" * 100, source="m.txt",
            format=DocumentFormat.TEXT,
        )
        with TestClient(app, headers=_auth_headers()) as client:
            resp = client.get("/api/v1/knowledge/meta-scope")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "meta-scope"
        assert data["chunk_count"] >= 1

    def test_get_scope_not_found_returns_404(self):
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app, headers=_auth_headers()) as client:
            resp = client.get("/api/v1/knowledge/no-such-scope-404")
        assert resp.status_code == 404

    def test_list_documents_endpoint(self):
        """GET /api/v1/knowledge/{scope}/documents lists sources."""
        from fastapi.testclient import TestClient
        from main import app
        from services.knowledge import get_knowledge_service
        from models.knowledge import DocumentFormat

        svc = get_knowledge_service()
        svc.ingest(scope="docs-scope", content="C" * 100, source="file1.py", format=DocumentFormat.CODE)
        svc.ingest(scope="docs-scope", content="D" * 100, source="file2.py", format=DocumentFormat.CODE)
        with TestClient(app, headers=_auth_headers()) as client:
            resp = client.get("/api/v1/knowledge/docs-scope/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope"] == "docs-scope"
        sources = [d["source"] for d in data["documents"]]
        assert "file1.py" in sources
        assert "file2.py" in sources

    def test_list_documents_service_level(self, svc):
        """list_documents() returns per-source chunk counts."""
        svc.ingest("multi-doc", "Alpha content\n" * 20, source="alpha.py")
        svc.ingest("multi-doc", "Beta content\n" * 20, source="beta.py")

        docs = svc.list_documents("multi-doc")
        sources = {d["source"] for d in docs}
        assert "alpha.py" in sources
        assert "beta.py" in sources
        for d in docs:
            assert d["chunk_count"] >= 1

    def test_delete_document_service_level(self, svc):
        """delete_document() removes chunks for a specific source."""
        svc.ingest("del-doc-scope", "Delete me content\n" * 20, source="delete.py")
        svc.ingest("del-doc-scope", "Keep me content\n" * 20, source="keep.py")

        chunk_count_before = svc._scopes["del-doc-scope"].chunk_count
        removed = svc.delete_document("del-doc-scope", "delete.py")
        assert removed > 0
        assert svc._scopes["del-doc-scope"].chunk_count == chunk_count_before - removed

        # keep.py still searchable
        results = svc.search("Keep me", scopes=["del-doc-scope"])
        assert len(results) > 0

    def test_delete_document_invalidates_hash_for_reindex(self, svc):
        """After delete_document(), re-ingesting same content creates new chunks."""
        content = "Function to delete and re-add\n" * 10
        svc.ingest("hash-test", content, source="fn.py")
        svc.delete_document("hash-test", "fn.py")
        # Re-ingest: should create chunks (hash cleared)
        r = svc.ingest("hash-test", content, source="fn.py")
        assert r.chunks_created > 0

    def test_stale_scopes_endpoint(self):
        """GET /api/v1/knowledge/stale returns list of stale scopes."""
        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app, headers=_auth_headers()) as client:
            resp = client.get("/api/v1/knowledge/stale?max_age_secs=0.001")
        # All scopes should be stale with effectively 0 max age
        assert resp.status_code == 200
        data = resp.json()
        assert "stale_scopes" in data
        assert "count" in data

    def test_full_lifecycle_ingest_list_delete_verify(self, svc):
        """Full lifecycle: ingest → list → delete → verify gone."""
        # 1. Ingest
        svc.ingest(
            "lifecycle",
            "Document content for lifecycle test\n" * 15,
            source="lifecycle.md",
        )
        assert len(svc.list_documents("lifecycle")) == 1

        # 2. Verify searchable
        results = svc.search("lifecycle test", scopes=["lifecycle"])
        assert len(results) > 0

        # 3. Delete document
        removed = svc.delete_document("lifecycle", "lifecycle.md")
        assert removed > 0

        # 4. Document no longer listed
        docs = svc.list_documents("lifecycle")
        assert len(docs) == 0

        # 5. No longer searchable
        results_after = svc.search("lifecycle test", scopes=["lifecycle"])
        assert len(results_after) == 0


# ── #25 Graph RAG — BLOCKER Handoff ───────────────────────────────────────


class TestGraphRAGBlocker:
    """#25 Graph RAG: formally BLOCKED to M3.1.

    Handoff artifact:
      - Dependency: separate graph store (e.g., Neo4j already running in cluster)
      - Required: entity extraction pipeline (spaCy/LLM-based NER)
      - Required: graph-augmented retrieval combining vector + graph paths
      - Owner: M3.1 milestone
      - Unblock condition: graph store API contract defined and Neo4j schema documented
    """

    def test_graph_rag_blocker_formal(self):
        """Graph RAG is explicitly blocked — no implementation in M3."""
        # Verify that no graph_rag module exists in the codebase
        import importlib

        with pytest.raises((ImportError, ModuleNotFoundError)):
            importlib.import_module("services.graph_rag")

    def test_search_does_not_perform_graph_augmentation(self, svc):
        """Current search is vector/text-only — no graph traversal (#25 not active)."""
        svc.ingest("g-scope", "Function calls graph traversal algorithm", source="g.py")
        results = svc.search("graph", scopes=["g-scope"])
        # Results exist but come from in-memory text match, not graph
        for r in results:
            assert hasattr(r, "citation")  # normal vector citation format
            # No graph_path attribute (graph RAG would add this)
            assert not hasattr(r, "graph_path")
