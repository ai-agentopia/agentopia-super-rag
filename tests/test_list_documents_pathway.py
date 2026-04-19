"""Tests for list_documents over Pathway-managed (Qdrant-only) scopes.

Pathway writes chunks directly to Qdrant and does NOT populate knowledge-api's
in-memory `_scopes` registry or DocumentStore. Before P4.5 the list_documents
route pre-checked the in-memory registry and returned 404 for every
Pathway-managed scope. These tests lock in the new behaviour:

  - QdrantBackend.list_documents aggregates unique document_ids from a
    collection by scrolling points + payloads.
  - KnowledgeService.list_documents merges DocumentStore + Qdrant results and
    returns an empty list rather than raising for unknown scopes.
  - The HTTP route no longer 404s on Pathway scopes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from services.knowledge import KnowledgeService, QdrantBackend


def _make_point(payload):
    return SimpleNamespace(payload=payload, id="pt", vector=None)


def _mock_qdrant_client(points):
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(points_count=len(points))
    client.scroll.return_value = (points, None)
    return client


def test_qdrant_list_documents_aggregates_pathway_payloads():
    points = [
        _make_point({
            "document_id": "architecture/overview.md",
            "document_hash": "hash-overview",
            "ingested_at": 1_776_000_000.0,
        }),
        _make_point({
            "document_id": "architecture/overview.md",
            "document_hash": "hash-overview",
            "ingested_at": 1_776_000_100.0,
        }),
        _make_point({
            "document_id": "architecture/repo-split.md",
            "document_hash": "hash-repo",
            "ingested_at": 1_776_000_200.0,
        }),
    ]

    backend = QdrantBackend.__new__(QdrantBackend)
    backend._client = _mock_qdrant_client(points)
    # Use the real classmethod so we exercise the format inference path.
    rows = QdrantBackend.list_documents(backend, "utop/oddspark")

    by_source = {r["source"]: r for r in rows}
    assert by_source["architecture/overview.md"]["chunk_count"] == 2
    assert by_source["architecture/overview.md"]["ingested_at"] == 1_776_000_100.0
    assert by_source["architecture/overview.md"]["document_hash"] == "hash-overview"
    assert by_source["architecture/overview.md"]["format"] == "markdown"
    assert by_source["architecture/repo-split.md"]["chunk_count"] == 1
    assert by_source["architecture/repo-split.md"]["format"] == "markdown"


def test_qdrant_list_documents_supports_legacy_nested_metadata():
    """Legacy direct-ingest stored source inside payload.metadata."""
    points = [
        _make_point({
            "metadata": {
                "source": "legacy/doc.pdf",
                "document_hash": "legacy-hash",
                "ingested_at": 1.0,
            },
        }),
    ]
    backend = QdrantBackend.__new__(QdrantBackend)
    backend._client = _mock_qdrant_client(points)
    rows = QdrantBackend.list_documents(backend, "legacy/scope")
    assert len(rows) == 1
    assert rows[0]["source"] == "legacy/doc.pdf"
    assert rows[0]["format"] == "pdf"
    assert rows[0]["document_hash"] == "legacy-hash"


def test_qdrant_list_documents_missing_collection_returns_empty():
    backend = QdrantBackend.__new__(QdrantBackend)
    client = MagicMock()
    client.get_collection.side_effect = Exception("Not found: collection missing")
    backend._client = client
    assert QdrantBackend.list_documents(backend, "utop/oddspark") == []


def test_service_list_documents_merges_doc_store_and_qdrant():
    """DocumentStore records win, but Qdrant-only sources are still included."""
    svc = KnowledgeService.__new__(KnowledgeService)
    # Minimal shape KnowledgeService.list_documents inspects.
    svc._chunks = {}

    doc_record = SimpleNamespace(
        source="legacy/store.md",
        chunk_count=4,
        scope="utop/oddspark",
        document_hash="legacy-hash",
        ingested_at=100.0,
        format=SimpleNamespace(value="markdown"),
    )
    svc._doc_store = SimpleNamespace(list_active=lambda scope: [doc_record])

    qdrant_row = {
        "source": "architecture/pathway-only.md",
        "chunk_count": 7,
        "document_hash": "pathway-hash",
        "ingested_at": 200.0,
        "format": "markdown",
    }
    svc._qdrant = SimpleNamespace(list_documents=lambda scope: [qdrant_row])

    result = KnowledgeService.list_documents(svc, "utop/oddspark")
    sources = {row["source"]: row for row in result}

    # Both universes must be represented.
    assert sources["legacy/store.md"]["chunk_count"] == 4
    assert sources["architecture/pathway-only.md"]["chunk_count"] == 7
    # DocumentStore entry stays canonical even when Qdrant also has chunks
    # for the same source — no double count.
    qdrant_overlap = {
        "source": "legacy/store.md",
        "chunk_count": 99,
        "document_hash": "should-not-win",
        "ingested_at": 999.0,
        "format": "text",
    }
    svc._qdrant = SimpleNamespace(
        list_documents=lambda scope: [qdrant_row, qdrant_overlap],
    )
    result2 = KnowledgeService.list_documents(svc, "utop/oddspark")
    overlap = next(r for r in result2 if r["source"] == "legacy/store.md")
    assert overlap["chunk_count"] == 4
    assert overlap["document_hash"] == "legacy-hash"


@pytest.fixture
def client(monkeypatch):
    """Route-level client with a stubbed KnowledgeService.

    Uses the already-imported `main.app` so we don't disturb the global
    auth state for other tests in the session (test_operator_api.py
    shares the same app).
    """
    import main
    from unittest.mock import patch

    svc = MagicMock()
    svc.list_documents.return_value = [
        {
            "source": "architecture/overview.md",
            "chunk_count": 5,
            "scope": "utop/oddspark",
            "document_hash": "abc123",
            "ingested_at": 123.0,
            "format": "markdown",
        }
    ]
    # Absent scope in the in-memory registry — would have 404'd before P4.5.
    svc.get_scope.return_value = None

    with patch("routers.knowledge.get_knowledge_service", return_value=svc):
        yield TestClient(main.app), svc


def test_documents_route_returns_200_for_pathway_only_scope(client):
    c, svc = client
    import os
    token = os.getenv("KNOWLEDGE_API_INTERNAL_TOKEN", "")
    headers = {"X-Internal-Token": token} if token else {}
    resp = c.get("/api/v1/knowledge/utop--oddspark/documents", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["scope"] == "utop/oddspark"
    assert len(data["documents"]) == 1
    assert data["documents"][0]["source"] == "architecture/overview.md"
    # The route must no longer gate on get_scope().
    svc.list_documents.assert_called_once_with("utop/oddspark")
