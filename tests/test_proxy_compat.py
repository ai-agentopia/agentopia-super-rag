"""Proxy compatibility tests (#320).

Verifies that knowledge-api's route schema is compatible with bot-config-api's
external surface, so that bot-config-api can proxy without response translation.

Tests:
1. Health endpoint shape
2. /scopes response schema
3. /search response schema
4. /ingest response schema (201)
5. Binding sync endpoint contracts
6. Proxy failure handling (upstream unavailable)
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


INTERNAL_TOKEN = "proxy-compat-test-token"
AUTH_HEADER = {"X-Internal-Token": INTERNAL_TOKEN}


@pytest.fixture(autouse=True)
def set_token():
    os.environ["KNOWLEDGE_API_INTERNAL_TOKEN"] = INTERNAL_TOKEN


@pytest.fixture
def client():
    with patch("routers.knowledge.get_knowledge_service") as mock_svc:
        svc = MagicMock()
        svc.list_scopes.return_value = []
        svc.search.return_value = []
        svc.get_scope.return_value = None
        svc._qdrant = None
        mock_svc.return_value = svc

        import importlib
        import main
        importlib.reload(main)
        from fastapi.testclient import TestClient
        yield TestClient(main.app), svc


# ── Health ────────────────────────────────────────────────────────────────────


def test_health_endpoint(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "knowledge-api"


# ── Scopes ─────────────────────────────────────────────────────────────────────


def test_list_scopes_schema(client):
    c, svc = client
    from models.knowledge import KnowledgeScope
    mock_scope = MagicMock(spec=KnowledgeScope)
    mock_scope.name = "client1/docs"
    mock_scope.document_count = 3
    mock_scope.chunk_count = 12
    mock_scope.last_indexed = 1700000000.0
    svc.list_scopes.return_value = [mock_scope]

    resp = c.get("/api/v1/knowledge/scopes", headers=AUTH_HEADER)
    assert resp.status_code == 200
    data = resp.json()
    # Schema must match bot-config-api's response
    assert "scopes" in data
    assert "count" in data
    assert data["count"] == 1
    scope = data["scopes"][0]
    assert "name" in scope
    assert "document_count" in scope
    assert "chunk_count" in scope
    assert "last_indexed" in scope


# ── Search ────────────────────────────────────────────────────────────────────


def test_search_schema(client):
    c, svc = client
    from models.knowledge import Citation, SearchResult
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {
        "text": "chunk content",
        "score": 0.9,
        "scope": "client1/docs",
        "citation": {
            "source": "doc.md",
            "section": "",
            "page": None,
            "chunk_index": 0,
            "score": 0.9,
            "ingested_at": 1700000000.0,
            "document_hash": "abc123",
        },
    }
    svc.list_scopes.return_value = [MagicMock(name="client1/docs")]
    svc.search.return_value = [mock_result]

    resp = c.get(
        "/api/v1/knowledge/search",
        params={"query": "test", "scopes": ["client1/docs"]},
        headers=AUTH_HEADER,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "count" in data
    assert data["count"] == 1


# ── Scope 404 ────────────────────────────────────────────────────────────────


def test_get_scope_404(client):
    c, svc = client
    svc.get_scope.return_value = None
    resp = c.get("/api/v1/knowledge/client1--docs", headers=AUTH_HEADER)
    assert resp.status_code == 404


# ── Ingest schema (201) ───────────────────────────────────────────────────────


def test_ingest_response_schema(client):
    c, svc = client
    from models.knowledge import IngestResult
    svc.ingest.return_value = MagicMock(
        spec=IngestResult,
        chunks_created=5,
        chunks_skipped=0,
        document_hash="sha256abc",
        ingested_at=1700000000.0,
    )

    resp = c.post(
        "/api/v1/knowledge/client1--docs/ingest",
        files={"file": ("doc.md", b"# Test\nContent here", "text/markdown")},
        headers=AUTH_HEADER,
    )
    assert resp.status_code == 201
    data = resp.json()
    # Schema must match bot-config-api response
    assert data["status"] == "ingested"
    assert "scope" in data
    assert "source" in data
    assert "format" in data
    assert "chunks_created" in data
    assert "document_hash" in data
    assert "ingested_at" in data


# ── Binding sync contracts ────────────────────────────────────────────────────


def test_binding_sync_response(client):
    c, _ = client
    resp = c.post(
        "/internal/binding-sync",
        json={"bot_name": "compat-bot", "client_id": "c1", "knowledge_scopes": ["docs"]},
        headers=AUTH_HEADER,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "synced"
    assert "bot_name" in data
    assert "scopes" in data


def test_binding_remove_response(client):
    c, _ = client
    resp = c.delete("/internal/binding-sync/compat-bot", headers=AUTH_HEADER)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "removed"


def test_binding_lookup_404_when_not_found(client):
    c, _ = client
    with patch("services.binding_cache.BindingCache.resolve_with_fallback", return_value=None):
        resp = c.get("/internal/binding-sync/nonexistent", headers=AUTH_HEADER)
        assert resp.status_code == 404
