"""Phase 0 Foundation Hardening tests (#316).

Tests for retry/backoff, circuit breaker, externalized config,
score_threshold, Qdrant health, and source_type metadata.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

# ── source_type model tests ──────────────────────────────────────────────────


def test_source_type_enum_values():
    from models.knowledge import SourceType
    assert SourceType.BUSINESS_DOC == "business_doc"
    assert SourceType.CODE_FILE == "code_file"
    assert SourceType.FEATURE_ARTIFACT == "feature_artifact"


def test_document_metadata_default_source_type():
    from models.knowledge import DocumentMetadata, SourceType
    meta = DocumentMetadata(source="test.pdf")
    assert meta.source_type == SourceType.BUSINESS_DOC


def test_document_metadata_explicit_source_type():
    from models.knowledge import DocumentMetadata, SourceType
    meta = DocumentMetadata(source="main.py", source_type=SourceType.CODE_FILE)
    assert meta.source_type == SourceType.CODE_FILE


def test_document_record_default_source_type():
    from models.knowledge import DocumentRecord, SourceType
    record = DocumentRecord(scope="t/s", source="doc.pdf", document_hash="abc")
    assert record.source_type == SourceType.BUSINESS_DOC


def test_document_record_serialization_roundtrip():
    from models.knowledge import DocumentRecord, SourceType
    record = DocumentRecord(
        scope="t/s", source="main.py", document_hash="abc",
        source_type=SourceType.CODE_FILE,
    )
    data = record.model_dump()
    restored = DocumentRecord(**data)
    assert restored.source_type == SourceType.CODE_FILE


def test_document_metadata_backward_compatible():
    """Data without source_type (e.g., from old Qdrant payload) defaults correctly."""
    from models.knowledge import DocumentMetadata, SourceType
    data = {"source": "test.pdf", "format": "text", "scope": "s1", "chunk_index": 0}
    meta = DocumentMetadata(**data)
    assert meta.source_type == SourceType.BUSINESS_DOC


# ── Embedding config externalization tests ───────────────────────────────────

# These tests use a helper that constructs QdrantBackend with mocked qdrant_client.

def _make_backend(**env_overrides):
    """Create a QdrantBackend with mocked Qdrant client and optional env overrides."""
    # Clean env
    for key in ["EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "EMBEDDING_TIMEOUT_SECONDS",
                 "EMBEDDING_VECTOR_DIMENSION"]:
        os.environ.pop(key, None)
    os.environ.update(env_overrides)

    mock_qdrant_module = MagicMock()
    mock_client = MagicMock()
    mock_qdrant_module.QdrantClient.return_value = mock_client

    with patch.dict("sys.modules", {"qdrant_client": mock_qdrant_module, "qdrant_client.models": mock_qdrant_module}):
        mock_qdrant_module.Distance = MagicMock()
        mock_qdrant_module.VectorParams = MagicMock()
        mock_qdrant_module.PointStruct = MagicMock()
        from services.knowledge import QdrantBackend
        backend = QdrantBackend("http://localhost:6333")

    # Clean up
    for key in env_overrides:
        os.environ.pop(key, None)

    return backend, mock_client


def test_qdrant_backend_default_config():
    backend, _ = _make_backend()
    assert backend.VECTOR_SIZE == 1536  # code default matches production dimension
    assert backend._embedding_model == "openai/text-embedding-3-small"
    assert backend._embedding_base_url == "https://openrouter.ai/api/v1/embeddings"
    assert backend._embedding_timeout == 30


def test_qdrant_backend_env_override():
    backend, _ = _make_backend(
        EMBEDDING_MODEL="test/model",
        EMBEDDING_BASE_URL="http://test:8080/embed",
        EMBEDDING_TIMEOUT_SECONDS="10",
        EMBEDDING_VECTOR_DIMENSION="768",
    )
    assert backend.VECTOR_SIZE == 768
    assert backend._embedding_model == "test/model"
    assert backend._embedding_base_url == "http://test:8080/embed"
    assert backend._embedding_timeout == 10


# ── Retry / backoff tests ────────────────────────────────────────────────────


def test_embed_retries_on_transient_failure():
    backend, _ = _make_backend()
    backend._cb_failures = 0
    backend._cb_open_until = 0.0

    call_count = 0
    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            import httpx
            raise httpx.ConnectError("transient")
        resp = MagicMock()
        resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
        return resp

    with patch("httpx.post", side_effect=mock_post), \
         patch("time.sleep"):
        result = backend._embed(["test text"])

    assert call_count == 3
    assert result == [[0.1, 0.2]]
    assert backend._cb_failures == 0


def test_embed_exhausts_retries():
    import httpx
    backend, _ = _make_backend()
    backend._cb_failures = 0
    backend._cb_open_until = 0.0

    with patch("httpx.post", side_effect=httpx.ConnectError("persistent")), \
         patch("time.sleep"):
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            backend._embed(["test text"])

    assert backend._cb_failures == 1


# ── Circuit breaker tests ────────────────────────────────────────────────────


def test_circuit_breaker_opens_after_threshold():
    backend, _ = _make_backend()
    backend._cb_failures = 4
    backend._cb_record_failure()  # 5th
    assert backend._cb_is_open()


def test_circuit_breaker_fast_fails():
    backend, _ = _make_backend()
    backend._cb_open_until = time.time() + 300
    with pytest.raises(RuntimeError, match="EmbeddingCircuitBreaker: OPEN"):
        backend._embed(["test"])


def test_circuit_breaker_recovers_after_cooldown():
    backend, _ = _make_backend()
    backend._cb_open_until = time.time() - 1
    backend._cb_failures = 5
    assert not backend._cb_is_open()
    assert backend._cb_failures == 0


def test_circuit_breaker_resets_on_success():
    backend, _ = _make_backend()
    backend._cb_failures = 3
    backend._cb_open_until = 0.0

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}
    with patch("httpx.post", return_value=resp):
        backend._embed(["test"])
    assert backend._cb_failures == 0


# ── score_threshold tests ────────────────────────────────────────────────────


def test_search_scope_passes_score_threshold():
    backend, mock_client = _make_backend()
    # Phase 2a: _search_dense_only uses query_points (qdrant-client v1.17+), not search
    mock_client.query_points.return_value = MagicMock(points=[])
    backend._collection_cache.add("kb-337970215e7c494a")  # hash of "test-scope"

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1536}]}
    with patch("httpx.post", return_value=resp):
        backend.search_scope("query", "test-scope", limit=5, min_score=0.5)

    call_kwargs = mock_client.query_points.call_args[1]
    assert call_kwargs.get("score_threshold") == 0.5


def test_search_scope_omits_threshold_when_zero():
    backend, mock_client = _make_backend()
    mock_client.query_points.return_value = MagicMock(points=[])
    backend._collection_cache.add("kb-337970215e7c494a")  # hash of "test-scope"

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1536}]}
    with patch("httpx.post", return_value=resp):
        backend.search_scope("query", "test-scope", limit=5, min_score=0.0)

    call_kwargs = mock_client.query_points.call_args[1]
    assert "score_threshold" not in call_kwargs


# ── Qdrant health check tests ───────────────────────────────────────────────


def test_qdrant_health_check_ok():
    backend, mock_client = _make_backend()
    collections = MagicMock()
    collections.collections = [MagicMock(), MagicMock()]
    mock_client.get_collections.return_value = collections

    result = backend.health_check()
    assert result["status"] == "ok"
    assert result["collections"] == "2"


def test_qdrant_health_check_unavailable():
    backend, mock_client = _make_backend()
    mock_client.get_collections.side_effect = ConnectionError("refused")

    result = backend.health_check()
    assert result["status"] == "unavailable"
    assert "refused" in result["message"]


# ── In-memory document store source_type tests ───────────────────────────────


def test_in_memory_store_preserves_source_type():
    from models.knowledge import DocumentRecord, SourceType
    from services.document_store import InMemoryDocumentStore

    store = InMemoryDocumentStore()
    record = DocumentRecord(
        scope="t/s", source="main.py", document_hash="abc",
        source_type=SourceType.CODE_FILE, chunk_count=5, ingested_at=time.time(),
    )
    created = store.create(record)
    assert created.source_type == SourceType.CODE_FILE

    active = store.list_active("t/s")
    assert len(active) == 1
    assert active[0].source_type == SourceType.CODE_FILE


# ── KnowledgeService integration ─────────────────────────────────────────────


def test_knowledge_service_search_accepts_min_score():
    from services.knowledge import KnowledgeService
    svc = KnowledgeService()
    results = svc.search("hello", scopes=["test"], limit=5, min_score=0.5)
    assert isinstance(results, list)


# ── Qdrant collection name sanitization tests (#327) ─────────────────────────
# Physical collection names use SHA-256 hash: kb-{sha256_hex[:16]}
# This is collision-safe — the old '/' → '--' mapping was NOT injective
# (e.g. 'acme/foo--bar' and 'acme--foo/bar' both mapped to 'acme--foo--bar').


def test_qdrant_collection_name_is_hashed():
    """All scope names produce a kb-{16-hex} physical collection name."""
    from services.knowledge import QdrantBackend
    cname = QdrantBackend._qdrant_collection_name("agentopia-architecture")
    assert cname == "kb-7ea3df51b2d83dfd"
    assert cname.startswith("kb-")
    assert len(cname) == 19  # "kb-" + 16 hex chars


def test_qdrant_collection_name_canonical_with_slash():
    """Canonical scope with '/' produces a hashed name, not a simple replacement."""
    from services.knowledge import QdrantBackend
    cname = QdrantBackend._qdrant_collection_name("acme-corp/api-docs")
    assert cname == "kb-316cdd68ce71c530"
    assert "/" not in cname


def test_qdrant_collection_name_distinct_for_different_clients():
    """Same short scope name under different client_ids maps to distinct collection names."""
    from services.knowledge import QdrantBackend
    cname_a = QdrantBackend._qdrant_collection_name("client-a/docs")
    cname_b = QdrantBackend._qdrant_collection_name("client-b/docs")
    assert cname_a != cname_b
    assert cname_a == "kb-f7da28654213d3d7"
    assert cname_b == "kb-587e71333ef03fe8"


def test_qdrant_collection_name_no_slash_in_output():
    """Physical collection name never contains '/' — safe for Qdrant API."""
    from services.knowledge import QdrantBackend
    for canonical in ["a/b", "client-x/scope-y", "org/team/scope"]:
        cname = QdrantBackend._qdrant_collection_name(canonical)
        assert "/" not in cname, f"Collection name for '{canonical}' must not contain '/': got '{cname}'"
        assert cname.startswith("kb-"), f"Collection name must start with 'kb-': got '{cname}'"


def test_qdrant_collection_name_deterministic():
    """Same canonical scope always produces the same physical name."""
    from services.knowledge import QdrantBackend
    a1 = QdrantBackend._qdrant_collection_name("acme-corp/api-docs")
    a2 = QdrantBackend._qdrant_collection_name("acme-corp/api-docs")
    assert a1 == a2


def test_qdrant_collection_name_embedded_double_dash_no_collision():
    """REGRESSION: 'acme/foo--bar' and 'acme--foo/bar' must NOT collide.

    The old '/' → '--' mapping collapsed both to 'acme--foo--bar'.
    SHA-256 hashing produces distinct physical names.
    """
    from services.knowledge import QdrantBackend
    cname_a = QdrantBackend._qdrant_collection_name("acme/foo--bar")
    cname_b = QdrantBackend._qdrant_collection_name("acme--foo/bar")
    assert cname_a != cname_b, (
        f"COLLISION: 'acme/foo--bar' and 'acme--foo/bar' must map to "
        f"different physical names, both got: {cname_a}"
    )
    assert cname_a == "kb-0ddc49bc06471ace"
    assert cname_b == "kb-9f95e162853258b4"


def test_qdrant_ingest_uses_hashed_collection_name():
    """ingest_chunks passes hashed collection name to Qdrant upsert."""
    backend, mock_client = _make_backend()
    mock_client.get_collection.side_effect = Exception("not found")
    mock_client.upsert.return_value = None

    from models.knowledge import DocumentChunk, DocumentMetadata
    chunk = DocumentChunk(
        text="test content",
        metadata=DocumentMetadata(source="doc.txt", scope="acme-corp/api-docs", chunk_index=0),
    )

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1536}]}
    with patch("httpx.post", return_value=resp):
        backend.ingest_chunks("acme-corp/api-docs", [chunk])

    upsert_call = mock_client.upsert.call_args
    collection_used = upsert_call[1].get("collection_name") or upsert_call[0][0]
    assert "/" not in collection_used, (
        f"Qdrant upsert must use hashed name (no '/'), got: {collection_used}"
    )
    assert collection_used == "kb-316cdd68ce71c530"


def test_qdrant_search_uses_hashed_collection_name():
    """_search_dense_only uses hashed collection name in query_points call."""
    from services.knowledge import QdrantBackend
    expected_cname = QdrantBackend._qdrant_collection_name("client-a/docs")

    backend, mock_client = _make_backend()
    backend._collection_cache.add(expected_cname)  # pre-populate so _ensure_collection is skipped

    mock_client.query_points.return_value = MagicMock(points=[])

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1536}]}
    with patch("httpx.post", return_value=resp):
        backend.search_scope("query", "client-a/docs", limit=5)

    call_kwargs = mock_client.query_points.call_args[1]
    cname = call_kwargs.get("collection_name")
    assert cname == expected_cname, (
        f"search_scope must pass hashed collection name to query_points, got: {cname}"
    )


# ── Dimension mismatch guard tests (#331) ──────────────────────────────────


def test_dimension_mismatch_logs_warning(caplog):
    """_validate_collection_dimensions logs WARNING when collection dim != config dim."""
    import logging

    backend, mock_client = _make_backend()
    # Simulate a collection with wrong dimension
    col_mock = MagicMock()
    col_mock.name = "kb-abc123"
    mock_client.get_collections.return_value = MagicMock(collections=[col_mock])
    # Collection has dim=4096 but config expects 1536
    vectors_mock = MagicMock()
    vectors_mock.size = 4096
    config_mock = MagicMock()
    config_mock.params.vectors = vectors_mock
    mock_client.get_collection.return_value = MagicMock(config=config_mock)

    with caplog.at_level(logging.WARNING):
        backend._validate_collection_dimensions()

    assert any("DIMENSION_MISMATCH" in r.message for r in caplog.records), (
        "Expected DIMENSION_MISMATCH warning when collection dim (4096) != config dim (1536)"
    )


def test_dimension_match_no_warning(caplog):
    """No warning when collection dimension matches config."""
    import logging

    backend, mock_client = _make_backend()
    col_mock = MagicMock()
    col_mock.name = "kb-abc123"
    mock_client.get_collections.return_value = MagicMock(collections=[col_mock])
    # Collection dim matches config (1536)
    vectors_mock = MagicMock()
    vectors_mock.size = 1536
    config_mock = MagicMock()
    config_mock.params.vectors = vectors_mock
    mock_client.get_collection.return_value = MagicMock(config=config_mock)

    with caplog.at_level(logging.WARNING):
        backend._validate_collection_dimensions()

    assert not any("DIMENSION_MISMATCH" in r.message for r in caplog.records), (
        "Should NOT warn when collection dim matches config dim"
    )
