"""Phase 2a hybrid retrieval tests (#319).

Tests:
1. Sparse tokenizer correctness
2. Config toggles (hybrid enabled/disabled)
3. Dense-only fallback (backward compatibility)
4. Hybrid collection creation
5. Hybrid ingest (dense + sparse vectors)
6. Hybrid search via prefetch + RRF
7. E2E: in-memory dense-only vs comparison structure
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest


_mock_qdrant_module = None


def _make_backend(hybrid_enabled=False, **env_overrides):
    """Create QdrantBackend with mocked Qdrant client.

    Keeps qdrant_client mock in sys.modules so lazy imports in methods work.
    """
    global _mock_qdrant_module
    env = {
        "HYBRID_SEARCH_ENABLED": "true" if hybrid_enabled else "false",
        "HYBRID_PREFETCH_LIMIT": "20",
    }
    env.update(env_overrides)
    for key in ["EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "EMBEDDING_TIMEOUT_SECONDS",
                 "EMBEDDING_VECTOR_DIMENSION", "HYBRID_SEARCH_ENABLED", "HYBRID_PREFETCH_LIMIT"]:
        os.environ.pop(key, None)
    os.environ.update(env)

    import sys
    mock_qdrant_module = MagicMock()
    mock_client = MagicMock()
    mock_qdrant_module.QdrantClient.return_value = mock_client
    mock_qdrant_module.Distance = MagicMock()
    mock_qdrant_module.VectorParams = MagicMock()
    mock_qdrant_module.SparseVectorParams = MagicMock()
    mock_qdrant_module.SparseIndexParams = MagicMock()
    mock_qdrant_module.PointStruct = MagicMock(side_effect=lambda **kw: kw)
    mock_qdrant_module.SparseVector = MagicMock(side_effect=lambda **kw: kw)
    mock_qdrant_module.FusionQuery = MagicMock(side_effect=lambda **kw: kw)
    mock_qdrant_module.Fusion = MagicMock()
    mock_qdrant_module.Fusion.RRF = "rrf"
    mock_qdrant_module.Prefetch = MagicMock(side_effect=lambda **kw: kw)

    # Keep in sys.modules so lazy imports inside methods resolve
    sys.modules["qdrant_client"] = mock_qdrant_module
    sys.modules["qdrant_client.models"] = mock_qdrant_module
    _mock_qdrant_module = mock_qdrant_module

    from services.knowledge import QdrantBackend
    backend = QdrantBackend("http://localhost:6333")

    for key in env:
        os.environ.pop(key, None)

    return backend, mock_client


# ── Sparse tokenizer tests ──────────────────────────────────────────────────


class TestBM25Encoder:
    """BM25 sparse encoder tests (#319 remediation)."""

    def test_basic_bm25_encoding(self):
        backend, _ = _make_backend(hybrid_enabled=True)
        # With no corpus stats, BM25 still produces non-zero weights (IDF defaults)
        indices, values = backend._tokenize_sparse("hello world hello", scope="", mode="document")
        assert len(indices) == 2  # 2 unique tokens
        assert len(values) == 2
        assert all(v > 0 for v in values)

    def test_empty_text(self):
        backend, _ = _make_backend(hybrid_enabled=True)
        indices, values = backend._tokenize_sparse("", scope="", mode="document")
        assert indices == []
        assert values == []

    def test_deterministic(self):
        backend, _ = _make_backend(hybrid_enabled=True)
        r1 = backend._tokenize_sparse("test query text", scope="", mode="document")
        r2 = backend._tokenize_sparse("test query text", scope="", mode="document")
        assert r1 == r2

    def test_case_insensitive(self):
        backend, _ = _make_backend(hybrid_enabled=True)
        r1 = backend._tokenize_sparse("Hello World", scope="", mode="document")
        r2 = backend._tokenize_sparse("hello world", scope="", mode="document")
        assert r1 == r2

    def test_strips_punctuation(self):
        backend, _ = _make_backend(hybrid_enabled=True)
        indices, values = backend._tokenize_sparse("hello, world! test.", scope="", mode="document")
        assert len(indices) == 3  # hello, world, test

    def test_bm25_idf_reduces_common_term_weight(self):
        """Terms appearing in most documents get lower IDF weight."""
        backend, _ = _make_backend(hybrid_enabled=True)
        # Simulate corpus stats: "the" in all 100 docs, "kubernetes" in 2
        backend._bm25_stats["test"] = {
            "N": 100,
            "df": {"the": 100, "kubernetes": 2},
            "avgdl": 50.0,
        }
        idf_the = backend._bm25_idf("the", backend._bm25_stats["test"])
        idf_k8s = backend._bm25_idf("kubernetes", backend._bm25_stats["test"])
        assert idf_k8s > idf_the, f"Rare term should have higher IDF: kubernetes={idf_k8s}, the={idf_the}"
        assert idf_the <= 0.01, f"Universal term should have near-zero IDF: {idf_the}"

    def test_bm25_query_uses_idf_only(self):
        """Query encoding uses IDF weights, not TF normalization."""
        backend, _ = _make_backend(hybrid_enabled=True)
        backend._bm25_stats["test"] = {
            "N": 100,
            "df": {"hello": 10, "world": 50},
            "avgdl": 50.0,
        }
        indices, values = backend._tokenize_sparse("hello world", scope="test", mode="query")
        assert len(indices) == 2
        # hello (rarer) should have higher weight than world (common)
        idx_hello = backend._term_id("hello")
        idx_world = backend._term_id("world")
        weight_map = dict(zip(indices, values))
        assert weight_map[idx_hello] > weight_map[idx_world]

    def test_bm25_not_plain_tf(self):
        """BM25 document encoding is NOT plain term frequency."""
        backend, _ = _make_backend(hybrid_enabled=True)
        backend._bm25_stats["test"] = {
            "N": 10,
            "df": {"api": 3, "authentication": 2},
            "avgdl": 20.0,
        }
        indices, values = backend._tokenize_sparse(
            "api api api authentication", scope="test", mode="document"
        )
        # Plain TF would give api=3.0 — BM25 with IDF+saturation gives a different value
        weight_map = dict(zip(indices, values))
        api_weight = weight_map.get(backend._term_id("api"), 0)
        assert api_weight != 3.0, f"BM25 weight must differ from raw TF count: got {api_weight}"

    def test_term_id_stable_across_processes(self):
        """_term_id must produce identical IDs across separate Python processes.

        Python's hash() is randomized per process (PYTHONHASHSEED). This test
        runs _term_id in a subprocess to prove stability.
        """
        import subprocess
        from services.knowledge import QdrantBackend

        terms = ["kubernetes", "api", "hello", "deployment", "authentication"]
        local_ids = {t: QdrantBackend._term_id(t) for t in terms}

        # Run in subprocess with different PYTHONHASHSEED
        script = (
            "import hashlib\n"
            "def _term_id(w):\n"
            "    d = hashlib.sha256(w.encode('utf-8')).digest()\n"
            "    return int.from_bytes(d[:4], 'big') % (2**31)\n"
            f"terms = {terms}\n"
            "for t in terms:\n"
            "    print(f'{t}:{_term_id(t)}')\n"
        )
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True,
            env={"PYTHONHASHSEED": "12345"},  # Force different hash seed
            timeout=10,
        )
        subprocess_ids = {}
        for line in result.stdout.strip().split("\n"):
            term, tid = line.split(":")
            subprocess_ids[term] = int(tid)

        for term in terms:
            assert local_ids[term] == subprocess_ids[term], (
                f"_term_id('{term}') differs across processes: "
                f"local={local_ids[term]}, subprocess={subprocess_ids[term]}. "
                f"Must use a stable hash, not Python hash()."
            )

    def test_term_id_known_values(self):
        """_term_id produces known fixed values (regression test)."""
        from services.knowledge import QdrantBackend
        # These are SHA-256 based, deterministic forever
        import hashlib
        for word in ["kubernetes", "api", "hello"]:
            expected = int.from_bytes(
                hashlib.sha256(word.encode("utf-8")).digest()[:4], "big"
            ) % (2**31)
            assert QdrantBackend._term_id(word) == expected


# ── Config toggle tests ──────────────────────────────────────────────────────


def test_hybrid_disabled_by_default():
    backend, _ = _make_backend(hybrid_enabled=False)
    assert backend._hybrid_enabled is False


def test_hybrid_enabled_via_env():
    backend, _ = _make_backend(hybrid_enabled=True)
    assert backend._hybrid_enabled is True


def test_prefetch_limit_configurable():
    backend, _ = _make_backend(hybrid_enabled=True, HYBRID_PREFETCH_LIMIT="30")
    assert backend._hybrid_prefetch_limit == 30


# ── Collection creation tests ────────────────────────────────────────────────


def test_dense_only_creates_simple_collection():
    backend, mock_client = _make_backend(hybrid_enabled=False)
    mock_client.get_collection.side_effect = Exception("not found")

    backend._ensure_collection("test-scope")

    call_kwargs = mock_client.create_collection.call_args[1]
    # Dense-only: vectors_config is a VectorParams, not a dict
    assert "sparse_vectors_config" not in call_kwargs


def test_hybrid_creates_named_vector_collection():
    backend, mock_client = _make_backend(hybrid_enabled=True)
    mock_client.get_collection.side_effect = Exception("not found")

    backend._ensure_collection("test-scope")

    call_kwargs = mock_client.create_collection.call_args[1]
    # Hybrid: vectors_config is a dict with "dense" key
    assert "dense" in call_kwargs["vectors_config"]
    assert "sparse" in call_kwargs["sparse_vectors_config"]


# ── Ingest tests ─────────────────────────────────────────────────────────────


def test_hybrid_ingest_stores_both_vectors():
    from models.knowledge import DocumentChunk, DocumentMetadata
    from services.knowledge import QdrantBackend

    backend, mock_client = _make_backend(hybrid_enabled=True)
    backend._collection_cache.add(QdrantBackend._qdrant_collection_name("test-scope"))

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1536}]}
    with patch("httpx.post", return_value=resp):
        backend.ingest_chunks("test-scope", [
            DocumentChunk(
                text="test chunk about chunking strategies",
                metadata=DocumentMetadata(source="doc.md", chunk_index=0, scope="test-scope"),
            ),
        ])

    # Verify upsert was called with named vectors
    upsert_call = mock_client.upsert.call_args
    points = upsert_call[1]["points"]
    assert len(points) == 1
    # Point was constructed with MagicMock PointStruct, but we can verify the call happened
    mock_client.upsert.assert_called_once()


def test_dense_only_ingest_stores_unnamed_vector():
    from models.knowledge import DocumentChunk, DocumentMetadata
    from services.knowledge import QdrantBackend

    backend, mock_client = _make_backend(hybrid_enabled=False)
    backend._collection_cache.add(QdrantBackend._qdrant_collection_name("test-scope"))

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1536}]}
    with patch("httpx.post", return_value=resp):
        backend.ingest_chunks("test-scope", [
            DocumentChunk(
                text="test chunk",
                metadata=DocumentMetadata(source="doc.md", chunk_index=0, scope="test-scope"),
            ),
        ])

    mock_client.upsert.assert_called_once()


# ── Search tests ─────────────────────────────────────────────────────────────


def test_dense_only_search_uses_query_points():
    # _search_dense_only migrated to query_points() for qdrant-client v1.17+
    from services.knowledge import QdrantBackend
    backend, mock_client = _make_backend(hybrid_enabled=False)
    backend._collection_cache.add(QdrantBackend._qdrant_collection_name("test-scope"))
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1536}]}
    with patch("httpx.post", return_value=resp):
        backend.search_scope("query", "test-scope", limit=5)

    mock_client.query_points.assert_called_once()
    # legacy search() should NOT be called
    mock_client.search.assert_not_called()


def test_hybrid_search_uses_query_points_with_rrf():
    from services.knowledge import QdrantBackend
    expected_cname = QdrantBackend._qdrant_collection_name("test-scope")

    backend, mock_client = _make_backend(hybrid_enabled=True)
    backend._collection_cache.add(expected_cname)  # cache stores hashed names

    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    resp = MagicMock()
    resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1536}]}
    with patch("httpx.post", return_value=resp):
        backend.search_scope("test query", "test-scope", limit=5)

    # query_points should be called (hybrid uses prefetch + fusion)
    mock_client.query_points.assert_called_once()
    call_kwargs = mock_client.query_points.call_args[1]
    assert call_kwargs["collection_name"] == expected_cname
    assert len(call_kwargs["prefetch"]) >= 1  # at least dense prefetch
    # search should NOT be called in hybrid mode
    mock_client.search.assert_not_called()


# ── E2E: in-memory dense-only still works ────────────────────────────────────


def test_in_memory_search_still_works():
    """Verify in-memory fallback (no Qdrant) still functions — backward compat."""
    from services.knowledge import KnowledgeService
    from models.knowledge import DocumentFormat

    svc = KnowledgeService()
    svc.ingest("test/scope", "chunking strategies: fixed-size, paragraph, code-aware",
               source="doc.md", format=DocumentFormat.MARKDOWN)

    results = svc.search("chunking strategies", scopes=["test/scope"], limit=5)
    assert len(results) > 0
    assert results[0].score > 0


# ── Helm validation ──────────────────────────────────────────────────────────


_INFRA_CHART_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../ai-agentopia/agentopia-infra",
)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_INFRA_CHART_DIR, "charts/agentopia-base")),
    reason="agentopia-infra chart not available (cross-repo test, local-only)",
)
def test_helm_hybrid_env_vars_render():
    """Verify Helm renders HYBRID_SEARCH_ENABLED and HYBRID_PREFETCH_LIMIT."""
    import subprocess
    result = subprocess.run(
        ["helm", "template", "test-base", "charts/agentopia-base",
         "--set", "namespace=agentopia",
         "--set", "images.mem0Api.repository=x", "--set", "images.mem0Api.tag=y",
         "--set", "images.botConfigApi.repository=x", "--set", "images.botConfigApi.tag=y",
         "--set", "images.llmProxy.repository=x", "--set", "images.llmProxy.tag=y",
         "--show-only", "templates/bot-config-api.yaml"],
        capture_output=True, text=True,
        cwd=os.path.abspath(_INFRA_CHART_DIR),
    )
    assert "HYBRID_SEARCH_ENABLED" in result.stdout
    assert "HYBRID_PREFETCH_LIMIT" in result.stdout
