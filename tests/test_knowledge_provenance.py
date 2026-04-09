"""#304 — Provenance + Citation Infrastructure tests (ADR-011).

Covers:
  1. ingested_at (float, Unix timestamp) stored and returned correctly
  2. document_hash (SHA-256 str) stored and returned correctly
  3. composite point ID differs: same text / different source
  4. composite point ID differs: same text / different scope
  5. delete-by-source removes only targeted source's chunks
  6. delete-collection removes the scope collection
  7. search/citation path includes provenance fields
  8. per-result audit log emits required fields
"""

import hashlib
import logging
import time
from unittest.mock import MagicMock

from models.knowledge import Citation, SearchResult
from services.knowledge import (
    KnowledgeService,
    compute_document_hash,
    compute_point_id,
)


# ── 1. ingested_at stored and returned ──────────────────────────────────────


class TestIngestedAtField:
    def setup_method(self):
        self.svc = KnowledgeService()

    def test_ingested_at_is_float_unix_timestamp(self):
        """IngestResult carries ingested_at as float Unix timestamp."""
        before = time.time()
        result = self.svc.ingest("scope", "Test content.", "test.txt")
        after = time.time()
        assert isinstance(result.ingested_at, float)
        assert before <= result.ingested_at <= after

    def test_ingested_at_nonzero(self):
        """ingested_at is set during ingestion, not defaulted to 0."""
        result = self.svc.ingest("scope", "Content.", "doc.txt")
        assert result.ingested_at > 0.0

    def test_ingested_at_stored_in_chunk_metadata(self):
        """Every chunk in memory carries the ingested_at timestamp."""
        result = self.svc.ingest("scope", "Some document text.", "doc.txt")
        for chunk in self.svc._chunks["scope"]:
            assert chunk.metadata.ingested_at == result.ingested_at

    def test_ingested_at_consistent_across_chunks(self):
        """All chunks from one ingest call share the same ingested_at."""
        content = "Word " * 500  # produces multiple chunks
        result = self.svc.ingest("scope", content, "big.txt")
        assert result.chunks_created > 1
        timestamps = {c.metadata.ingested_at for c in self.svc._chunks["scope"]}
        assert len(timestamps) == 1  # all same timestamp


# ── 2. document_hash stored and returned ────────────────────────────────────


class TestDocumentHashField:
    def setup_method(self):
        self.svc = KnowledgeService()

    def test_document_hash_in_ingest_result(self):
        """IngestResult carries SHA-256 document_hash."""
        content = "Test content for hashing."
        result = self.svc.ingest("scope", content, "test.txt")
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert result.document_hash == expected

    def test_document_hash_stored_in_chunk_metadata(self):
        """Every chunk carries the document_hash."""
        content = "Document content here."
        result = self.svc.ingest("scope", content, "doc.txt")
        for chunk in self.svc._chunks["scope"]:
            assert chunk.metadata.document_hash == result.document_hash

    def test_compute_document_hash_is_sha256(self):
        """compute_document_hash returns SHA-256 hex digest."""
        text = "hello world"
        expected = hashlib.sha256(text.encode()).hexdigest()
        assert compute_document_hash(text) == expected

    def test_different_content_different_hash(self):
        """Different documents produce different hashes."""
        h1 = compute_document_hash("Document A")
        h2 = compute_document_hash("Document B")
        assert h1 != h2


# ── 3+4. composite point ID: collision avoidance ───────────────────────────


class TestCompositePointId:
    """ADR-011 §2: composite point ID avoids collisions."""

    def test_same_text_different_source_different_id(self):
        """Identical text in different files → different point IDs."""
        id1 = compute_point_id("scope", "file1.txt", 0)
        id2 = compute_point_id("scope", "file2.txt", 0)
        assert id1 != id2

    def test_same_text_different_scope_different_id(self):
        """Identical text in different scopes → different point IDs."""
        id1 = compute_point_id("scope-a", "file.txt", 0)
        id2 = compute_point_id("scope-b", "file.txt", 0)
        assert id1 != id2

    def test_same_inputs_deterministic(self):
        """Same (scope, source, chunk_index) → same point ID every time."""
        id1 = compute_point_id("scope", "file.txt", 3)
        id2 = compute_point_id("scope", "file.txt", 3)
        assert id1 == id2

    def test_different_chunk_index_different_id(self):
        """Different chunk positions → different point IDs."""
        id1 = compute_point_id("scope", "file.txt", 0)
        id2 = compute_point_id("scope", "file.txt", 1)
        assert id1 != id2

    def test_point_id_is_integer(self):
        """Point ID is an integer (int(SHA256[:16], 16) per contract)."""
        pid = compute_point_id("scope", "file.txt", 0)
        assert isinstance(pid, int)
        assert pid > 0


# ── 5. delete-by-source removes only targeted source ───────────────────────


class TestDeleteBySource:
    def setup_method(self):
        self.svc = KnowledgeService()

    def test_delete_removes_only_target_source(self):
        """delete_document removes target source, keeps others."""
        self.svc.ingest("scope", "Keep this content.", "keep.txt")
        self.svc.ingest("scope", "Delete this content.", "delete.txt")

        removed = self.svc.delete_document("scope", "delete.txt")
        assert removed >= 1

        # keep.txt still searchable
        results = self.svc.search("Keep", ["scope"])
        assert len(results) > 0
        assert all(r.citation.source == "keep.txt" for r in results)

        # delete.txt gone
        results_del = self.svc.search("Delete", ["scope"])
        assert len(results_del) == 0

    def test_delete_calls_qdrant_delete_by_source(self):
        """When Qdrant is present, delete_document calls delete_by_source."""
        mock_qdrant = MagicMock()
        self.svc._qdrant = mock_qdrant

        self.svc.ingest("scope", "Content to delete.", "target.txt")
        self.svc.delete_document("scope", "target.txt")

        mock_qdrant.delete_by_source.assert_called_once_with("scope", "target.txt")

    def test_delete_nonexistent_source_returns_zero(self):
        """Deleting a source that doesn't exist returns 0 removed."""
        self.svc.ingest("scope", "Some content.", "exists.txt")
        removed = self.svc.delete_document("scope", "nonexistent.txt")
        assert removed == 0


# ── 6. delete-collection removes scope ─────────────────────────────────────


class TestDeleteCollection:
    def setup_method(self):
        self.svc = KnowledgeService()

    def test_delete_scope_removes_all_data(self):
        """delete_scope removes scope metadata, chunks, and hash cache."""
        self.svc.ingest("doomed", "Content.", "file.txt")
        assert self.svc.delete_scope("doomed") is True
        assert self.svc.get_scope("doomed") is None
        assert "doomed" not in self.svc._chunks
        assert "doomed" not in self.svc._chunk_hashes

    def test_delete_scope_calls_qdrant_delete_collection(self):
        """When Qdrant is present, delete_scope calls delete_collection."""
        mock_qdrant = MagicMock()
        self.svc._qdrant = mock_qdrant

        self.svc.ingest("doomed", "Content.", "file.txt")
        self.svc.delete_scope("doomed")

        mock_qdrant.delete_collection.assert_called_once_with("doomed")

    def test_delete_scope_does_not_affect_other_scopes(self):
        """Deleting one scope leaves others intact."""
        self.svc.ingest("keep", "Persistent content.", "keep.txt")
        self.svc.ingest("delete", "Temporary content.", "del.txt")

        self.svc.delete_scope("delete")

        results = self.svc.search("Persistent", ["keep"])
        assert len(results) > 0
        assert self.svc.get_scope("delete") is None


# ── 7. search/citation path includes provenance fields ─────────────────────


class TestSearchProvenanceCarrythrough:
    def setup_method(self):
        self.svc = KnowledgeService()

    def test_search_result_citation_has_ingested_at(self):
        """Search results carry ingested_at as float in citation."""
        self.svc.ingest("scope", "Kubernetes deployment guide.", "k8s.md")
        results = self.svc.search("Kubernetes", ["scope"])
        assert len(results) >= 1
        assert isinstance(results[0].citation.ingested_at, float)
        assert results[0].citation.ingested_at > 0.0

    def test_search_result_citation_has_document_hash(self):
        """Search results carry document_hash in citation."""
        content = "Kubernetes deployment guide."
        self.svc.ingest("scope", content, "k8s.md")
        results = self.svc.search("Kubernetes", ["scope"])
        assert len(results) >= 1
        expected_hash = compute_document_hash(content)
        assert results[0].citation.document_hash == expected_hash

    def test_build_citations_carries_provenance(self):
        """build_citations extracts provenance fields from Qdrant payload."""
        from services.knowledge import build_citations

        raw = [
            {
                "score": 0.9,
                "payload": {
                    "text": "chunk text",
                    "metadata": {
                        "source": "doc.md",
                        "section": "Intro",
                        "scope": "s1",
                        "chunk_index": 2,
                        "ingested_at": 1711800000.0,
                        "document_hash": "abc123",
                    },
                },
            }
        ]
        results = build_citations(raw)
        assert results[0].citation.ingested_at == 1711800000.0
        assert results[0].citation.document_hash == "abc123"

    def test_provenance_fields_default_for_legacy_data(self):
        """Qdrant payloads without provenance fields default to zero/empty."""
        from services.knowledge import build_citations

        raw = [
            {
                "score": 0.8,
                "payload": {
                    "text": "old chunk",
                    "metadata": {
                        "source": "old.md",
                        "scope": "legacy",
                        "chunk_index": 0,
                    },
                },
            }
        ]
        results = build_citations(raw)
        assert results[0].citation.ingested_at == 0.0
        assert results[0].citation.document_hash == ""

    def test_model_dump_includes_provenance(self):
        """SearchResult.model_dump() includes provenance fields."""
        sr = SearchResult(
            text="chunk",
            score=0.9,
            scope="scope",
            citation=Citation(
                source="file.md",
                ingested_at=1711800000.0,
                document_hash="deadbeef",
            ),
        )
        data = sr.model_dump()
        assert data["citation"]["ingested_at"] == 1711800000.0
        assert data["citation"]["document_hash"] == "deadbeef"


# ── 8. per-result audit log ────────────────────────────────────────────────


class TestSearchAuditLog:
    """ADR-011 §5: per-result provenance audit logging."""

    def test_search_emits_audit_log_per_result(self, caplog):
        """Each search result produces a knowledge_search_result log line."""
        svc = KnowledgeService()
        svc.ingest("scope", "Kubernetes deployment guide.", "k8s.md")

        with caplog.at_level(logging.INFO, logger="services.knowledge"):
            results = svc.search("Kubernetes", ["scope"])

        assert len(results) >= 1
        audit_logs = [
            r for r in caplog.records if "knowledge_search_result" in r.message
        ]
        assert len(audit_logs) == len(results)

    def test_audit_log_contains_required_fields(self, caplog):
        """Audit log contains source, chunk_index, scope, document_hash, score."""
        svc = KnowledgeService()
        svc.ingest("scope", "Terraform provisioning workflow.", "tf.md")

        with caplog.at_level(logging.INFO, logger="services.knowledge"):
            svc.search("Terraform", ["scope"])

        audit_logs = [
            r for r in caplog.records if "knowledge_search_result" in r.message
        ]
        assert len(audit_logs) >= 1
        msg = audit_logs[0].message
        assert "source=" in msg
        assert "chunk_index=" in msg
        assert "scope=" in msg
        assert "document_hash=" in msg
        assert "score=" in msg

    def test_no_results_no_audit_log(self, caplog):
        """When search returns 0 results, no audit log is emitted."""
        svc = KnowledgeService()
        svc.ingest("scope", "Unrelated content.", "other.txt")

        with caplog.at_level(logging.INFO, logger="services.knowledge"):
            results = svc.search("xyznonexistent", ["scope"])

        assert len(results) == 0
        audit_logs = [
            r for r in caplog.records if "knowledge_search_result" in r.message
        ]
        assert len(audit_logs) == 0
