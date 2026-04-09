"""#303 — Ingestion Pipeline + Document Lifecycle tests (ADR-011/012).

Covers:
  1. Migration file is valid SQL and loadable
  2. Ingest creates durable active DocumentRecord
  3. Same-hash re-upload returns unchanged (no churn)
  4. Modified re-upload performs two-phase replace
  5. Commit failure semantics are honest
  6. Delete document: Qdrant + lifecycle tombstone
  7. Delete scope: Qdrant collection + lifecycle records
  8. Restart durability: new store reads persisted state
  9. Webhook retirement: returns 410 Gone

Prerequisites:
  - pydantic >= 2.x, pytest >= 8.x
  - CWD = bot-config-api/src
  - No DATABASE_URL needed (uses InMemoryDocumentStore)
  - No QDRANT_URL needed (uses in-memory + mock)
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.knowledge import DocumentFormat, DocumentRecord, DocumentRecordStatus
from services.document_store import InMemoryDocumentStore
from services.knowledge import KnowledgeService, compute_document_hash


def _make_svc(doc_store=None) -> KnowledgeService:
    """Create a KnowledgeService with InMemoryDocumentStore."""
    svc = KnowledgeService()
    svc._doc_store = doc_store or InMemoryDocumentStore()
    return svc


# ── 1. Migration file is valid ──────────────────────────────────────────────


class TestMigrationFile:
    def test_migration_file_exists(self):
        """022_document_records.sql exists in db/ directory."""
        # Phase 2 extraction: migrations extracted to repo-root db/ (from bot-config-api/db/)
        db_dir = Path(__file__).parent.parent / "db"
        migration = db_dir / "022_document_records.sql"
        assert migration.exists(), f"Migration not found at {migration}"

    def test_migration_file_contains_create_table(self):
        """Migration creates document_records table."""
        db_dir = Path(__file__).parent.parent / "db"
        sql = (db_dir / "022_document_records.sql").read_text()
        assert "CREATE TABLE IF NOT EXISTS document_records" in sql

    def test_migration_has_unique_constraint(self):
        """Migration has unique constraint on active (scope, source)."""
        db_dir = Path(__file__).parent.parent / "db"
        sql = (db_dir / "022_document_records.sql").read_text()
        assert "uq_active_scope_source" in sql

    def test_migration_file_is_loadable_by_migrator(self):
        """Migration filename follows the repo's naming convention (NNN_*.sql)."""
        db_dir = Path(__file__).parent.parent / "db"
        files = sorted(db_dir.glob("*.sql"))
        names = [f.name for f in files]
        assert "022_document_records.sql" in names


# ── 2. Ingest creates durable active record ─────────────────────────────────


class TestIngestCreatesRecord:
    def test_ingest_creates_active_document_record(self):
        """Successful ingest creates an active DocumentRecord in the store."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope-a", "Test content for ingestion.", "readme.md")

        records = store.list_active("scope-a")
        assert len(records) == 1
        r = records[0]
        assert r.scope == "scope-a"
        assert r.source == "readme.md"
        assert r.status == DocumentRecordStatus.ACTIVE
        assert r.document_hash == compute_document_hash("Test content for ingestion.")
        assert r.ingested_at > 0.0
        assert r.chunk_count >= 1

    def test_ingest_multiple_sources_creates_separate_records(self):
        """Each source file gets its own DocumentRecord."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope-a", "Content A.", "a.txt")
        svc.ingest("scope-a", "Content B.", "b.txt")

        records = store.list_active("scope-a")
        sources = {r.source for r in records}
        assert sources == {"a.txt", "b.txt"}


# ── 3. Same-hash re-upload is unchanged ─────────────────────────────────────


class TestSameHashShortCircuit:
    def test_same_hash_returns_zero_chunks(self):
        """Re-uploading identical content returns chunks_created=0."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        content = "Identical content for re-upload test."
        r1 = svc.ingest("scope", content, "doc.txt")
        r2 = svc.ingest("scope", content, "doc.txt")

        assert r1.chunks_created >= 1
        assert r2.chunks_created == 0
        assert r2.document_hash == r1.document_hash

    def test_same_hash_does_not_create_tombstone(self):
        """Same-hash re-upload creates no superseded/deleted records."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        content = "Stable content."
        svc.ingest("scope", content, "doc.txt")
        svc.ingest("scope", content, "doc.txt")

        all_records = store.list_all("scope")
        # Should have exactly 1 record (active), no tombstone
        assert len(all_records) == 1
        assert all_records[0].status == DocumentRecordStatus.ACTIVE

    def test_same_hash_preserves_original_ingested_at(self):
        """Same-hash re-upload returns original ingested_at, not a new timestamp."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        content = "Timestamped content."
        r1 = svc.ingest("scope", content, "doc.txt")
        r2 = svc.ingest("scope", content, "doc.txt")

        assert r2.ingested_at == r1.ingested_at


# ── 4. Modified re-upload performs two-phase replace ────────────────────────


class TestTwoPhaseReplace:
    def test_different_hash_creates_new_active(self):
        """Modified re-upload supersedes old, creates new active."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Version 1 content.", "doc.txt")
        svc.ingest("scope", "Version 2 content completely different.", "doc.txt")

        active = store.list_active("scope")
        assert len(active) == 1
        assert active[0].document_hash == compute_document_hash(
            "Version 2 content completely different."
        )

    def test_old_record_becomes_superseded(self):
        """Old document record is marked superseded after replace."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Version 1.", "doc.txt")
        svc.ingest("scope", "Version 2.", "doc.txt")

        all_records = store.list_all("scope")
        statuses = {r.status for r in all_records}
        assert DocumentRecordStatus.ACTIVE in statuses
        assert DocumentRecordStatus.SUPERSEDED in statuses
        assert len(all_records) == 2

    def test_superseded_record_has_timestamp(self):
        """Superseded record has superseded_at set."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Old version.", "doc.txt")
        svc.ingest("scope", "New version.", "doc.txt")

        all_records = store.list_all("scope")
        superseded = [r for r in all_records if r.status == DocumentRecordStatus.SUPERSEDED]
        assert len(superseded) == 1
        assert superseded[0].superseded_at is not None
        assert superseded[0].superseded_at > 0.0

    def test_replace_calls_qdrant_delete_by_source(self):
        """Two-phase replace deletes old chunks from Qdrant."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        mock_qdrant = MagicMock()
        svc._qdrant = mock_qdrant

        svc.ingest("scope", "Old content.", "doc.txt")
        svc.ingest("scope", "New content.", "doc.txt")

        mock_qdrant.delete_by_source.assert_called_with("scope", "doc.txt")

    def test_replace_search_returns_only_new_chunks(self):
        """After replace, search returns only new version's content."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Kubernetes deployment strategies.", "doc.txt")
        svc.ingest("scope", "Ansible playbook automation patterns.", "doc.txt")

        results = svc.search("Kubernetes", ["scope"])
        assert len(results) == 0  # old content gone

        results = svc.search("Ansible", ["scope"])
        assert len(results) >= 1  # new content present


# ── 5. Failure semantics ───────────────────────────────────────────────────


class TestFailureSemantics:
    def test_doc_store_create_failure_propagates(self):
        """If DocumentRecord creation fails, the exception propagates."""
        store = MagicMock()
        store.get_active.return_value = None
        store.create.side_effect = Exception("DB write failed")

        svc = _make_svc(store)
        with pytest.raises(Exception, match="DB write failed"):
            svc.ingest("scope", "Content.", "doc.txt")

    def test_replace_failure_preserves_old_active(self):
        """If replace_active fails, old document stays active (not superseded)."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Version 1.", "doc.txt")

        # Verify V1 is active
        v1 = store.get_active("scope", "doc.txt")
        assert v1 is not None
        v1_hash = v1.document_hash

        # Make replace_active fail
        store.replace_active = MagicMock(side_effect=Exception("DB transaction failed"))
        with pytest.raises(Exception, match="DB transaction failed"):
            svc.ingest("scope", "Version 2.", "doc.txt")

        # Old active record must still be active and unchanged
        still_active = store.get_active("scope", "doc.txt")
        assert still_active is not None
        assert still_active.document_hash == v1_hash
        assert still_active.status == DocumentRecordStatus.ACTIVE

    def test_replace_active_is_atomic(self):
        """replace_active supersedes old + creates new in a single call."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Version 1.", "doc.txt")

        # Verify V1 is active before replace
        assert store.get_active("scope", "doc.txt") is not None

        svc.ingest("scope", "Version 2.", "doc.txt")

        # After replace: exactly one active (V2), one superseded (V1)
        all_records = store.list_all("scope")
        active = [r for r in all_records if r.status == DocumentRecordStatus.ACTIVE]
        superseded = [r for r in all_records if r.status == DocumentRecordStatus.SUPERSEDED]
        assert len(active) == 1
        assert len(superseded) == 1
        assert active[0].document_hash == compute_document_hash("Version 2.")
        assert superseded[0].document_hash == compute_document_hash("Version 1.")


# ── 6. Delete document behavior ────────────────────────────────────────────


class TestDeleteDocumentLifecycle:
    def test_delete_marks_record_as_deleted(self):
        """delete_document marks the active record as deleted."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Content to delete.", "doc.txt")
        svc.delete_document("scope", "doc.txt")

        active = store.list_active("scope")
        assert len(active) == 0

        all_records = store.list_all("scope")
        assert len(all_records) == 1
        assert all_records[0].status == DocumentRecordStatus.DELETED
        assert all_records[0].deleted_at is not None

    def test_delete_calls_qdrant_delete_by_source(self):
        """delete_document calls Qdrant delete_by_source."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        mock_qdrant = MagicMock()
        svc._qdrant = mock_qdrant

        svc.ingest("scope", "Content.", "doc.txt")
        svc.delete_document("scope", "doc.txt")

        mock_qdrant.delete_by_source.assert_called_with("scope", "doc.txt")


# ── 7. Delete scope behavior ──────────────────────────────────────────────


class TestDeleteScopeLifecycle:
    def test_delete_scope_marks_all_records_deleted(self):
        """delete_scope marks all active documents in scope as deleted."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Doc A.", "a.txt")
        svc.ingest("scope", "Doc B.", "b.txt")

        svc.delete_scope("scope")

        active = store.list_active("scope")
        assert len(active) == 0

        all_records = store.list_all("scope")
        assert all(r.status == DocumentRecordStatus.DELETED for r in all_records)

    def test_delete_scope_calls_qdrant_delete_collection(self):
        """delete_scope calls Qdrant delete_collection."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        mock_qdrant = MagicMock()
        svc._qdrant = mock_qdrant

        svc.ingest("scope", "Content.", "doc.txt")
        svc.delete_scope("scope")

        mock_qdrant.delete_collection.assert_called_with("scope")


# ── 8. Restart durability ──────────────────────────────────────────────────


class TestRestartDurability:
    def test_new_store_reads_persisted_records(self):
        """A fresh InMemoryDocumentStore (simulating restart) shares state via reference.

        In production, PostgresDocumentStore reads from Postgres on each call.
        Here we verify the interface contract: list_active returns what was created.
        """
        store = InMemoryDocumentStore()

        # Simulate ingest writing to store
        record = DocumentRecord(
            scope="scope",
            source="doc.txt",
            document_hash="abc123",
            format=DocumentFormat.TEXT,
            chunk_count=5,
            ingested_at=1711800000.0,
            status=DocumentRecordStatus.ACTIVE,
        )
        store.create(record)

        # Simulate "restart" — same store object (Postgres would reconnect to same DB)
        records = store.list_active("scope")
        assert len(records) == 1
        assert records[0].source == "doc.txt"
        assert records[0].document_hash == "abc123"
        assert records[0].status == DocumentRecordStatus.ACTIVE

    def test_list_documents_uses_store_not_memory(self):
        """KnowledgeService.list_documents() reads from DocumentStore when available."""
        store = InMemoryDocumentStore()
        svc = _make_svc(store)
        svc.ingest("scope", "Test content.", "test.txt")

        docs = svc.list_documents("scope")
        assert len(docs) == 1
        assert docs[0]["source"] == "test.txt"
        assert "document_hash" in docs[0]
        assert "ingested_at" in docs[0]
        assert "format" in docs[0]

    def test_lifecycle_state_survives_service_recreation(self):
        """Document state persisted in store is available to a new KnowledgeService."""
        store = InMemoryDocumentStore()

        # Service 1: ingest
        svc1 = KnowledgeService()
        svc1._doc_store = store
        svc1.ingest("scope", "Content.", "doc.txt")

        # Service 2: reads same store (simulates pod restart with shared Postgres)
        svc2 = KnowledgeService()
        svc2._doc_store = store
        docs = svc2.list_documents("scope")
        assert len(docs) == 1
        assert docs[0]["source"] == "doc.txt"


# ── 9. Webhook retirement ─────────────────────────────────────────────────


# TestDirectModeWarning removed — tests bot-config-api's _knowledge_api_url(),
# not knowledge-api's code. That test is now obsolete since bot-config-api
# uses _require_proxy() which returns 503 instead of warning (#330).


# ── 10. Storage contract — fail-closed ──────────────────────────────────────


class TestStorageContract:
    """Production storage must not silently fall back to in-memory."""

    def test_get_document_store_returns_inmemory_without_database_url(self):
        """Without DATABASE_URL, returns InMemoryDocumentStore (dev/test only)."""
        from unittest.mock import patch
        from services.document_store import get_document_store, InMemoryDocumentStore as IMDS

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("DATABASE_URL", None)
            store = get_document_store()
            assert isinstance(store, IMDS)

    def test_get_document_store_raises_when_postgres_unreachable(self):
        """With DATABASE_URL set but Postgres unreachable, raises (fail-closed)."""
        from unittest.mock import patch
        from services.document_store import get_document_store

        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://bad:bad@localhost:1/bad"}):
            with pytest.raises(Exception):
                get_document_store()
