"""Postgres-backed DocumentRecord store (#303, ADR-011/012).

Durable lifecycle persistence for document records.
Source of truth across restarts — no in-memory fallback for production.

For testing without Postgres, InMemoryDocumentStore provides the same interface.
"""

import logging
import os
import time
from typing import Protocol

from models.knowledge import DocumentFormat, DocumentRecord, DocumentRecordStatus, SourceType

logger = logging.getLogger(__name__)


class DocumentStore(Protocol):
    """Interface for document lifecycle persistence."""

    def get_active(self, scope: str, source: str) -> DocumentRecord | None: ...
    def create(self, record: DocumentRecord) -> DocumentRecord: ...
    def replace_active(self, scope: str, source: str, new_record: DocumentRecord) -> DocumentRecord:
        """Atomically supersede old active + create new active.

        If this fails, the old active record remains unchanged.
        This is the commit point of two-phase replace (ADR-012).
        """
        ...
    def supersede(self, scope: str, source: str) -> None: ...
    def mark_deleted(self, scope: str, source: str) -> None: ...
    def mark_scope_deleted(self, scope: str) -> None: ...
    def list_active(self, scope: str) -> list[DocumentRecord]: ...
    def list_all(self, scope: str) -> list[DocumentRecord]: ...


class InMemoryDocumentStore:
    """In-memory document store for testing (no Postgres required)."""

    def __init__(self) -> None:
        self._records: list[DocumentRecord] = []
        self._next_id = 1

    def get_active(self, scope: str, source: str) -> DocumentRecord | None:
        for r in self._records:
            if r.scope == scope and r.source == source and r.status == DocumentRecordStatus.ACTIVE:
                return r
        return None

    def create(self, record: DocumentRecord) -> DocumentRecord:
        record.id = self._next_id
        self._next_id += 1
        self._records.append(record)
        return record

    def replace_active(self, scope: str, source: str, new_record: DocumentRecord) -> DocumentRecord:
        """Atomic supersede-old + create-new. Old stays active until this succeeds."""
        now = time.time()
        for r in self._records:
            if r.scope == scope and r.source == source and r.status == DocumentRecordStatus.ACTIVE:
                r.status = DocumentRecordStatus.SUPERSEDED
                r.superseded_at = now
        new_record.id = self._next_id
        self._next_id += 1
        self._records.append(new_record)
        return new_record

    def supersede(self, scope: str, source: str) -> None:
        now = time.time()
        for r in self._records:
            if r.scope == scope and r.source == source and r.status == DocumentRecordStatus.ACTIVE:
                r.status = DocumentRecordStatus.SUPERSEDED
                r.superseded_at = now

    def mark_deleted(self, scope: str, source: str) -> None:
        now = time.time()
        for r in self._records:
            if r.scope == scope and r.source == source and r.status == DocumentRecordStatus.ACTIVE:
                r.status = DocumentRecordStatus.DELETED
                r.deleted_at = now

    def mark_scope_deleted(self, scope: str) -> None:
        now = time.time()
        for r in self._records:
            if r.scope == scope and r.status == DocumentRecordStatus.ACTIVE:
                r.status = DocumentRecordStatus.DELETED
                r.deleted_at = now

    def list_active(self, scope: str) -> list[DocumentRecord]:
        return [r for r in self._records if r.scope == scope and r.status == DocumentRecordStatus.ACTIVE]

    def list_all(self, scope: str) -> list[DocumentRecord]:
        return [r for r in self._records if r.scope == scope]


class PostgresDocumentStore:
    """Postgres-backed document store for production (#303).

    Uses synchronous psycopg (same pattern as db_migrate.py).
    Connection created on first use, reused across calls.
    """

    def __init__(self, database_url: str) -> None:
        import psycopg
        self._conn = psycopg.connect(database_url)
        logger.info("PostgresDocumentStore: connected")

    def get_active(self, scope: str, source: str) -> DocumentRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT id, scope, source, document_hash, format, source_type, chunk_count,
                          ingested_at, status, superseded_at, deleted_at
                   FROM document_records
                   WHERE scope = %s AND source = %s AND status = 'active'
                   LIMIT 1""",
                (scope, source),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def create(self, record: DocumentRecord) -> DocumentRecord:
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO document_records
                   (scope, source, document_hash, format, source_type, chunk_count, ingested_at, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    record.scope,
                    record.source,
                    record.document_hash,
                    record.format.value,
                    record.source_type.value,
                    record.chunk_count,
                    record.ingested_at,
                    record.status.value,
                ),
            )
            record.id = cur.fetchone()[0]
        self._conn.commit()
        return record

    def replace_active(self, scope: str, source: str, new_record: DocumentRecord) -> DocumentRecord:
        """Atomic supersede-old + create-new in a single transaction.

        If any step fails, the transaction rolls back and the old active stays.
        This is the commit point of two-phase replace (ADR-012).
        """
        now = time.time()
        try:
            with self._conn.cursor() as cur:
                # Step 1: supersede old active
                cur.execute(
                    """UPDATE document_records
                       SET status = 'superseded', superseded_at = %s
                       WHERE scope = %s AND source = %s AND status = 'active'""",
                    (now, scope, source),
                )
                # Step 2: create new active
                cur.execute(
                    """INSERT INTO document_records
                       (scope, source, document_hash, format, source_type, chunk_count, ingested_at, status)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        new_record.scope,
                        new_record.source,
                        new_record.document_hash,
                        new_record.format.value,
                        new_record.source_type.value,
                        new_record.chunk_count,
                        new_record.ingested_at,
                        new_record.status.value,
                    ),
                )
                new_record.id = cur.fetchone()[0]
            self._conn.commit()
            return new_record
        except Exception:
            self._conn.rollback()
            raise

    def supersede(self, scope: str, source: str) -> None:
        now = time.time()
        with self._conn.cursor() as cur:
            cur.execute(
                """UPDATE document_records
                   SET status = 'superseded', superseded_at = %s
                   WHERE scope = %s AND source = %s AND status = 'active'""",
                (now, scope, source),
            )
        self._conn.commit()

    def mark_deleted(self, scope: str, source: str) -> None:
        now = time.time()
        with self._conn.cursor() as cur:
            cur.execute(
                """UPDATE document_records
                   SET status = 'deleted', deleted_at = %s
                   WHERE scope = %s AND source = %s AND status = 'active'""",
                (now, scope, source),
            )
        self._conn.commit()

    def mark_scope_deleted(self, scope: str) -> None:
        now = time.time()
        with self._conn.cursor() as cur:
            cur.execute(
                """UPDATE document_records
                   SET status = 'deleted', deleted_at = %s
                   WHERE scope = %s AND status = 'active'""",
                (now, scope),
            )
        self._conn.commit()

    def list_active(self, scope: str) -> list[DocumentRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT id, scope, source, document_hash, format, source_type, chunk_count,
                          ingested_at, status, superseded_at, deleted_at
                   FROM document_records
                   WHERE scope = %s AND status = 'active'
                   ORDER BY source""",
                (scope,),
            )
            return [self._row_to_record(row) for row in cur.fetchall()]

    def list_all(self, scope: str) -> list[DocumentRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT id, scope, source, document_hash, format, source_type, chunk_count,
                          ingested_at, status, superseded_at, deleted_at
                   FROM document_records
                   WHERE scope = %s
                   ORDER BY source, created_at""",
                (scope,),
            )
            return [self._row_to_record(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_record(row) -> DocumentRecord:
        return DocumentRecord(
            id=row[0],
            scope=row[1],
            source=row[2],
            document_hash=row[3],
            format=DocumentFormat(row[4]),
            source_type=SourceType(row[5]) if row[5] else SourceType.BUSINESS_DOC,
            chunk_count=row[6],
            ingested_at=row[7],
            status=DocumentRecordStatus(row[8]),
            superseded_at=row[9],
            deleted_at=row[10],
        )


def get_document_store() -> DocumentStore:
    """Return document store for the current environment.

    Production (DATABASE_URL set): PostgresDocumentStore. Raises on failure.
    Dev/test (DATABASE_URL unset): InMemoryDocumentStore — explicit, not a silent fallback.
    """
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        # Production path: Postgres required, fail-closed on init error
        return PostgresDocumentStore(database_url)
    # Dev/test path only — no DATABASE_URL means no Postgres available
    logger.info("DocumentStore: DATABASE_URL not set — using InMemoryDocumentStore (dev/test only)")
    return InMemoryDocumentStore()
