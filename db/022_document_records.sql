-- #303: DocumentRecord lifecycle persistence (ADR-011, ADR-012)
--
-- Tracks document lifecycle: active → superseded → deleted
-- Source of truth for provenance metadata across restarts.
-- Tombstone records retained for historical provenance.

CREATE TABLE IF NOT EXISTS document_records (
    id              BIGSERIAL PRIMARY KEY,
    scope           TEXT NOT NULL,               -- {client_id}/{scope_name}
    source          TEXT NOT NULL,               -- filename / path
    document_hash   TEXT NOT NULL,               -- SHA-256 of content
    format          TEXT NOT NULL DEFAULT 'text', -- pdf, markdown, html, text, code
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    ingested_at     DOUBLE PRECISION NOT NULL,   -- Unix timestamp
    status          TEXT NOT NULL DEFAULT 'active', -- active | superseded | deleted
    superseded_at   DOUBLE PRECISION,            -- when superseded by newer version
    deleted_at      DOUBLE PRECISION,            -- when explicitly deleted
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Only one active document per (scope, source) at any time (partial unique index)
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_scope_source
    ON document_records (scope, source) WHERE (status = 'active');

-- Fast lookups by scope + status (list active documents per scope)
CREATE INDEX IF NOT EXISTS idx_document_records_scope_status
    ON document_records (scope, status);

-- Fast lookups by scope + source (lifecycle checks during re-upload)
CREATE INDEX IF NOT EXISTS idx_document_records_scope_source
    ON document_records (scope, source);
