-- Migration 024: add metadata JSONB column to document_records
-- Used by the orchestrator ingest path to store {"version": N}
-- so prior-version detection works without parsing the source string.
ALTER TABLE document_records
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';
