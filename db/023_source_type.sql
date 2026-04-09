-- Super RAG Phase 0: Add source_type column to document_records
-- Backward-compatible: existing records default to 'business_doc'
-- Foundation for Track C (code repo / feature artifact knowledge)

ALTER TABLE document_records
    ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'business_doc';

-- Index for filtering by source type (useful when code/feature scopes are added)
CREATE INDEX IF NOT EXISTS idx_document_records_source_type
    ON document_records (source_type);
