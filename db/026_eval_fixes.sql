-- Migration 026: evaluation result fixes
-- Idempotent: safe to run multiple times

-- Finding 2: add operator_identity so overrides are attributable
ALTER TABLE evaluation_results
    ADD COLUMN IF NOT EXISTS operator_identity VARCHAR(255);
