-- Migration 025: evaluation tables for per-scope baselines, golden questions, and results
-- Idempotent: safe to run multiple times

-- Per-scope golden question set
-- Each row is one query with expected document sources for relevance grading.
-- human-authored; operator adds/removes via API or direct SQL.
CREATE TABLE IF NOT EXISTS golden_questions (
    id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    scope      VARCHAR(255) NOT NULL,
    query      TEXT         NOT NULL,
    -- expected_sources: JSON array of {source: str, relevance: int}
    -- relevance: 2 = fully relevant, 1 = partially relevant, 0 = not relevant
    -- "source" matches document_records.source (filename/path key)
    expected_sources JSONB  NOT NULL DEFAULT '[]',
    weight     FLOAT        NOT NULL DEFAULT 1.0,
    created_by VARCHAR(255),
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_golden_questions_scope ON golden_questions(scope);

-- Per-scope evaluation baselines
-- One row per scope; updated when a new baseline is established.
-- Human-curated; never auto-deleted.
CREATE TABLE IF NOT EXISTS evaluation_baselines (
    id                    UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    scope                 VARCHAR(255) NOT NULL UNIQUE,
    ndcg_5                FLOAT,
    mrr                   FLOAT,
    p_5                   FLOAT,
    r_5                   FLOAT,
    golden_question_count INT,
    established_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes                 TEXT
);

-- Evaluation results — append-only audit trail
-- One row per evaluation run. Never deleted.
CREATE TABLE IF NOT EXISTS evaluation_results (
    id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    scope             VARCHAR(255) NOT NULL,
    document_id       VARCHAR(255),  -- logical document_id from upstream registry (optional)
    document_version  INT,
    run_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    trigger           VARCHAR(50)  NOT NULL,
    -- replacement | manual | scheduled
    ndcg_5            FLOAT,
    mrr               FLOAT,
    p_5               FLOAT,
    r_5               FLOAT,
    delta_ndcg_5      FLOAT,        -- delta vs scope baseline (negative = regression)
    verdict           VARCHAR(50)  NOT NULL,
    -- passed | warning | blocked | overridden | no_baseline | no_questions
    operator_override BOOLEAN      NOT NULL DEFAULT FALSE,
    operator_note     TEXT
);

CREATE INDEX IF NOT EXISTS idx_evaluation_results_scope ON evaluation_results(scope, run_at DESC);
