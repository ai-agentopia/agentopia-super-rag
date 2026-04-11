# Super RAG Data Model

This document covers data models owned by `agentopia-super-rag`. For upstream ingest service data models (PostgreSQL registry, S3 artifacts), see `agentopia-knowledge-ingest/docs/data-model.md`.

---

## Qdrant Chunk Payload

Every chunk indexed into Qdrant carries this payload alongside its embedding vector. The payload is the contract between the Ingest Orchestrator (writer) and Super RAG retrieval (reader + filter).

### Schema

```json
{
  "document_id":   "uuid",
  "scope":         "tenant/domain",
  "version":       1,
  "chunk_index":   0,
  "section_path":  ["Authentication", "Token Format"],
  "title":         "API Reference",
  "status":        "active",
  "created_at":    "2026-04-11T00:00:00Z"
}
```

### Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `document_id` | UUID string | Yes | Stable identifier for the document across all versions |
| `scope` | string | Yes | Scope tag in `{tenant}/{domain}` format. Used as retrieval filter. |
| `version` | integer | Yes | Document version number. Chunks from all versions are retained in Qdrant. Only the `active` version is returned by retrieval queries. |
| `chunk_index` | integer | Yes | Zero-based index of this chunk within the document version. Used for ordering and deduplication. |
| `section_path` | string array | No | Ordered list of section headings from document root to this chunk's location. Populated from `extracted.json` hierarchy. Empty array if document has no detected structure. |
| `title` | string | No | Document title from `extracted.json`. Empty string if not extracted. |
| `status` | string | Yes | One of: `active`, `superseded`, `deleted`. Controls retrieval visibility. |
| `created_at` | ISO8601 string | Yes | Timestamp when this chunk was indexed. |

### Retrieval Filter Contract

Every retrieval query applies both filters. Neither is optional.

```
scope IN [resolved_scopes] AND status = "active"
```

- `resolved_scopes` is the list of scopes the requesting bot is authorized to query, resolved from K8s CRD binding cache at request time
- `status = "active"` ensures that superseded and deleted document versions are never returned, regardless of scope authorization

Chunks from superseded versions are retained in Qdrant with `status = "superseded"` for potential rollback. They are invisible to retrieval until explicitly restored.

### Status Transitions in Qdrant

Status is written as a payload field update — vectors are not re-embedded on status change.

| Event | Old status | New status |
|---|---|---|
| New version indexed | (new chunk) | `active` |
| Replacement completes (new version active) | `active` (prior version) | `superseded` |
| Explicit rollback to prior version | `superseded` (prior) | `active` |
| Explicit rollback replaces current active | `active` (current) | `superseded` |
| Operator delete | `active` or `superseded` | `deleted` |

Deleted chunks remain in Qdrant storage but are filtered at query time. Permanent purge is a separate maintenance operation.

### section_path Population

`section_path` is derived from the `hierarchy` field in `extracted.json` (produced by the Metadata Extractor). For a chunk that falls under:

```
# Authentication
  ## Token Format
    ### Bearer Tokens
```

The `section_path` is: `["Authentication", "Token Format", "Bearer Tokens"]`

For documents with no detected heading structure (e.g., flat PDFs), `section_path` is an empty array `[]`.

---

## PostgreSQL Tables (Super RAG)

Super RAG owns these tables. They are separate from the upstream ingest registry tables (which live in `agentopia-knowledge-ingest`).

### document_records

Tracks the lifecycle state of each indexed document version within Super RAG. This is the authoritative record for what is visible to retrieval.

```sql
CREATE TABLE document_records (
  id            SERIAL PRIMARY KEY,
  document_id   UUID         NOT NULL,
  scope         VARCHAR(255) NOT NULL,
  version       INT          NOT NULL DEFAULT 1,
  status        VARCHAR(50)  NOT NULL DEFAULT 'active',
  -- active | superseded | deleted
  chunk_count   INT,
  source_type   VARCHAR(50),
  -- fixed_size | markdown_aware
  created_at    TIMESTAMP    NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMP    NOT NULL DEFAULT NOW(),
  metadata      JSONB        DEFAULT '{}',
  UNIQUE (document_id, version)
);

CREATE INDEX idx_document_records_scope_status ON document_records(scope, status);
CREATE INDEX idx_document_records_document_id  ON document_records(document_id);
```

### evaluation_baselines

Per-scope baseline metrics. Human-curated. Rarely changed. Source of truth for regression gate comparisons.

```sql
CREATE TABLE evaluation_baselines (
  id                    UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
  scope                 VARCHAR(255) NOT NULL UNIQUE,
  ndcg_5                FLOAT,
  mrr                   FLOAT,
  p_5                   FLOAT,
  golden_question_count INT,
  established_at        TIMESTAMP NOT NULL DEFAULT NOW(),
  notes                 TEXT
);
```

### evaluation_results

Append-only log of all evaluation runs. Never deleted.

```sql
CREATE TABLE evaluation_results (
  id                UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
  scope             VARCHAR(255) NOT NULL,
  document_id       UUID,
  document_version  INT,
  run_at            TIMESTAMP NOT NULL DEFAULT NOW(),
  trigger           VARCHAR(50),
  -- replacement | scheduled | manual
  ndcg_5            FLOAT,
  mrr               FLOAT,
  p_5               FLOAT,
  delta_ndcg_5      FLOAT,
  -- delta vs scope baseline; negative = regression
  verdict           VARCHAR(50),
  -- passed | warning | blocked | overridden
  operator_override BOOLEAN   NOT NULL DEFAULT FALSE,
  operator_note     TEXT
);

CREATE INDEX idx_evaluation_results_scope ON evaluation_results(scope, run_at DESC);
```

### golden_questions

Per-scope labeled queries for evaluation. Each row is one question with expected document IDs.

```sql
CREATE TABLE golden_questions (
  id                   UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
  scope                VARCHAR(255) NOT NULL,
  query                TEXT         NOT NULL,
  expected_document_ids UUID[]       NOT NULL,
  weight               FLOAT        NOT NULL DEFAULT 1.0,
  created_by           VARCHAR(255),
  created_at           TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_golden_questions_scope ON golden_questions(scope);
```
