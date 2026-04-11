# Async Event Schema

This document defines all async events emitted and consumed by the Agentopia knowledge ingestion pipeline.

**Day-1 note:** On day-1, these events are implemented as synchronous in-process function calls inside the Ingest Orchestrator. The event schemas here define the payload contracts that will be carried over queues when the hardening phase introduces async workers. No schema changes are required at that transition — only the transport layer changes (function call → queue message).

---

## Event Naming Convention

Format: `ingest.document.{stage}` or `evaluation.{event}`

All events carry an `idempotency_key` derived from `{document_id}:{version}`. Consumers must treat duplicate deliveries of the same key as no-ops.

---

## Events

### ingest.document.uploaded

**Producer:** Document Ingest Service  
**Consumer:** Normalizer Service  
**Trigger:** Document file stored in S3, registry entry created, job status = `submitted`

```json
{
  "event":           "ingest.document.uploaded",
  "idempotency_key": "{document_id}:{version}",
  "timestamp":       "ISO8601",
  "document_id":     "uuid",
  "version":         1,
  "scope":           "tenant/domain",
  "s3_prefix":       "documents/{scope}/{document_id}/v{version}/",
  "format":          "pdf",
  "job_id":          "uuid"
}
```

---

### ingest.document.normalized

**Producer:** Normalizer Service  
**Consumer:** Metadata Extractor Service  
**Trigger:** `normalized.json` successfully written to S3, document state = `normalized`

```json
{
  "event":           "ingest.document.normalized",
  "idempotency_key": "{document_id}:{version}",
  "timestamp":       "ISO8601",
  "document_id":     "uuid",
  "version":         1,
  "scope":           "tenant/domain",
  "s3_normalized_key": "documents/{scope}/{document_id}/v{version}/normalized.json",
  "job_id":          "uuid",
  "normalizer_version": "1.0.0"
}
```

---

### ingest.document.extracted

**Producer:** Metadata Extractor Service  
**Consumer:** Knowledge Compiler (if enabled) OR Ingest Orchestrator (default)  
**Trigger:** `extracted.json` successfully written to S3, document state = `extracted`

```json
{
  "event":           "ingest.document.extracted",
  "idempotency_key": "{document_id}:{version}",
  "timestamp":       "ISO8601",
  "document_id":     "uuid",
  "version":         1,
  "scope":           "tenant/domain",
  "s3_extracted_key": "documents/{scope}/{document_id}/v{version}/extracted.json",
  "job_id":          "uuid",
  "extraction_method": "heuristic",
  "extractor_version": "1.0.0",
  "partial":         false
  // true if extraction succeeded but some fields are missing (e.g., no title detected)
}
```

---

### ingest.document.compiled

**Producer:** Knowledge Compiler Service (optional — phase 2+)  
**Consumer:** Ingest Orchestrator  
**Trigger:** Compiled representation written; document state = `compiled`

Only emitted if Knowledge Compiler is enabled. If the Compiler is not enabled, this event is never produced and the Orchestrator consumes `ingest.document.extracted` directly.

```json
{
  "event":           "ingest.document.compiled",
  "idempotency_key": "{document_id}:{version}",
  "timestamp":       "ISO8601",
  "document_id":     "uuid",
  "version":         1,
  "scope":           "tenant/domain",
  "compiled_artifact_ref": "git:{commit_sha} or db:{record_id}",
  "job_id":          "uuid"
}
```

---

### ingest.document.indexing

**Producer:** Ingest Orchestrator  
**Consumer:** Job status updater (internal)  
**Trigger:** Orchestrator has submitted the ingest call to Super RAG `/ingest`; waiting for confirmation

```json
{
  "event":           "ingest.document.indexing",
  "idempotency_key": "{document_id}:{version}",
  "timestamp":       "ISO8601",
  "document_id":     "uuid",
  "version":         1,
  "scope":           "tenant/domain",
  "job_id":          "uuid"
}
```

---

### ingest.document.active

**Producer:** Ingest Orchestrator  
**Consumer:** Evaluation Service (triggers regression check on replacement)  
**Trigger:** Super RAG confirms indexing complete; document state = `active`

```json
{
  "event":           "ingest.document.active",
  "idempotency_key": "{document_id}:{version}",
  "timestamp":       "ISO8601",
  "document_id":     "uuid",
  "version":         1,
  "scope":           "tenant/domain",
  "chunk_count":     42,
  "is_replacement":  true,
  // true if a prior version was superseded as part of this ingest
  "superseded_version": 1,
  // null if is_replacement = false
  "job_id":          "uuid"
}
```

When `is_replacement = true`, the Evaluation Service runs the regression gate against the scope's golden question set.

---

### ingest.document.failed

**Producer:** Any pipeline stage (Normalizer, Extractor, Compiler, Orchestrator)  
**Consumer:** Alerting service + job status updater  
**Trigger:** Retries exhausted at any stage; document state = `failed`

```json
{
  "event":           "ingest.document.failed",
  "idempotency_key": "{document_id}:{version}",
  "timestamp":       "ISO8601",
  "document_id":     "uuid",
  "version":         1,
  "scope":           "tenant/domain",
  "failed_stage":    "normalizer",
  // normalizer | extractor | compiler | orchestrator
  "error_type":      "parsing_failed",
  "error_message":   "PDF is password-protected and cannot be parsed",
  "retry_count":     3,
  "job_id":          "uuid"
}
```

On failure, the prior active version of this document (if any) remains active and queryable. No retrieval impact.

---

### evaluation.regression_blocked

**Producer:** Evaluation Service  
**Consumer:** Operator notification service  
**Trigger:** Regression gate fires: `delta_ndcg_5 < -0.02` on document replacement

```json
{
  "event":           "evaluation.regression_blocked",
  "timestamp":       "ISO8601",
  "scope":           "tenant/domain",
  "document_id":     "uuid",
  "new_version":     2,
  "superseded_version": 1,
  "delta_ndcg_5":    -0.035,
  "baseline_ndcg_5": 0.925,
  "new_ndcg_5":      0.890,
  "verdict":         "blocked",
  "evaluation_result_id": "uuid"
}
```

The new document version is indexed and has `status = active` in Qdrant at this point. The regression gate is a governance notification, not a pre-indexing gate. If the operator does not intervene, the new version remains active. The operator can trigger rollback via `POST /documents/{id}/rollback`.

---

### evaluation.baseline_established

**Producer:** Evaluation Service  
**Consumer:** Quality dashboard  
**Trigger:** A new per-scope baseline is recorded (initial baseline or baseline update)

```json
{
  "event":           "evaluation.baseline_established",
  "timestamp":       "ISO8601",
  "scope":           "tenant/domain",
  "ndcg_5":          0.925,
  "mrr":             0.960,
  "p_5":             0.840,
  "golden_question_count": 20,
  "evaluation_baseline_id": "uuid"
}
```

---

## Idempotency Requirements

| Consumer | Idempotency behavior |
|---|---|
| Normalizer | Check if `s3_normalized_key` already exists in S3; skip re-parsing if present |
| Extractor | Check if `s3_extracted_key` already exists in S3; skip re-extraction if present |
| Orchestrator | Check if `(document_id, version)` already `active` in Super RAG; skip re-ingest if present |
| Evaluation Service | Check `evaluation_results` for existing row with `(scope, document_id, document_version)`; skip if present |

All consumers must handle duplicate delivery without side effects. The `idempotency_key` is the primary deduplication key; S3 artifact existence checks provide a second layer.

---

## Day-1 → Phase-2 Transition

On day-1, all stages run synchronously. Events are emitted as structured log entries with the same payload schema defined above. No queue is required.

In phase-2 (hardening), a message broker (SQS or Kafka) is introduced. Each event becomes a queue message. Consumer workers poll their queue and process events. The payload schema is unchanged. The log entries from day-1 become the queue messages in phase-2 — no contract migration required.

Queue topic names (for phase-2 provisioning reference):

| Event | Topic |
|---|---|
| `ingest.document.uploaded` | `ingest.to_normalize` |
| `ingest.document.normalized` | `ingest.to_extract` |
| `ingest.document.extracted` | `ingest.to_compile` or `ingest.to_index` |
| `ingest.document.compiled` | `ingest.to_index` |
| `ingest.document.active` | `ingest.completed` |
| `ingest.document.failed` | `ingest.failed` |
| `evaluation.regression_blocked` | `evaluation.alerts` |
| `evaluation.baseline_established` | `evaluation.updates` |
