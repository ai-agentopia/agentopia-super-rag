# ADR-001: Agentopia Knowledge Ingestion System Design

**Status:** Accepted  
**Date:** 2026-04-11  
**Repos:** `agentopia-super-rag` (retrieval plane), `agentopia-knowledge-ingest` (ingest plane)

---

## Context

Super RAG exists as a production retrieval plane: it chunks, embeds, indexes documents into Qdrant, enforces per-scope access, and provides evaluation tooling. Its `/ingest` API accepts pre-parsed text with metadata.

The gap is upstream. Agentopia knowledge includes PDFs, DOCX, HTML, markdown, and other file types. None of those formats reach Super RAG in a normalized form today. There is no governed place to store originals, no parsing pipeline, no metadata extraction, and no operator-facing ingest workflow.

This ADR records the agreed design for the full upstream knowledge ingestion system built around Super RAG.

---

## Decision

Build a governed upstream knowledge ingestion system consisting of:

1. **Raw Document Store** — S3, versioned, immutable per version
2. **Document Registry** — PostgreSQL, source of truth for document metadata and lifecycle state
3. **Normalizer Service** — format parsing: PDF/DOCX/HTML/markdown → text + structure hints
4. **Metadata Extractor Service** — structural extraction: title, author, date, section hierarchy
5. **Ingest Orchestrator** — coordination: validates, calls Super RAG `/ingest`, confirms indexing
6. **Super RAG** (existing) — chunking, embedding, Qdrant indexing, scoped retrieval, evaluation
7. **Operator Control Plane** — upload API, job status, document management, scope management

An optional **Knowledge Compiler** (derived curated representation) may be introduced later if operator demand is confirmed. It is not part of this design.

---

## System Layer Diagram

```
Sources (S3, upload API, connectors)
    |
    v
Raw Document Store (S3, versioned immutable)
    |---> Document Registry (PostgreSQL: scope, version, state, provenance)
    |
    v [Async Queue — phase 2; day-1: synchronous]
Normalizer Service
    |
    v [Async Queue — phase 2; day-1: synchronous]
Metadata Extractor Service
    |
    v [Sync boundary: Ingest Orchestrator]
Super RAG Indexing (Qdrant + PostgreSQL lifecycle)
    |
    v
Runtime Retrieval (scope-filtered, only active documents)
    |
    v
Evaluation (per-scope baselines, regression gates)

Operator Control Plane:
    Upload UI -> Job Status -> Document Management -> Scope Management -> Quality Dashboard -> Retrieval Debugger
```

---

## Service Boundaries

### agentopia-knowledge-ingest

| Service | Owns | Does NOT Own |
|---|---|---|
| Document Ingest Service | Upload API, job tracking, document registry, S3 raw store | Parsing, extraction, indexing |
| Normalizer Service | Format parsing (PDF/DOCX/HTML/MD → text + structure hints); S3 normalized artifact | Document registry, metadata extraction |
| Metadata Extractor Service | Structural extraction (title, author, date, hierarchy); S3 extracted artifact | Document registry, indexing |
| Ingest Orchestrator | Coordination: validate → call Super RAG → confirm → mark active | Parsing, metadata, retrieval |
| Operator UI | Upload, job status, document management, scope management, quality dashboard | Backend logic, retrieval implementation |

### agentopia-super-rag

| Service | Owns | Does NOT Own |
|---|---|---|
| Super RAG | Chunking, embedding, Qdrant vectors, document lifecycle (active/superseded/deleted), scoped retrieval, evaluation framework, bot-scope binding cache | Format parsing, metadata extraction, raw document storage, compiled knowledge |

---

## Document State Machine

```
submitted
  |
  v [normalizer picks up]
normalizing --[retries exhausted]--> failed
  |
  v [success]
normalized
  |
  v [extractor picks up]
extracting --[partial ok: proceed]--> extracted (partial metadata)
  |
  v [success]
extracted
  |
  v [orchestrator picks up]
indexing --[retries exhausted]--> failed
  |
  v [Super RAG confirms]
active <----[rollback restores prior version here]----
  |
  v [replacement upload]
superseded
  |
  v [operator explicit delete]
deleted
```

**Rollback rule:** When state transitions to `failed` at any stage after a prior version exists, the prior version remains `active`. No manual intervention required.

**Explicit rollback:** `POST /documents/{id}/rollback?version=N` — restores version N to `active`; current active version transitions to `superseded`.

---

## Ingest Job Model

- `POST /documents/upload` returns `{job_id, document_id}` immediately — non-blocking
- `GET /jobs/{job_id}` returns current state: `submitted | normalizing | normalized | extracting | extracted | indexing | active | failed`
- UI polls job status; state transitions are written to PostgreSQL `ingest_jobs` table as they occur
- No long-running sync HTTP calls; operator never waits for parsing or embedding to complete in-request

---

## Visibility Rule

**A document is visible to retrieval queries if and only if its status = `active` in the Qdrant chunk payload.**

- Documents in any other state (`normalizing`, `indexing`, `failed`, `superseded`, `deleted`) are filtered out at query time
- Qdrant query filter: `scope IN allowed_scopes AND status = "active"`
- This is enforced at Super RAG retrieval layer, not at the operator API layer
- There is no partial visibility or eventual consistency window

---

## Source of Truth Per Layer

| Layer | Source of Truth | Storage | Mutable? |
|---|---|---|---|
| Raw document | Original file | S3 `original.{ext}` | Immutable per version |
| Document metadata + state | Document registry | PostgreSQL `documents` | Mutable (state evolves) |
| Normalized text | Parsed artifact | S3 `normalized.json` | Immutable per version |
| Extracted metadata | Structural artifact | S3 `extracted.json` | Immutable per version |
| Compiled knowledge (optional) | Derived representation | Git or PostgreSQL | Mutable, never replaces raw |
| Chunk vectors | Embeddings per chunk | Qdrant (per-scope collection) | Immutable per version |
| Document visibility state | active/superseded/deleted | Super RAG PostgreSQL | Mutable (controls retrieval) |
| Scope-to-bot binding | Bot authorization | K8s CRD annotations | Changed via K8s only |
| Per-scope baselines | Evaluation anchor | PostgreSQL `evaluation_baselines` | Human-curated, rarely changed |
| Evaluation history | Quality audit trail | PostgreSQL `evaluation_results` | Append-only |

**Critical rule:** S3 is source of truth for raw and intermediate artifacts. All S3 artifacts are immutable once written (write-once per version prefix). Compiled knowledge (if introduced) is derived and never the source for retrieval indexing.

---

## Sync vs Async Boundaries

### Day-1 (synchronous pipeline)

All stages run synchronously within the Ingest Orchestrator process. Same external API contract (non-blocking upload → poll job status). Same state machine. No queues.

Suitable for: < ~20 documents/day per scope.

### Target end-state (async pipeline)

| Boundary | Mechanism | Reason |
|---|---|---|
| Ingest Service → Normalizer | Async queue (`ingest.document.uploaded`) | Parsing can take minutes for large PDFs |
| Normalizer → Extractor | Async queue (`ingest.document.normalized`) | Extraction can take seconds to minutes |
| Extractor → Orchestrator | Async queue (`ingest.document.extracted`) | Decouples for independent scaling |
| Orchestrator → Super RAG | **Synchronous** | Must confirm indexing before marking `active` |

The Orchestrator → Super RAG boundary is always synchronous. This ensures the visibility rule is atomic: a document becomes queryable only after Super RAG confirms indexing is complete.

Transition from day-1 to target: replace synchronous function calls with queue workers. No external API changes. No state machine changes. No operator retraining.

---

## Tenant / Scope Isolation

**Scope format:** `{tenant}/{domain}` — e.g., `joblogic-kb/api-docs`, `joblogic-qa/faq`

**Isolation enforcement chain:**

1. **Ingest time:** Scope is required at upload. Validated against scope registry. Documents stored under `{bucket}/documents/{scope}/{doc_id}/` in S3.
2. **Indexing time:** Scope written into every Qdrant chunk payload. Documents in different scopes never share Qdrant collections.
3. **Retrieval time:** Qdrant query filter enforces `scope IN resolved_scopes`. Bot token → K8s CRD binding cache → resolved scope list. Operator cannot change query scope.
4. **Failure mode:** If K8s API unreachable, binding cache falls back to direct API lookup. If both fail, request is rejected (fail-safe).

Scope cannot be changed on an existing document without re-ingesting to the new scope. This is intentional: scope is a security boundary, not a mutable tag.

---

## Evaluation Model

**No global nDCG floor applies across all scopes.**

Instead:
- Each scope has an independently established baseline (`evaluation_baselines` table)
- On document replacement, golden questions are run against both versions
- Gate logic:
  - delta >= 0: auto-approve
  - -0.02 ≤ delta < 0: approve with warning
  - delta < -0.02: block; require operator explicit override with recorded note
- Evaluation results are append-only (never deleted)
- Baselines are human-curated per scope; they change rarely and only intentionally

---

## Optional Compiled Knowledge Layer

The compiled knowledge layer (e.g., Obsidian vault, knowledge graph) is **not part of this design.**

If introduced in future:
- It is a derived representation of raw documents + extracted metadata
- It is never the source of truth for retrieval indexing
- It never replaces or overwrites raw S3 artifacts
- Sync direction: raw source → compiled layer (not bidirectional without explicit operator approval)
- It requires its own promotion decision with confirmed operator demand

---

## Failure and Retry Model

| Stage | Retries | On Exhaustion |
|---|---|---|
| Normalizer | 3×, exponential backoff | state = `failed`; prior active version untouched |
| Extractor (heuristic) | 2×, fixed backoff | Proceed with partial metadata |
| Extractor (LLM, phase 2) | 3×, exponential | Fall back to heuristic |
| Orchestrator → Super RAG | 5×, exponential (critical path) | state = `failed`; prior active version untouched |
| Evaluation service | 2×, fixed | Skip gate, log warning; do not block promotion |

Failure at any stage never corrupts the prior active version. The prior version remains queryable until a new version successfully reaches `active`.

---

## Consequences

**Accepted trade-offs:**
- Day-1 synchronous pipeline is slower for large documents but simpler to operate
- Scope cannot be changed on existing documents (re-ingest required) — acceptable given scope is a security boundary
- LLM-assisted extraction is deferred — heuristic extraction is sufficient for day-1 corpus types
- Evaluation gate may be temporarily skipped if evaluation service fails — this is logged and alerted, never silent

**What this enables:**
- Multiple bots querying the same corpus with guaranteed isolation
- Document replacement with automatic rollback on quality regression
- Reproducible retrieval evaluation against per-scope labeled ground truth
- Clear operator workflow: upload → track → manage → debug
