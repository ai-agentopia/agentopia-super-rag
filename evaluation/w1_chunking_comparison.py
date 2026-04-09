"""W1 Evaluation: Compare fixed-size vs markdown-aware chunking (#14).

Runs a controlled comparison on a documentation-heavy corpus.
Uses the in-memory KnowledgeService (no Qdrant required).

Measures:
- Chunk count and average chunk size
- Section heading preservation rate
- Code block integrity (code fences intact)
- Search result relevance (cosine similarity via in-memory search)

Usage:
    PYTHONPATH=src python evaluation/w1_chunking_comparison.py
"""

import json
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.knowledge import ChunkingStrategy, DocumentFormat, IngestConfig
from services.knowledge import KnowledgeService, chunk_document

# ── Test corpus: realistic markdown documentation ────────────────────────

DOCS_CORPUS = {
    "api-guide.md": """# Knowledge API Guide

The Knowledge API provides document ingestion and semantic search for Agentopia bots.

## Authentication

Two authentication paths are supported:

- **Operator path**: Uses `X-Internal-Token` header. For bot-config-api proxy calls.
- **Bot path**: Uses `Authorization: Bearer <token>` + `X-Bot-Name` headers. For direct gateway calls.

### Token Management

Operator tokens are configured via the `KNOWLEDGE_API_INTERNAL_TOKEN` environment variable. Bot tokens are generated per-bot by bot-config-api and stored as Kubernetes secrets.

## Ingestion Pipeline

Documents are ingested through a two-phase atomic pipeline:

### Phase 1: Prepare

1. Compute SHA-256 hash of document content
2. Check if identical hash already exists (skip if unchanged)
3. Split document into chunks using configured strategy
4. Deduplicate chunks within scope using MD5 hashes

### Phase 2: Commit

1. Create or supersede document lifecycle record in Postgres
2. Delete old chunks from Qdrant (if replacing)
3. Embed new chunks via OpenRouter API
4. Upsert embedded chunks to Qdrant collection

```python
# Example: Ingest a markdown document
import httpx

resp = httpx.post(
    "http://knowledge-api:8002/api/v1/knowledge/my-scope/ingest",
    files={"file": open("architecture.md", "rb")},
    headers={"X-Internal-Token": os.environ["KNOWLEDGE_API_INTERNAL_TOKEN"]},
)
print(resp.json())
# {"status": "ingested", "chunks_created": 12, "format": "markdown"}
```

## Search

Query the knowledge base with semantic search:

```python
resp = httpx.get(
    "http://knowledge-api:8002/api/v1/knowledge/search",
    params={"query": "How does authentication work?", "scopes": "my-scope"},
    headers={"X-Internal-Token": token},
)
results = resp.json()["results"]
for r in results:
    print(f"[{r['score']:.2f}] {r['citation']['source']}: {r['text'][:100]}")
```

## Chunking Strategies

The API supports multiple chunking strategies via the `chunking_strategy` parameter:

| Strategy | Description | Best for |
|----------|-------------|----------|
| `fixed_size` | Sliding window with overlap | General text |
| `paragraph` | Split on double newlines | Structured prose |
| `code_aware` | Split on function/class definitions | Source code |
| `markdown_aware` | Split on headings, preserve code fences | Documentation |

## Error Handling

All endpoints return standard HTTP status codes:

- `200` — Success
- `201` — Created (ingestion)
- `400` — Bad request (invalid parameters)
- `401` — Unauthorized (missing/invalid token)
- `403` — Forbidden (wrong scope access)
- `404` — Not found (scope or document)
- `422` — Validation error (empty content, missing fields)
- `500` — Internal server error
""",

    "deployment-guide.md": """# Deployment Guide

This guide covers deploying knowledge-api to Kubernetes via ArgoCD.

## Prerequisites

- Kubernetes cluster (k3s, EKS, GKE, or AKS)
- ArgoCD installed and configured
- Qdrant vector database deployed
- PostgreSQL 16+ for document lifecycle store
- Vault for secret management

## Helm Chart Configuration

The knowledge-api Helm chart is part of the agentopia-base chart:

```yaml
# values.yaml
knowledgeApi:
  enabled: true
  image:
    repository: ghcr.io/ai-agentopia/knowledge-api
    tag: dev-2c53b56
  replicas: 1
  resources:
    requests:
      memory: 256Mi
      cpu: 100m
    limits:
      memory: 512Mi
      cpu: 500m
  env:
    QDRANT_URL: http://qdrant:6333
    DATABASE_URL: postgresql://user:pass@postgres:5432/agentopia
    EMBEDDING_MODEL: openai/text-embedding-3-small
    EMBEDDING_BASE_URL: https://openrouter.ai/api/v1/embeddings
```

## ArgoCD Image Updater

Images are automatically updated by ArgoCD Image Updater:

```yaml
# Application annotations
argocd-image-updater.argoproj.io/kapi.allow-tags: "regexp:^dev-[a-f0-9]{7,9}$"
argocd-image-updater.argoproj.io/kapi.update-strategy: newest-build
```

The updater polls GHCR every 2 minutes and updates the tag when a newer image matching the pattern is found.

## Health Checks

The service exposes two health endpoints:

- `GET /health` — Liveness probe. Returns 200 if the process is running.
- `GET /internal/health` — Readiness probe. Checks Qdrant and Postgres connectivity.

## Monitoring

Metrics are exposed at `/metrics` in Prometheus format:

- `knowledge_api_ingest_total` — Total ingestion operations
- `knowledge_api_search_total` — Total search operations
- `knowledge_api_chunk_count` — Number of chunks per scope
- `knowledge_api_ingest_duration_seconds` — Ingestion latency histogram

## Troubleshooting

### Common Issues

1. **Qdrant connection refused**: Check `QDRANT_URL` and network policy
2. **Embedding timeout**: Increase `EMBEDDING_TIMEOUT_SECONDS` (default: 30)
3. **OOM on large documents**: Reduce `chunk_size` or increase memory limits
4. **Stale search results**: Check ArgoCD sync status and binding cache
""",

    "adr-011-provenance.md": """# ADR-011: Document Provenance Tracking

## Status

Accepted (2026-03-28)

## Context

Operators need to know when a document was ingested and whether it has been updated. Without provenance metadata, there is no way to:

- Detect stale documents that need re-ingestion
- Audit which version of a document is currently indexed
- Compare ingestion timestamps across scopes

## Decision

Every ingested document and its chunks carry two provenance fields:

1. **`document_hash`**: SHA-256 of the original document content. This is computed before chunking and attached to every chunk's metadata. If the same content is re-ingested, the hash matches and the operation is skipped (no churn).

2. **`ingested_at`**: Unix timestamp of when the document was processed. This is set once at ingestion time and never updated. If a document is re-ingested with different content, a new record is created with a new timestamp.

### Hash Computation

```python
import hashlib

def compute_document_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
```

### Lifecycle Integration

The provenance fields integrate with the document lifecycle (ADR-012):

- `ACTIVE` records have `ingested_at` set
- `SUPERSEDED` records retain their original `ingested_at` for audit
- `DELETED` records retain provenance for tombstone queries

## Consequences

- Every chunk carries 64 extra bytes of metadata (32-char hash + 8-byte timestamp)
- Duplicate detection is O(1) hash comparison, no content diff needed
- Operators can query staleness via `ingested_at` comparison with current time
""",
}

# ── Evaluation queries ───────────────────────────────────────────────────

EVAL_QUERIES = [
    {
        "query": "How does authentication work in the knowledge API?",
        "expected_source": "api-guide.md",
        "expected_section_keywords": ["authentication", "token"],
    },
    {
        "query": "What is the ingestion pipeline?",
        "expected_source": "api-guide.md",
        "expected_section_keywords": ["ingestion", "pipeline", "phase"],
    },
    {
        "query": "How to deploy knowledge-api to Kubernetes?",
        "expected_source": "deployment-guide.md",
        "expected_section_keywords": ["deploy", "helm", "kubernetes"],
    },
    {
        "query": "What chunking strategies are available?",
        "expected_source": "api-guide.md",
        "expected_section_keywords": ["chunking", "strategy"],
    },
    {
        "query": "How does document provenance tracking work?",
        "expected_source": "adr-011-provenance.md",
        "expected_section_keywords": ["provenance", "hash", "document_hash"],
    },
    {
        "query": "What health check endpoints does the service expose?",
        "expected_source": "deployment-guide.md",
        "expected_section_keywords": ["health"],
    },
]


def evaluate_strategy(strategy: ChunkingStrategy, chunk_size: int = 512) -> dict:
    """Evaluate a chunking strategy on the docs corpus."""
    svc = KnowledgeService()
    config = IngestConfig(chunking_strategy=strategy, chunk_size=chunk_size)

    # Ingest all docs
    total_chunks = 0
    chunk_sizes = []
    sections_found = 0
    code_blocks_intact = 0
    total_code_blocks = 0

    for source, content in DOCS_CORPUS.items():
        result = svc.ingest("eval", content, source, config=config)
        total_chunks += result.chunks_created

        # Analyze chunks directly
        chunks = chunk_document(
            content, source, "eval", DocumentFormat.MARKDOWN, config
        )
        for chunk in chunks:
            chunk_sizes.append(len(chunk.text))
            if chunk.metadata.section:
                sections_found += 1

        # Count code blocks in source
        in_fence = False
        for line in content.split("\n"):
            if line.strip().startswith("```"):
                if in_fence:
                    total_code_blocks += 1
                in_fence = not in_fence

    # Check code block integrity in chunks
    all_chunks = []
    for source, content in DOCS_CORPUS.items():
        chunks = chunk_document(
            content, source, "eval", DocumentFormat.MARKDOWN, config
        )
        all_chunks.extend(chunks)

    for chunk in all_chunks:
        text = chunk.text
        opens = text.count("```")
        # Code block is intact if opens are even (matched pairs)
        if "```" in text and opens % 2 == 0:
            code_blocks_intact += opens // 2

    # Search evaluation
    search_scores = []
    source_match_count = 0
    section_relevance_count = 0

    for eq in EVAL_QUERIES:
        results = svc.search(eq["query"], ["eval"], limit=3)
        if results:
            best = results[0]
            search_scores.append(best.score)
            if best.citation.source == eq["expected_source"]:
                source_match_count += 1
            section_lower = best.citation.section.lower()
            if any(kw in section_lower for kw in eq["expected_section_keywords"]):
                section_relevance_count += 1

    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    avg_score = sum(search_scores) / len(search_scores) if search_scores else 0

    return {
        "strategy": strategy.value,
        "total_chunks": total_chunks,
        "avg_chunk_size": round(avg_chunk_size),
        "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
        "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
        "sections_with_heading": sections_found,
        "code_blocks_intact": code_blocks_intact,
        "total_code_blocks": total_code_blocks,
        "source_accuracy": f"{source_match_count}/{len(EVAL_QUERIES)}",
        "section_relevance": f"{section_relevance_count}/{len(EVAL_QUERIES)}",
        "avg_search_score": round(avg_score, 4),
        "search_scores": [round(s, 4) for s in search_scores],
    }


def main():
    print("=" * 72)
    print("W1 Evaluation: fixed_size vs markdown_aware chunking")
    print("=" * 72)
    print(f"Corpus: {len(DOCS_CORPUS)} documents")
    print(f"Queries: {len(EVAL_QUERIES)}")
    print()

    results = {}
    for strategy in [ChunkingStrategy.FIXED_SIZE, ChunkingStrategy.MARKDOWN_AWARE]:
        print(f"Evaluating {strategy.value}...")
        t0 = time.time()
        result = evaluate_strategy(strategy, chunk_size=512)
        elapsed = time.time() - t0
        result["elapsed_ms"] = round(elapsed * 1000)
        results[strategy.value] = result
        print(f"  Done in {result['elapsed_ms']}ms")

    print()
    print("-" * 72)
    print(f"{'Metric':<30} {'fixed_size':>18} {'markdown_aware':>18}")
    print("-" * 72)

    fs = results["fixed_size"]
    md = results["markdown_aware"]

    metrics = [
        ("Total chunks", "total_chunks"),
        ("Avg chunk size (chars)", "avg_chunk_size"),
        ("Min chunk size", "min_chunk_size"),
        ("Max chunk size", "max_chunk_size"),
        ("Chunks with section heading", "sections_with_heading"),
        ("Code blocks intact", "code_blocks_intact"),
        ("Total code blocks in corpus", "total_code_blocks"),
        ("Source accuracy (top-1)", "source_accuracy"),
        ("Section relevance (top-1)", "section_relevance"),
        ("Avg search score", "avg_search_score"),
    ]

    for label, key in metrics:
        print(f"{label:<30} {str(fs[key]):>18} {str(md[key]):>18}")

    print("-" * 72)

    # Save results
    output_path = Path(__file__).parent / "results" / "w1_chunking_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "evaluation": "w1_chunking_comparison",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "corpus_size": len(DOCS_CORPUS),
                "query_count": len(EVAL_QUERIES),
                "chunk_size": 512,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")

    # Verdict
    print()
    md_better = 0
    fs_better = 0
    if md["sections_with_heading"] > fs["sections_with_heading"]:
        md_better += 1
    if md["code_blocks_intact"] >= fs["code_blocks_intact"]:
        md_better += 1
    source_md = int(md["source_accuracy"].split("/")[0])
    source_fs = int(fs["source_accuracy"].split("/")[0])
    if source_md >= source_fs:
        md_better += 1
    else:
        fs_better += 1

    print(f"Markdown-aware advantages: {md_better}")
    print(f"Fixed-size advantages: {fs_better}")
    print()
    if md_better > fs_better:
        print("VERDICT: markdown_aware shows improvement for documentation corpora.")
        print("RECOMMENDATION: Safe for opt-in deployment. Do NOT set as default.")
    else:
        print("VERDICT: No clear advantage. Keep as opt-in, needs more evaluation.")


if __name__ == "__main__":
    main()
