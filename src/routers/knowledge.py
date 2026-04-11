"""knowledge-api knowledge routes (#320, Phase 2b).

Implements the same API surface as bot-config-api/routers/knowledge.py.
Auth model differs: writes require internal proxy token; reads accept
internal proxy OR direct bot bearer.

Scope resolution for bots uses the local BindingCache (not BotKnowledgeIndex).
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile

from auth.guards import require_internal_auth, require_knowledge_read
from models.knowledge import (
    ChunkingStrategy,
    DocumentFormat,
    OrchestratorIngestRequest,
    OrchestratorIngestResponse,
)
from services.knowledge import get_knowledge_service, parse_html, parse_pdf

logger = logging.getLogger(__name__)

# ── Extension-to-format mapping ──────────────────────────────────────────────
_EXT_FORMAT = {
    ".pdf": DocumentFormat.PDF,
    ".html": DocumentFormat.HTML,
    ".htm": DocumentFormat.HTML,
    ".md": DocumentFormat.MARKDOWN,
    ".py": DocumentFormat.CODE,
    ".ts": DocumentFormat.CODE,
    ".js": DocumentFormat.CODE,
    ".go": DocumentFormat.CODE,
    ".rs": DocumentFormat.CODE,
    ".java": DocumentFormat.CODE,
    ".txt": DocumentFormat.TEXT,
}

# ── Sub-routers ──────────────────────────────────────────────────────────────
# Write routes: internal proxy only (operator auth handled by bot-config-api)
write_router = APIRouter(dependencies=[Depends(require_internal_auth)])
# Read routes: internal proxy OR direct bot bearer
read_router = APIRouter()


# ── Scope resolution helper ───────────────────────────────────────────────────


def _resolve_bot_scopes(bot_name: str) -> list[str]:
    """Resolve bot_name → subscribed scopes via local BindingCache.

    Cache-miss triggers K8s fallback lookup.
    Raises 403 if bot has no knowledge bindings.
    """
    from services.binding_cache import get_binding_cache

    cache = get_binding_cache()
    binding = cache.resolve_with_fallback(bot_name)
    if not binding:
        logger.warning("knowledge_auth_denied: bot=%s reason=no_knowledge_binding", bot_name)
        raise HTTPException(status_code=403, detail="bot has no knowledge scope subscriptions")
    return binding.resolved_scopes()


# ── Read routes ──────────────────────────────────────────────────────────────


@read_router.get("/scopes")
async def list_scopes(
    auth_ctx=Depends(require_knowledge_read),
) -> dict[str, Any]:
    """List knowledge scopes. Internal: all. Bot: subscribed only."""
    svc = get_knowledge_service()
    auth_type, identity = auth_ctx

    if auth_type == "bot":
        allowed = set(_resolve_bot_scopes(identity))
        scopes = [s for s in svc.list_scopes() if s.name in allowed]
    else:
        scopes = svc.list_scopes()

    return {
        "scopes": [
            {
                "name": s.name,
                "document_count": s.document_count,
                "chunk_count": s.chunk_count,
                "last_indexed": s.last_indexed,
            }
            for s in scopes
        ],
        "count": len(scopes),
    }


@read_router.get("/stale")
async def list_stale_scopes(
    max_age_secs: float = Query(86400.0),
    auth_ctx=Depends(require_knowledge_read),
) -> dict[str, Any]:
    """Return stale scopes. Internal: all. Bot: subscribed only."""
    svc = get_knowledge_service()
    auth_type, identity = auth_ctx
    stale = svc.list_stale_scopes(max_age_secs=max_age_secs)

    if auth_type == "bot":
        allowed = set(_resolve_bot_scopes(identity))
        stale = [s for s in stale if s in allowed]

    return {"stale_scopes": stale, "count": len(stale)}


@read_router.get("/search")
async def search_knowledge(
    query: str = Query(..., min_length=1),
    scopes: list[str] = Query(default=[]),
    limit: int = Query(default=5, ge=1, le=50),
    query_expansion: bool = Query(default=False, description="Enable W3a query expansion (per-scope opt-in)"),
    hyde: bool = Query(default=False, description="Enable W3b HyDE retrieval (per-scope opt-in)"),
    rerank: bool = Query(default=False, description="Enable W4 LLM listwise reranking (per-scope opt-in)"),
    auth_ctx=Depends(require_knowledge_read),
) -> dict[str, Any]:
    """Search knowledge. Bot: server-side scope resolution via BindingCache."""
    svc = get_knowledge_service()
    auth_type, identity = auth_ctx

    if query_expansion and hyde:
        raise HTTPException(
            status_code=400,
            detail="query_expansion and hyde cannot both be enabled",
        )
    if rerank and query_expansion:
        raise HTTPException(
            status_code=400,
            detail="rerank and query_expansion cannot both be enabled",
        )
    if rerank and hyde:
        raise HTTPException(
            status_code=400,
            detail="rerank and hyde cannot both be enabled",
        )

    if auth_type == "bot":
        effective_scopes = _resolve_bot_scopes(identity)
        logger.info(
            "knowledge_search: bot=%s resolved_scopes=%s query_len=%d expansion=%s hyde=%s rerank=%s",
            identity, effective_scopes, len(query), query_expansion, hyde, rerank,
        )
    else:
        effective_scopes = scopes if scopes else [s.name for s in svc.list_scopes()]

    if not effective_scopes:
        return {"results": [], "count": 0}

    results = svc.search(
        query=query,
        scopes=effective_scopes,
        limit=limit,
        query_expansion_enabled=query_expansion,
        hyde_enabled=hyde,
        rerank_enabled=rerank,
    )
    return {"results": [r.model_dump() for r in results], "count": len(results)}


@read_router.get("/{scope}")
async def get_scope(
    scope: str,
    auth_ctx=Depends(require_knowledge_read),
) -> dict[str, Any]:
    scope = scope.replace("--", "/")
    auth_type, identity = auth_ctx
    if auth_type == "bot":
        allowed = set(_resolve_bot_scopes(identity))
        if scope not in allowed:
            raise HTTPException(status_code=403, detail="scope not in bot subscriptions")

    svc = get_knowledge_service()
    s = svc.get_scope(scope)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
    return {
        "name": s.name,
        "document_count": s.document_count,
        "chunk_count": s.chunk_count,
        "last_indexed": s.last_indexed,
    }


@read_router.get("/{scope}/documents")
async def list_documents(
    scope: str,
    auth_ctx=Depends(require_knowledge_read),
) -> dict[str, Any]:
    scope = scope.replace("--", "/")
    auth_type, identity = auth_ctx
    if auth_type == "bot":
        allowed = set(_resolve_bot_scopes(identity))
        if scope not in allowed:
            raise HTTPException(status_code=403, detail="scope not in bot subscriptions")

    svc = get_knowledge_service()
    if svc.get_scope(scope) is None:
        raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
    docs = svc.list_documents(scope)
    return {"scope": scope, "documents": docs}


# ── Write routes (internal proxy only) ───────────────────────────────────────


@write_router.post("/webhook", status_code=410)
async def webhook_ingest() -> dict[str, Any]:
    """Webhook ingestion retired. Use POST /{scope}/ingest."""
    raise HTTPException(status_code=410, detail="Webhook ingestion is retired.")


@write_router.post("/{scope}/ingest", status_code=201)
async def ingest_file(
    scope: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    scope = scope.replace("--", "/")
    svc = get_knowledge_service()
    raw_bytes = await file.read()
    filename = file.filename or "upload"

    import os as _os
    ext = _os.path.splitext(filename)[1].lower()
    fmt = _EXT_FORMAT.get(ext, DocumentFormat.TEXT)

    if fmt == DocumentFormat.PDF:
        try:
            content = parse_pdf(raw_bytes)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {exc}")
    elif fmt == DocumentFormat.HTML:
        try:
            content = parse_html(raw_bytes)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to parse HTML: {exc}")
    else:
        content = raw_bytes.decode("utf-8", errors="replace")

    if not content.strip():
        raise HTTPException(status_code=422, detail="File contains no extractable text")

    result = svc.ingest(scope=scope, content=content, source=filename, format=fmt)
    return {
        "status": "ingested",
        "scope": scope,
        "source": filename,
        "format": fmt.value,
        "chunks_created": result.chunks_created,
        "chunks_skipped": result.chunks_skipped,
        "document_hash": result.document_hash,
        "ingested_at": result.ingested_at,
    }


@write_router.post("/{scope}/ingest-document", response_model=OrchestratorIngestResponse, status_code=201)
async def ingest_document_from_orchestrator(
    scope: str,
    body: OrchestratorIngestRequest,
) -> OrchestratorIngestResponse:
    """Ingest pre-parsed document text from agentopia-knowledge-ingest Orchestrator.

    Called after normalization and extraction complete upstream.
    Accepts plain text + structured metadata; handles chunking and embedding here.
    Tags every chunk with document_id, version, and status=active in Qdrant payload.
    Supersedes prior-version chunks for the same document_id in the same scope.

    Idempotent: if (document_id, version) already indexed as active, returns without re-embedding.
    """
    scope = scope.replace("--", "/")
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text field is empty after normalization")

    svc = get_knowledge_service()
    result = svc.ingest_from_orchestrator(scope=scope, request=body)
    return result


@write_router.post("/{scope}/reindex")
async def reindex_scope(scope: str) -> dict[str, Any]:
    scope = scope.replace("--", "/")
    svc = get_knowledge_service()
    result = svc.reindex(scope)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
    return result


@write_router.delete("/{scope}")
async def delete_scope(scope: str) -> dict[str, Any]:
    scope = scope.replace("--", "/")
    svc = get_knowledge_service()
    deleted = svc.delete_scope(scope)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
    return {"status": "deleted", "scope": scope}


@write_router.delete("/{scope}/documents/{source:path}")
async def delete_document(scope: str, source: str) -> dict[str, Any]:
    scope = scope.replace("--", "/")
    svc = get_knowledge_service()
    if svc.get_scope(scope) is None:
        raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
    removed = svc.delete_document(scope, source)
    if removed == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for source '{source}' in scope '{scope}'",
        )
    return {"status": "deleted", "scope": scope, "source": source, "chunks_removed": removed}


# ── Combined router ───────────────────────────────────────────────────────────
router = APIRouter()
router.include_router(read_router)
router.include_router(write_router)
