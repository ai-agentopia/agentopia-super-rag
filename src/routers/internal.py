"""knowledge-api internal routes (#320, Phase 2b).

Internal-only endpoints:
  GET  /internal/health         — detailed health + binding cache state
  POST /internal/binding-sync   — bot-config-api notifies binding change (deploy)
  DELETE /internal/binding-sync/{bot_name} — bot-config-api notifies deletion
  GET  /internal/binding-sync/{bot_name}   — cache-miss lookup (bot-config-api proxy fallback)

All routes require X-Internal-Token auth.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.guards import require_internal_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal", dependencies=[Depends(require_internal_auth)])


# ── Request models ────────────────────────────────────────────────────────────


class BindingSyncRequest(BaseModel):
    bot_name: str
    client_id: str
    knowledge_scopes: list[str]


# ── Health ────────────────────────────────────────────────────────────────────


@router.get("/health")
async def internal_health() -> dict[str, Any]:
    """Detailed health including binding cache state and upstream connectivity."""
    from services.binding_cache import get_binding_cache
    from services.knowledge import get_knowledge_service

    result: dict[str, Any] = {
        "service": "knowledge-api",
        "status": "ok",
        "binding_cache": get_binding_cache().summary(),
    }

    # Qdrant connectivity
    try:
        svc = get_knowledge_service()
        if svc._qdrant:
            svc._qdrant.health_check()
            result["qdrant"] = "ok"
        else:
            result["qdrant"] = "in-memory (no QDRANT_URL)"
    except Exception as exc:
        result["qdrant"] = f"error: {exc}"
        result["status"] = "degraded"

    # Proxy mode config
    result["proxy_mode"] = {
        "knowledge_api_url": os.getenv("KNOWLEDGE_API_URL", ""),
        "internal_token_configured": bool(os.getenv("KNOWLEDGE_API_INTERNAL_TOKEN")),
    }

    return result


# ── Binding sync ──────────────────────────────────────────────────────────────


@router.post("/binding-sync", status_code=200)
async def sync_binding(body: BindingSyncRequest) -> dict[str, Any]:
    """Sync a bot's knowledge binding (called by bot-config-api on deploy/update).

    Immediately updates the in-memory cache. No persistence needed here —
    bot-config-api DB (bot_knowledge_bindings table) is the durable source of
    truth (#KB-BINDING-V2). Cache is rebuilt from control-plane on startup.
    """
    from services.binding_cache import get_binding_cache

    cache = get_binding_cache()
    cache.update(
        bot_name=body.bot_name,
        client_id=body.client_id,
        knowledge_scopes=body.knowledge_scopes,
    )
    logger.info(
        "binding_sync: updated bot=%s client=%s scopes=%s",
        body.bot_name, body.client_id, body.knowledge_scopes,
    )
    return {
        "status": "synced",
        "bot_name": body.bot_name,
        "scopes": body.knowledge_scopes,
        "total_bots": cache.bot_count(),
    }


@router.delete("/binding-sync/{bot_name}", status_code=200)
async def remove_binding(bot_name: str) -> dict[str, Any]:
    """Remove a bot's knowledge binding (called by bot-config-api on delete)."""
    from services.binding_cache import get_binding_cache

    cache = get_binding_cache()
    cache.remove(bot_name)
    logger.info("binding_sync: removed bot=%s", bot_name)
    return {"status": "removed", "bot_name": bot_name, "total_bots": cache.bot_count()}


@router.get("/binding-sync/{bot_name}", status_code=200)
async def lookup_binding(bot_name: str) -> dict[str, Any]:
    """Cache-miss lookup: returns binding for bot_name including K8s fallback.

    Used by bot-config-api proxy when it needs to verify scope access without
    knowledge-api being the cache-miss handler itself.
    """
    from services.binding_cache import get_binding_cache

    cache = get_binding_cache()
    binding = cache.resolve_with_fallback(bot_name)
    if not binding:
        raise HTTPException(status_code=404, detail=f"No binding found for bot '{bot_name}'")

    return {
        "bot_name": bot_name,
        "client_id": binding.client_id,
        "knowledge_scopes": binding.knowledge_scopes,
        "resolved_scopes": binding.resolved_scopes(),
    }
