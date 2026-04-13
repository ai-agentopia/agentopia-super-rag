"""knowledge-api — Production-grade RAG service.

Standalone FastAPI service. Source of truth: ai-agentopia/agentopia-super-rag.
Extraction from agentopia-protocol complete 2026-04-09.

Routes:
  GET/POST/DELETE /api/v1/knowledge/* — knowledge management
  GET/POST/DELETE /internal/*         — internal service API (binding sync, health)
  GET /health                         — liveness probe

Auth:
  Write routes: X-Internal-Token (from bot-config-api proxy)
  Read routes:  X-Internal-Token OR bot bearer (Authorization + X-Bot-Name)
  Internal routes: X-Internal-Token

Binding sync: startup rebuild from bot-config-api control-plane (V2, #KB-BINDING-V2),
sync webhook, cache-miss fallback to control-plane, periodic reconcile.
K8s CRD annotations are DEPRECATED as binding source (transitional fallback only).
"""

import asyncio
import json as _json
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.knowledge import router as knowledge_router
from routers.internal import router as internal_router
from routers.evaluation import router as evaluation_router

VERSION = "1.0.0"


# ── Logging setup ─────────────────────────────────────────────────────────────
class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return _json.dumps(entry, ensure_ascii=False)


if os.getenv("LOG_FORMAT", "text") == "json":
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(_JSONFormatter())
    logging.root.handlers = [_handler]
    logging.root.setLevel(logging.INFO)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("knowledge-api v%s starting", VERSION)

    # ── Startup: rebuild binding cache from control-plane (#KB-BINDING-V2) ──
    from services.binding_cache import get_binding_cache

    cache = get_binding_cache()
    bot_config_api_url = os.getenv("BOT_CONFIG_API_URL", "")
    k8s_enabled = bool(os.getenv("KUBERNETES_SERVICE_HOST") or os.getenv("KUBECONFIG"))
    if bot_config_api_url or k8s_enabled:
        try:
            count = cache.rebuild_from_control_plane()
            logger.info("binding_cache: startup rebuild complete — %d bots", count)
        except Exception:
            logger.warning("binding_cache: startup rebuild failed — cache is empty", exc_info=True)
    else:
        logger.info("binding_cache: no control-plane or K8s available, starting with empty cache (local dev)")

    # ── Evaluation schema migration ───────────────────────────────────────
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        try:
            import pathlib, psycopg
            db_dir = pathlib.Path(__file__).parent.parent / "db"
            conn = psycopg.connect(database_url, autocommit=True)
            for migration in ("025_evaluation.sql", "026_eval_fixes.sql"):
                sql_path = db_dir / migration
                if sql_path.exists():
                    conn.execute(sql_path.read_text())
                    logger.info("evaluation: applied migration %s", migration)
            conn.close()
        except Exception as exc:
            logger.warning("evaluation: schema migration failed (non-fatal): %s", exc)

    # ── Qdrant connectivity check ─────────────────────────────────────────
    qdrant_url = os.getenv("QDRANT_URL", "")
    if qdrant_url:
        try:
            from services.knowledge import get_knowledge_service

            svc = get_knowledge_service()
            if svc._qdrant:
                svc._qdrant.health_check()
                logger.info("Qdrant: connected at %s", qdrant_url)
        except Exception as exc:
            logger.warning("Qdrant: connectivity check failed: %s", exc)
    else:
        logger.info("Qdrant: QDRANT_URL not set — using in-memory fallback")

    # ── Periodic binding reconcile background task ────────────────────────
    from services.binding_cache import BINDING_RECONCILE_INTERVAL_SECS
    _reconcile_task = None

    if (bot_config_api_url or k8s_enabled) and BINDING_RECONCILE_INTERVAL_SECS > 0:
        async def _reconcile_loop():
            while True:
                await asyncio.sleep(BINDING_RECONCILE_INTERVAL_SECS)
                try:
                    count = cache.rebuild_from_control_plane()
                    logger.info("binding_cache: periodic reconcile — %d bots", count)
                except Exception:
                    logger.warning("binding_cache: periodic reconcile failed", exc_info=True)

        _reconcile_task = asyncio.create_task(_reconcile_loop())
        logger.info(
            "binding_cache: periodic reconcile every %.0fs", BINDING_RECONCILE_INTERVAL_SECS,
        )

    logger.info("knowledge-api v%s ready", VERSION)
    yield

    if _reconcile_task and not _reconcile_task.done():
        _reconcile_task.cancel()
        logger.info("binding_cache: reconcile task stopped")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="knowledge-api",
    description="Agentopia knowledge service — RAG ingestion, search, and scope management",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(knowledge_router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(internal_router, tags=["internal"])
app.include_router(evaluation_router, prefix="/api/v1", tags=["evaluation"])


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok", "service": "knowledge-api", "version": VERSION}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_config=None,  # use our logging setup
        access_log=True,
    )
