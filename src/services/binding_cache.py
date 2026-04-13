"""Bot-scope binding cache for knowledge-api (#320, Phase 2b, V2 #KB-BINDING-V2).

Source of truth: bot_knowledge_bindings table in bot-config-api Postgres.
knowledge-api reads/syncs bindings via:
  1. Startup rebuild from bot-config-api control-plane API (GET /api/v1/runtime/knowledge-bindings)
  2. Sync webhook from bot-config-api (POST /internal/binding-sync on deploy/update/delete)
  3. Cache-miss fallback: bot-config-api runtime endpoint (GET /api/v1/runtime/bots/{bot}/knowledge-binding)
  4. Periodic reconcile: full rebuild from control-plane every BINDING_RECONCILE_INTERVAL_SECS (default 300s)

K8s CRD annotation reads are DEPRECATED as of V2. Kept as emergency fallback only.
Remove K8s fallback in V2.4 cleanup once full migration is confirmed stable.

This module owns the binding cache lifecycle. Hot path (resolve) is O(1) dict lookup.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

BINDING_RECONCILE_INTERVAL_SECS = float(
    os.getenv("BINDING_RECONCILE_INTERVAL_SECS", "300")
)
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "agentopia")
ARGOCD_NAMESPACE = os.getenv("ARGOCD_NAMESPACE", "argocd")


def _bot_config_api_url() -> str:
    """URL of bot-config-api control-plane (V2 binding authority)."""
    return os.getenv("BOT_CONFIG_API_URL", "").rstrip("/")


def _internal_token() -> str:
    """Internal token sent as X-Internal-Token to bot-config-api runtime API."""
    return os.getenv("INTERNAL_API_TOKEN", "")


@dataclass
class BotKnowledgeBinding:
    """A bot's knowledge scope subscription."""

    client_id: str
    knowledge_scopes: list[str] = field(default_factory=list)
    synced_at: float = field(default_factory=time.time)

    def resolved_scopes(self) -> list[str]:
        """Return canonical scope identities for this bot.

        Format: {client_id}/{scope_name}  (canonical — may contain /)
        Example: acme-corp/api-docs

        Physical Qdrant collection names are derived separately at the
        storage boundary via QdrantBackend._qdrant_collection_name(), which
        hashes the canonical identity to kb-{sha256_hex[:16]} (#327 fix).

        Scope-level cross-client isolation is preserved: two clients with the
        same short scope name produce distinct canonical identities and
        distinct physical collections.
        """
        return [f"{self.client_id}/{sn}" for sn in self.knowledge_scopes]


class BindingCache:
    """In-memory binding cache with K8s-backed reconcile.

    Thread-safe for single-process async FastAPI apps (no locking needed
    since Python GIL protects dict reads/writes from asyncio tasks).
    """

    def __init__(self) -> None:
        self._bindings: dict[str, BotKnowledgeBinding] = {}
        self._last_reconcile: float = 0.0

    # ── Write operations (from sync webhook + reconcile) ──────────────────

    def update(
        self, bot_name: str, client_id: str, knowledge_scopes: list[str],
    ) -> None:
        """Add or update a bot's binding. Called by sync webhook on deploy."""
        if not client_id or not knowledge_scopes:
            self._bindings.pop(bot_name, None)
            logger.info("binding_cache: removed bot=%s (no scopes)", bot_name)
            return
        self._bindings[bot_name] = BotKnowledgeBinding(
            client_id=client_id,
            knowledge_scopes=knowledge_scopes,
        )
        logger.info(
            "binding_cache: updated bot=%s client=%s scopes=%s",
            bot_name, client_id, knowledge_scopes,
        )

    def remove(self, bot_name: str) -> None:
        """Remove a bot's binding. Called by sync webhook on delete."""
        removed = self._bindings.pop(bot_name, None)
        if removed:
            logger.info("binding_cache: removed bot=%s", bot_name)

    # ── Read operations (hot path) ─────────────────────────────────────────

    def resolve(self, bot_name: str) -> BotKnowledgeBinding | None:
        """O(1) binding lookup. Returns None on cache miss."""
        return self._bindings.get(bot_name)

    def resolve_with_fallback(self, bot_name: str) -> BotKnowledgeBinding | None:
        """Cache-miss fallback: if not in cache, query bot-config-api control-plane.

        V2: Fallback uses DB-backed control-plane API, not K8s CRD.
        On success, populates cache for future hits.
        """
        binding = self._bindings.get(bot_name)
        if binding:
            return binding

        logger.info("binding_cache: cache miss for bot=%s — querying control-plane", bot_name)
        try:
            binding = self._fetch_from_control_plane_for_bot(bot_name)
            if binding:
                self._bindings[bot_name] = binding
                logger.info(
                    "binding_cache: populated from control-plane bot=%s scopes=%s",
                    bot_name, binding.knowledge_scopes,
                )
            return binding
        except Exception as exc:
            logger.warning(
                "binding_cache: control-plane fallback failed for bot=%s: %s", bot_name, exc,
            )
            return None

    def bot_count(self) -> int:
        return len(self._bindings)

    def summary(self) -> dict:
        """Return cache state summary for diagnostics."""
        return {
            "bot_count": len(self._bindings),
            "last_reconcile": self._last_reconcile,
            "reconcile_interval_secs": BINDING_RECONCILE_INTERVAL_SECS,
            "bots": list(self._bindings.keys()),
        }

    # ── Reconcile (startup + periodic) ────────────────────────────────────

    def rebuild_from_control_plane(self) -> int:
        """Full rebuild from bot-config-api DB-backed control-plane API.

        V2 primary rebuild path (replaces rebuild_from_k8s).
        Called on startup and by periodic reconcile task.
        Falls back to rebuild_from_k8s() if BOT_CONFIG_API_URL is not set.
        Returns number of bots with knowledge bindings found.
        """
        base = _bot_config_api_url()
        if not base:
            logger.warning(
                "binding_cache: BOT_CONFIG_API_URL not set — falling back to K8s rebuild"
            )
            return self.rebuild_from_k8s()

        import urllib.request

        token = _internal_token()
        headers = {"X-Internal-Token": token} if token else {}
        req = urllib.request.Request(
            f"{base}/api/v1/runtime/knowledge-bindings",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            logger.warning(
                "binding_cache: control-plane bulk rebuild failed: %s — falling back to K8s", exc,
            )
            return self.rebuild_from_k8s()

        new_bindings: dict[str, BotKnowledgeBinding] = {}
        for entry in data.get("bindings", []):
            bot_name = entry.get("bot_name", "")
            client_id = entry.get("client_id", "")
            scopes = entry.get("knowledge_scopes", [])
            if bot_name and client_id and scopes:
                new_bindings[bot_name] = BotKnowledgeBinding(
                    client_id=client_id, knowledge_scopes=scopes,
                )

        self._bindings = new_bindings
        self._last_reconcile = time.time()
        count = len(new_bindings)
        logger.info("binding_cache: rebuilt from control-plane — %d bots with knowledge", count)
        return count

    def rebuild_from_k8s(self) -> int:
        """Full rebuild from Application CRD annotations.

        DEPRECATED: Use rebuild_from_control_plane() for V2.
        Kept as emergency fallback when BOT_CONFIG_API_URL is not configured.
        Will be removed in V2.4 cleanup.
        """
        try:
            bots = self._list_bot_applications_from_k8s()
        except Exception as exc:
            logger.warning(
                "binding_cache: K8s unavailable, skipping rebuild: %s", exc,
            )
            return 0

        new_bindings: dict[str, BotKnowledgeBinding] = {}
        count = 0
        for bot in bots:
            bot_name = bot.get("bot_name", "")
            client_id = bot.get("client_id", "")
            scopes_raw = bot.get("knowledge_scopes", "")
            if not bot_name or not client_id or not scopes_raw:
                continue
            try:
                scopes = (
                    json.loads(scopes_raw) if isinstance(scopes_raw, str) else scopes_raw
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "binding_cache: invalid scopes JSON for bot=%s: %s", bot_name, scopes_raw,
                )
                continue
            if isinstance(scopes, list) and scopes:
                new_bindings[bot_name] = BotKnowledgeBinding(
                    client_id=client_id, knowledge_scopes=scopes,
                )
                count += 1

        self._bindings = new_bindings
        self._last_reconcile = time.time()
        logger.info(
            "binding_cache: rebuilt from K8s (transitional fallback) — %d bots with knowledge",
            count,
        )
        return count

    # ── Control-plane helpers (V2) ────────────────────────────────────────

    def _fetch_from_control_plane_for_bot(self, bot_name: str) -> BotKnowledgeBinding | None:
        """Cache-miss fallback: query bot-config-api runtime endpoint.

        V2: replaces _fetch_from_k8s_for_bot as the primary fallback.
        Falls back to K8s if BOT_CONFIG_API_URL is not set (transitional).
        """
        base = _bot_config_api_url()
        if not base:
            logger.debug(
                "binding_cache: BOT_CONFIG_API_URL not set, using K8s fallback for bot=%s", bot_name,
            )
            return self._fetch_from_k8s_for_bot(bot_name)

        import urllib.request

        token = _internal_token()
        headers = {"X-Internal-Token": token} if token else {}
        req = urllib.request.Request(
            f"{base}/api/v1/runtime/bots/{bot_name}/knowledge-binding",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            logger.warning(
                "binding_cache: control-plane fallback failed for bot=%s: %s — trying K8s",
                bot_name, exc,
            )
            # Transitional: try K8s if control-plane HTTP fails
            return self._fetch_from_k8s_for_bot(bot_name)

        if not data.get("enabled"):
            return None

        client_id = data.get("client_id") or ""
        scopes = data.get("knowledge_scopes") or []
        if not client_id or not scopes:
            return None

        return BotKnowledgeBinding(client_id=client_id, knowledge_scopes=scopes)

    # ── K8s helpers (DEPRECATED — transitional fallback only) ─────────────

    def _get_k8s_client(self):
        from kubernetes import client as k8s_client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        return k8s_client

    def _list_bot_applications_from_k8s(self) -> list[dict]:
        """Read Application CRDs for this env, keyed by the bot slug label.

        Label selector:
          agentopia/managed-by=bot-config-api  — created by bot-config-api (not legacy AppSet)
          agentopia/env={K8S_NAMESPACE}         — scope to this environment's bots

        Bot identity is labels["agentopia/bot"] (the slug used everywhere as the
        runtime identity). NOT annotations["agentopia/agent-name"] which is the
        display name and may differ from the bot slug.
        """
        k8s_client = self._get_k8s_client()
        custom_api = k8s_client.CustomObjectsApi()

        label_selector = (
            f"agentopia/managed-by=bot-config-api,agentopia/env={K8S_NAMESPACE}"
        )

        try:
            result = custom_api.list_namespaced_custom_object(
                group="argoproj.io",
                version="v1alpha1",
                namespace=ARGOCD_NAMESPACE,
                plural="applications",
                label_selector=label_selector,
            )
        except Exception as exc:
            logger.warning("binding_cache: failed to list Applications: %s", exc)
            return []

        bots = []
        for item in result.get("items", []):
            meta = item.get("metadata", {})
            # Bot slug from label — the runtime identity used by all auth paths
            bot_name = meta.get("labels", {}).get("agentopia/bot", "")
            annotations = meta.get("annotations", {})
            client_id = annotations.get("agentopia/client-id", "")
            knowledge_scopes = annotations.get("agentopia/knowledge-scopes", "")
            if bot_name:
                bots.append({
                    "bot_name": bot_name,
                    "client_id": client_id,
                    "knowledge_scopes": knowledge_scopes,
                })
        return bots

    def _fetch_from_k8s_for_bot(self, bot_name: str) -> BotKnowledgeBinding | None:
        """Read a single Application CRD for cache-miss fallback."""
        k8s_client = self._get_k8s_client()
        custom_api = k8s_client.CustomObjectsApi()

        try:
            item = custom_api.get_namespaced_custom_object(
                group="argoproj.io",
                version="v1alpha1",
                namespace=ARGOCD_NAMESPACE,
                plural="applications",
                name=f"agentopia-{bot_name}",
            )
        except Exception as exc:
            logger.warning(
                "binding_cache: failed to read Application for bot=%s: %s", bot_name, exc,
            )
            return None

        annotations = item.get("metadata", {}).get("annotations", {})
        client_id = annotations.get("agentopia/client-id", "")
        scopes_raw = annotations.get("agentopia/knowledge-scopes", "")
        if not client_id or not scopes_raw:
            return None

        try:
            scopes = json.loads(scopes_raw) if isinstance(scopes_raw, str) else scopes_raw
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(scopes, list) or not scopes:
            return None

        return BotKnowledgeBinding(client_id=client_id, knowledge_scopes=scopes)


# ── Singleton ─────────────────────────────────────────────────────────────────

_cache: BindingCache | None = None


def get_binding_cache() -> BindingCache:
    """Return the singleton BindingCache."""
    global _cache
    if _cache is None:
        _cache = BindingCache()
    return _cache
