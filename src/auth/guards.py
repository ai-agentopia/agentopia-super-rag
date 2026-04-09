"""knowledge-api auth guards (#320, Phase 2b).

Two auth paths:
  1. Internal proxy:  X-Internal-Token header (from bot-config-api proxy)
  2. Bot bearer:      Authorization: Bearer {relay_token} + X-Bot-Name header

Internal proxy auth is used for:
  - All write operations (ingest, delete, reindex) — operator writes are
    proxied by bot-config-api which handles operator session auth first.
  - Operator read operations proxied from bot-config-api.

Bot bearer auth is used for:
  - Direct bot search/read — gateway bots call knowledge-api directly.
  - knowledge-api reads K8s Secret agentopia-gateway-env-{bot_name},
    key AGENTOPIA_RELAY_TOKEN to verify the bearer token.
    This is the same contract used by bot-config-api's relay.py.
"""

import logging
import os
import secrets as _secrets

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)


def _k8s_enabled() -> bool:
    return bool(os.getenv("KUBERNETES_SERVICE_HOST") or os.getenv("KUBECONFIG"))


def _internal_token() -> str:
    return os.getenv("KNOWLEDGE_API_INTERNAL_TOKEN", "")


def _auth_enabled() -> bool:
    return bool(_internal_token())


# ── Internal proxy auth ───────────────────────────────────────────────────────


async def require_internal_auth(
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> None:
    """Require internal service token (from bot-config-api proxy).

    Raises 401 if token is missing or invalid.
    Bypassed when KNOWLEDGE_API_INTERNAL_TOKEN is not set (local dev).
    """
    token = _internal_token()
    if not token:
        return  # local dev: bypass

    if not x_internal_token:
        raise HTTPException(status_code=401, detail="X-Internal-Token required")

    if not _secrets.compare_digest(x_internal_token, token):
        logger.warning("knowledge_api_auth_denied: reason=invalid_internal_token")
        raise HTTPException(status_code=401, detail="invalid internal token")


# ── Bot bearer auth ───────────────────────────────────────────────────────────


def _verify_relay_token(bot_name: str, bearer_token: str) -> None:
    """Verify bot relay token against K8s Secret agentopia-gateway-env-{bot_name}.

    Reads key AGENTOPIA_RELAY_TOKEN — the same contract used by bot-config-api's
    relay.py and K8sService.get_relay_token(). NOT the Telegram bot token.

    Raises HTTPException 401 on failure.
    Skipped when K8s is not available (local dev).
    """
    if not _k8s_enabled():
        return  # local dev: skip K8s Secret lookup

    try:
        from kubernetes import client as k8s_client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()

        v1 = k8s_client.CoreV1Api()
        namespace = os.getenv("K8S_NAMESPACE", "agentopia")
        secret_name = f"agentopia-gateway-env-{bot_name}"

        try:
            secret = v1.read_namespaced_secret(secret_name, namespace)
        except Exception:
            logger.warning(
                "knowledge_auth_denied: bot=%s reason=gateway_env_secret_not_found secret=%s",
                bot_name, secret_name,
            )
            raise HTTPException(status_code=401, detail="bot relay token not found")

        import base64
        raw = (secret.data or {}).get("AGENTOPIA_RELAY_TOKEN", "")
        if isinstance(raw, bytes):
            expected = base64.b64decode(raw).decode()
        else:
            expected = base64.b64decode(raw.encode()).decode() if raw else ""

        if not expected or not _secrets.compare_digest(bearer_token, expected):
            logger.warning(
                "knowledge_auth_denied: bot=%s reason=invalid_relay_token", bot_name,
            )
            raise HTTPException(status_code=401, detail="invalid bot relay token")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("knowledge_auth: K8s Secret read failed for bot=%s: %s", bot_name, exc)
        raise HTTPException(status_code=401, detail="token verification failed")


async def require_bot_bearer(
    request: Request,
    authorization: str | None = Header(default=None),
    x_bot_name: str | None = Header(default=None, alias="X-Bot-Name"),
) -> str:
    """Require valid bot bearer token.

    Returns bot_name on success. Raises 401 on failure.
    """
    if not authorization or not x_bot_name:
        raise HTTPException(
            status_code=401,
            detail="Authorization (Bearer) and X-Bot-Name headers required for bot access",
        )

    bearer = authorization.removeprefix("Bearer ").strip()
    if not bearer:
        raise HTTPException(status_code=401, detail="Bearer token missing")

    _verify_relay_token(x_bot_name, bearer)
    return x_bot_name


# ── Dual-path read auth ───────────────────────────────────────────────────────


async def require_knowledge_read(
    request: Request,
    authorization: str | None = Header(default=None),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
    x_bot_name: str | None = Header(default=None, alias="X-Bot-Name"),
) -> tuple[str, str | None]:
    """Dual-path auth for knowledge read routes.

    Returns: ("internal", None) for proxy requests,
             ("bot", bot_name) for direct bot bearer requests.

    Priority: internal proxy > bot bearer > dev bypass.
    """
    token = _internal_token()

    # Path 1: internal proxy (bot-config-api operator path)
    if x_internal_token:
        if token and not _secrets.compare_digest(x_internal_token, token):
            logger.warning("knowledge_auth_denied: reason=invalid_internal_token")
            raise HTTPException(status_code=401, detail="invalid internal token")
        return ("internal", None)

    # Path 2: bot bearer (direct gateway → knowledge-api)
    if authorization and x_bot_name:
        bearer = authorization.removeprefix("Bearer ").strip()
        if bearer:
            _verify_relay_token(x_bot_name, bearer)
            return ("bot", x_bot_name)

    # Dev bypass: no token configured
    if not token:
        return ("internal", None)

    raise HTTPException(status_code=401, detail="authentication required")
