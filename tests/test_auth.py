"""Auth guard tests for knowledge-api (#320).

Tests:
1. Internal proxy auth: valid / invalid / missing token
2. Bot bearer auth: valid / invalid / missing headers
3. Dual-path read auth: internal wins over bot
4. Write routes enforce internal-only auth
5. Dev bypass: no token configured → all allowed
6. Relay verification contract: reads agentopia-gateway-env-{bot} / AGENTOPIA_RELAY_TOKEN
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_app(internal_token: str = "test-token"):
    """Create a fresh app instance with given token config."""
    os.environ["KNOWLEDGE_API_INTERNAL_TOKEN"] = internal_token
    # Patch out K8s for service creation
    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith(("services.", "routers.", "auth.", "main")):
            sys.modules.pop(mod, None)

    import importlib
    import main as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


@pytest.fixture
def client():
    prev = os.environ.get("KNOWLEDGE_API_INTERNAL_TOKEN")
    os.environ["KNOWLEDGE_API_INTERNAL_TOKEN"] = "test-token-abc"
    with patch("routers.knowledge.get_knowledge_service") as mock_svc:
        mock_svc.return_value = MagicMock()
        mock_svc.return_value.list_scopes.return_value = []
        mock_svc.return_value.search.return_value = []
        from fastapi.testclient import TestClient
        import main
        import importlib
        importlib.reload(main)
        yield TestClient(main.app)
    if prev is None:
        os.environ.pop("KNOWLEDGE_API_INTERNAL_TOKEN", None)
    else:
        os.environ["KNOWLEDGE_API_INTERNAL_TOKEN"] = prev


# ── Internal proxy auth ───────────────────────────────────────────────────────


class TestInternalAuth:
    def test_write_requires_internal_token(self, client):
        resp = client.post("/api/v1/knowledge/test-scope/reindex")
        assert resp.status_code == 401

    def test_write_accepts_valid_internal_token(self, client):
        with patch("routers.knowledge.get_knowledge_service") as mock_svc:
            svc = MagicMock()
            svc.reindex.return_value = {"status": "ok", "scope": "test-scope"}
            mock_svc.return_value = svc
            resp = client.post(
                "/api/v1/knowledge/test-scope/reindex",
                headers={"X-Internal-Token": "test-token-abc"},
            )
            # 200 or 404 (scope not found) — either proves auth passed
            assert resp.status_code in (200, 404)

    def test_write_rejects_invalid_internal_token(self, client):
        resp = client.post(
            "/api/v1/knowledge/test-scope/reindex",
            headers={"X-Internal-Token": "wrong-token"},
        )
        assert resp.status_code == 401

    def test_write_rejects_missing_token(self, client):
        resp = client.post("/api/v1/knowledge/test-scope/reindex")
        assert resp.status_code == 401


# ── Read dual-path auth ───────────────────────────────────────────────────────


class TestReadAuth:
    def test_read_accepts_internal_token(self, client):
        resp = client.get(
            "/api/v1/knowledge/scopes",
            headers={"X-Internal-Token": "test-token-abc"},
        )
        assert resp.status_code == 200

    def test_read_rejects_invalid_internal_token(self, client):
        resp = client.get(
            "/api/v1/knowledge/scopes",
            headers={"X-Internal-Token": "wrong"},
        )
        assert resp.status_code == 401

    def test_read_rejects_unauthenticated(self, client):
        resp = client.get("/api/v1/knowledge/scopes")
        assert resp.status_code == 401

    def test_read_accepts_bot_bearer(self, client):
        with patch("auth.guards._verify_relay_token"), \
             patch("routers.knowledge._resolve_bot_scopes", return_value=[]):
            resp = client.get(
                "/api/v1/knowledge/scopes",
                headers={
                    "Authorization": "Bearer bot-relay-token",
                    "X-Bot-Name": "test-bot",
                },
            )
            assert resp.status_code == 200

    def test_read_rejects_bot_bearer_without_bot_name(self, client):
        resp = client.get(
            "/api/v1/knowledge/scopes",
            headers={"Authorization": "Bearer some-token"},
        )
        assert resp.status_code == 401


# ── Dev bypass ────────────────────────────────────────────────────────────────


class TestDevBypass:
    def test_no_token_configured_allows_all(self):
        """When KNOWLEDGE_API_INTERNAL_TOKEN is not set, all requests pass."""
        saved = os.environ.pop("KNOWLEDGE_API_INTERNAL_TOKEN", None)
        try:
            import sys
            for mod in list(sys.modules.keys()):
                if mod.startswith(("auth.", "routers.", "main")):
                    sys.modules.pop(mod, None)

            with patch("services.knowledge.get_knowledge_service") as mock_svc:
                mock_svc.return_value = MagicMock()
                mock_svc.return_value.list_scopes.return_value = []
                import importlib
                import main
                importlib.reload(main)
                from fastapi.testclient import TestClient
                c = TestClient(main.app)
                resp = c.get("/api/v1/knowledge/scopes")
                assert resp.status_code == 200
        finally:
            if saved is not None:
                os.environ["KNOWLEDGE_API_INTERNAL_TOKEN"] = saved
            # Purge dirty modules so subsequent tests get fresh imports
            import sys as _sys
            for mod in list(_sys.modules.keys()):
                if mod.startswith(("auth.", "routers.", "main")):
                    _sys.modules.pop(mod, None)


# ── Relay verification contract ───────────────────────────────────────────────


class TestRelayVerificationContract:
    """Prove _verify_relay_token reads the correct secret and key.

    Contract (must match bot-config-api K8sService.get_relay_token):
      Secret: agentopia-gateway-env-{bot_name}
      Key:    AGENTOPIA_RELAY_TOKEN
    """

    def test_relay_verification_reads_gateway_env_secret(self):
        """_verify_relay_token reads agentopia-gateway-env-{bot}, not agentopia-bot-token-{bot}."""
        import os
        import base64
        from unittest.mock import MagicMock, patch
        from auth.guards import _verify_relay_token

        os.environ["KUBERNETES_SERVICE_HOST"] = "fake-host"
        try:
            relay_token = "my-relay-token-abc"
            encoded = base64.b64encode(relay_token.encode()).decode()

            mock_secret = MagicMock()
            mock_secret.data = {"AGENTOPIA_RELAY_TOKEN": encoded}

            mock_v1 = MagicMock()
            mock_v1.read_namespaced_secret.return_value = mock_secret

            mock_k8s_client = MagicMock()
            mock_k8s_client.CoreV1Api.return_value = mock_v1

            with patch("auth.guards.k8s_client", mock_k8s_client, create=True), \
                 patch("auth.guards.k8s_config", MagicMock(), create=True):
                from kubernetes import client as real_client, config as real_config
                with patch.object(real_config, "load_incluster_config"), \
                     patch.object(real_client, "CoreV1Api", return_value=mock_v1):
                    _verify_relay_token("my-bot", relay_token)

            # Verify: read_namespaced_secret called with gateway-env secret name (not bot-token)
            call_args = mock_v1.read_namespaced_secret.call_args
            secret_name_used = call_args[0][0] if call_args[0] else call_args[1].get("name", "")
            assert "gateway-env" in secret_name_used, (
                f"Expected agentopia-gateway-env-* secret, got: {secret_name_used}"
            )
            assert "bot-token" not in secret_name_used, (
                f"Must NOT read agentopia-bot-token-*, got: {secret_name_used}"
            )
        finally:
            os.environ.pop("KUBERNETES_SERVICE_HOST", None)

    def test_relay_verification_uses_relay_token_key(self):
        """_verify_relay_token reads AGENTOPIA_RELAY_TOKEN key, not 'token'."""
        import os
        import base64
        from unittest.mock import MagicMock, patch
        from auth.guards import _verify_relay_token

        os.environ["KUBERNETES_SERVICE_HOST"] = "fake-host"
        try:
            relay_token = "relay-secret-xyz"
            encoded = base64.b64encode(relay_token.encode()).decode()

            # Secret has AGENTOPIA_RELAY_TOKEN but NOT 'token'
            mock_secret = MagicMock()
            mock_secret.data = {"AGENTOPIA_RELAY_TOKEN": encoded}  # no 'token' key

            mock_v1 = MagicMock()
            mock_v1.read_namespaced_secret.return_value = mock_secret

            with patch("kubernetes.client.CoreV1Api", return_value=mock_v1), \
                 patch("kubernetes.config.load_incluster_config"):
                # Should succeed using AGENTOPIA_RELAY_TOKEN key
                _verify_relay_token("my-bot", relay_token)

        finally:
            os.environ.pop("KUBERNETES_SERVICE_HOST", None)

    def test_relay_verification_rejects_if_only_telegram_token_key_present(self):
        """If secret has 'token' key but not AGENTOPIA_RELAY_TOKEN, auth must fail."""
        import os
        import base64
        from fastapi import HTTPException
        from unittest.mock import MagicMock, patch
        from auth.guards import _verify_relay_token

        os.environ["KUBERNETES_SERVICE_HOST"] = "fake-host"
        try:
            relay_token = "some-token"
            encoded = base64.b64encode(relay_token.encode()).decode()

            # Secret has old 'token' key only — wrong contract
            mock_secret = MagicMock()
            mock_secret.data = {"token": encoded}  # old agentopia-bot-token contract

            mock_v1 = MagicMock()
            mock_v1.read_namespaced_secret.return_value = mock_secret

            with patch("kubernetes.client.CoreV1Api", return_value=mock_v1), \
                 patch("kubernetes.config.load_incluster_config"):
                with pytest.raises(HTTPException) as exc_info:
                    _verify_relay_token("my-bot", relay_token)
                assert exc_info.value.status_code == 401
        finally:
            os.environ.pop("KUBERNETES_SERVICE_HOST", None)


# ── Internal endpoints ────────────────────────────────────────────────────────


class TestInternalEndpoints:
    def test_internal_health_requires_token(self, client):
        resp = client.get("/internal/health")
        assert resp.status_code == 401

    def test_internal_health_returns_status(self, client):
        with patch("routers.internal.get_knowledge_service", create=True) as mock_svc:
            mock_svc.return_value = MagicMock()
            mock_svc.return_value._qdrant = None
            resp = client.get(
                "/internal/health",
                headers={"X-Internal-Token": "test-token-abc"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "binding_cache" in data
            assert data["service"] == "knowledge-api"

    def test_binding_sync_requires_token(self, client):
        resp = client.post(
            "/internal/binding-sync",
            json={"bot_name": "test-bot", "client_id": "c1", "knowledge_scopes": ["docs"]},
        )
        assert resp.status_code == 401

    def test_binding_sync_updates_cache(self, client):
        resp = client.post(
            "/internal/binding-sync",
            json={"bot_name": "test-bot", "client_id": "client1", "knowledge_scopes": ["docs"]},
            headers={"X-Internal-Token": "test-token-abc"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "synced"
        assert data["bot_name"] == "test-bot"

    def test_binding_remove_updates_cache(self, client):
        # First sync
        client.post(
            "/internal/binding-sync",
            json={"bot_name": "del-bot", "client_id": "c1", "knowledge_scopes": ["s"]},
            headers={"X-Internal-Token": "test-token-abc"},
        )
        # Then remove
        resp = client.delete(
            "/internal/binding-sync/del-bot",
            headers={"X-Internal-Token": "test-token-abc"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"
