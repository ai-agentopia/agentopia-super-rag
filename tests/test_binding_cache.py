"""BindingCache tests (#320, #KB-BINDING-V2).

Tests:
1. update() / remove() — direct cache manipulation
2. resolve() — O(1) cache lookup
3. resolve_with_fallback() V2 — cache-miss → control-plane (not K8s)
4. rebuild_from_control_plane() — V2 primary rebuild from bot-config-api
5. _fetch_from_control_plane_for_bot() — single-bot V2 fallback
6. rebuild_from_k8s() — DEPRECATED transitional fallback
7. Summary diagnostics
8. Bot identity contract: rebuild uses labels["agentopia/bot"] (slug), not agent-name
9. Env scoping: _list_bot_applications_from_k8s uses agentopia/env label selector
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from services.binding_cache import BindingCache, BotKnowledgeBinding


@pytest.fixture
def cache():
    return BindingCache()


# ── Direct cache manipulation ─────────────────────────────────────────────────


class TestCacheOperations:
    def test_update_adds_binding(self, cache):
        cache.update("my-bot", "client-a", ["docs", "code"])
        b = cache.resolve("my-bot")
        assert b is not None
        assert b.client_id == "client-a"
        assert b.knowledge_scopes == ["docs", "code"]

    def test_update_replaces_existing(self, cache):
        cache.update("my-bot", "client-a", ["docs"])
        cache.update("my-bot", "client-a", ["docs", "code"])
        b = cache.resolve("my-bot")
        assert b.knowledge_scopes == ["docs", "code"]

    def test_update_with_empty_scopes_removes(self, cache):
        cache.update("my-bot", "client-a", ["docs"])
        cache.update("my-bot", "client-a", [])
        assert cache.resolve("my-bot") is None

    def test_update_with_empty_client_removes(self, cache):
        cache.update("my-bot", "client-a", ["docs"])
        cache.update("my-bot", "", ["docs"])
        assert cache.resolve("my-bot") is None

    def test_remove_existing(self, cache):
        cache.update("my-bot", "client-a", ["docs"])
        cache.remove("my-bot")
        assert cache.resolve("my-bot") is None

    def test_remove_nonexistent_is_noop(self, cache):
        cache.remove("nonexistent-bot")  # should not raise
        assert cache.bot_count() == 0

    def test_bot_count(self, cache):
        assert cache.bot_count() == 0
        cache.update("bot-1", "c1", ["s1"])
        cache.update("bot-2", "c2", ["s2"])
        assert cache.bot_count() == 2
        cache.remove("bot-1")
        assert cache.bot_count() == 1


# ── Binding resolution ────────────────────────────────────────────────────────


class TestResolve:
    def test_resolve_hit(self, cache):
        cache.update("bot-a", "client-1", ["docs"])
        b = cache.resolve("bot-a")
        assert b is not None

    def test_resolve_miss_returns_none(self, cache):
        assert cache.resolve("unknown-bot") is None

    def test_resolved_scopes_format(self, cache):
        cache.update("bot-a", "client-1", ["docs", "code"])
        b = cache.resolve("bot-a")
        scopes = b.resolved_scopes()
        # Canonical scope identity is {client_id}/{scope_name}.
        # Physical Qdrant collection name uses '--' instead of '/' (#327 fix).
        assert "client-1/docs" in scopes
        assert "client-1/code" in scopes

    def test_resolve_with_fallback_hit(self, cache):
        """Cache hit → no control-plane call."""
        cache.update("bot-a", "client-1", ["docs"])
        with patch.object(cache, "_fetch_from_control_plane_for_bot") as mock_cp:
            b = cache.resolve_with_fallback("bot-a")
            assert b is not None
            mock_cp.assert_not_called()

    def test_resolve_with_fallback_miss_calls_control_plane(self, cache):
        """Cache miss → control-plane fallback (V2, not K8s)."""
        mock_binding = BotKnowledgeBinding(client_id="c1", knowledge_scopes=["docs"])
        with patch.object(cache, "_fetch_from_control_plane_for_bot", return_value=mock_binding) as mock_cp:
            b = cache.resolve_with_fallback("unknown-bot")
            assert b is not None
            assert b.client_id == "c1"
            mock_cp.assert_called_once_with("unknown-bot")

    def test_resolve_with_fallback_populates_cache(self, cache):
        """Control-plane fallback result is cached for future hits."""
        mock_binding = BotKnowledgeBinding(client_id="c1", knowledge_scopes=["docs"])
        with patch.object(cache, "_fetch_from_control_plane_for_bot", return_value=mock_binding):
            cache.resolve_with_fallback("new-bot")

        # Second call should hit cache, no control-plane call
        with patch.object(cache, "_fetch_from_control_plane_for_bot") as mock_cp:
            b = cache.resolve_with_fallback("new-bot")
            assert b is not None
            mock_cp.assert_not_called()

    def test_resolve_with_fallback_error_returns_none(self, cache):
        """Control-plane error → returns None gracefully."""
        with patch.object(cache, "_fetch_from_control_plane_for_bot", side_effect=Exception("cp down")):
            b = cache.resolve_with_fallback("unknown-bot")
            assert b is None


# ── Rebuild from K8s ──────────────────────────────────────────────────────────


class TestRebuildFromK8s:
    def test_rebuild_populates_from_k8s(self, cache):
        mock_bots = [
            {"bot_name": "bot-1", "client_id": "c1", "knowledge_scopes": '["docs"]'},
            {"bot_name": "bot-2", "client_id": "c2", "knowledge_scopes": '["code", "api"]'},
        ]
        with patch.object(cache, "_list_bot_applications_from_k8s", return_value=mock_bots):
            count = cache.rebuild_from_k8s()

        assert count == 2
        assert cache.resolve("bot-1") is not None
        b2 = cache.resolve("bot-2")
        assert b2.knowledge_scopes == ["code", "api"]

    def test_rebuild_replaces_existing(self, cache):
        """Full rebuild replaces old bindings — bots not in K8s are removed."""
        cache.update("old-bot", "c0", ["old"])
        mock_bots = [
            {"bot_name": "new-bot", "client_id": "c1", "knowledge_scopes": '["new"]'},
        ]
        with patch.object(cache, "_list_bot_applications_from_k8s", return_value=mock_bots):
            cache.rebuild_from_k8s()

        assert cache.resolve("old-bot") is None  # removed
        assert cache.resolve("new-bot") is not None  # added

    def test_rebuild_skips_bots_without_scopes(self, cache):
        mock_bots = [
            {"bot_name": "no-scope-bot", "client_id": "c1", "knowledge_scopes": ""},
        ]
        with patch.object(cache, "_list_bot_applications_from_k8s", return_value=mock_bots):
            count = cache.rebuild_from_k8s()
        assert count == 0

    def test_rebuild_handles_invalid_json(self, cache):
        mock_bots = [
            {"bot_name": "bad-bot", "client_id": "c1", "knowledge_scopes": "invalid-json"},
            {"bot_name": "good-bot", "client_id": "c2", "knowledge_scopes": '["docs"]'},
        ]
        with patch.object(cache, "_list_bot_applications_from_k8s", return_value=mock_bots):
            count = cache.rebuild_from_k8s()
        assert count == 1  # only good-bot counted

    def test_rebuild_graceful_when_k8s_unavailable(self, cache):
        with patch.object(cache, "_list_bot_applications_from_k8s", side_effect=Exception("no k8s")):
            count = cache.rebuild_from_k8s()
        assert count == 0
        assert cache.bot_count() == 0

    def test_rebuild_updates_last_reconcile(self, cache):
        before = cache._last_reconcile
        with patch.object(cache, "_list_bot_applications_from_k8s", return_value=[]):
            cache.rebuild_from_k8s()
        assert cache._last_reconcile > before


# ── Bot identity contract ─────────────────────────────────────────────────────


class TestBotIdentityContract:
    """Verify rebuild keys by labels["agentopia/bot"] (slug), not agent-name.

    bot-config-api stores:
      labels["agentopia/bot"] = bot_slug      (runtime identity, e.g. "devops-helper")
      annotations["agentopia/agent-name"] = display_name  (may differ, e.g. "DevOps Helper")

    knowledge-api must use the slug (label) as cache key — that's what auth uses.
    """

    def test_rebuild_keys_by_bot_label_not_agent_name(self, cache):
        """When bot slug != agent display name, rebuild stores the slug."""
        mock_apps = [
            {
                "metadata": {
                    "labels": {
                        "agentopia/bot": "devops-helper",     # runtime slug
                        "agentopia/managed-by": "bot-config-api",
                        "agentopia/env": "agentopia",
                    },
                    "annotations": {
                        "agentopia/agent-name": "DevOps Helper Bot",  # display name — different
                        "agentopia/client-id": "client-acme",
                        "agentopia/knowledge-scopes": '["runbooks","oncall"]',
                    },
                }
            }
        ]

        mock_k8s_client = MagicMock()
        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": mock_apps}
        mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

        with patch.object(cache, "_get_k8s_client", return_value=mock_k8s_client):
            count = cache.rebuild_from_k8s()

        assert count == 1
        # Must be keyed by slug, NOT the display name
        assert cache.resolve("devops-helper") is not None, \
            "Cache must be keyed by agentopia/bot label (slug)"
        assert cache.resolve("DevOps Helper Bot") is None, \
            "Cache must NOT be keyed by agentopia/agent-name annotation"

    def test_rebuild_skips_items_without_bot_label(self, cache):
        """Items missing labels["agentopia/bot"] are silently skipped."""
        mock_apps = [
            {
                "metadata": {
                    "labels": {},  # no agentopia/bot label
                    "annotations": {
                        "agentopia/agent-name": "Some Bot",
                        "agentopia/client-id": "c1",
                        "agentopia/knowledge-scopes": '["docs"]',
                    },
                }
            }
        ]

        mock_k8s_client = MagicMock()
        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": mock_apps}
        mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

        with patch.object(cache, "_get_k8s_client", return_value=mock_k8s_client):
            count = cache.rebuild_from_k8s()

        assert count == 0  # no valid bots found


class TestEnvScopedListSelector:
    """Verify _list_bot_applications_from_k8s uses env-scoped label selector."""

    def test_list_uses_env_scoped_selector(self, cache):
        """Label selector must include agentopia/env=<namespace> for env isolation."""
        mock_k8s_client = MagicMock()
        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": []}
        mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

        with patch.object(cache, "_get_k8s_client", return_value=mock_k8s_client):
            cache._list_bot_applications_from_k8s()

        call_kwargs = mock_custom_api.list_namespaced_custom_object.call_args[1]
        selector = call_kwargs.get("label_selector", "")
        assert "agentopia/managed-by=bot-config-api" in selector, \
            f"Selector must filter by managed-by: {selector}"
        assert "agentopia/env=" in selector, \
            f"Selector must scope to env namespace: {selector}"

    def test_list_selector_namespace_value_matches_env(self, cache):
        """agentopia/env= value must match K8S_NAMESPACE."""
        import os
        namespace = os.getenv("K8S_NAMESPACE", "agentopia")

        mock_k8s_client = MagicMock()
        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": []}
        mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

        with patch.object(cache, "_get_k8s_client", return_value=mock_k8s_client):
            cache._list_bot_applications_from_k8s()

        call_kwargs = mock_custom_api.list_namespaced_custom_object.call_args[1]
        selector = call_kwargs.get("label_selector", "")
        assert f"agentopia/env={namespace}" in selector, \
            f"Expected agentopia/env={namespace} in selector, got: {selector}"


# ── V2: Control-plane rebuild (#KB-BINDING-V2) ────────────────────────────────


class TestRebuildFromControlPlane:
    """rebuild_from_control_plane() — V2 primary startup/reconcile path."""

    def test_rebuilds_from_control_plane_response(self, cache):
        """Parses bot-config-api bulk binding response into cache."""
        import json
        from io import BytesIO
        from unittest.mock import patch, MagicMock

        payload = json.dumps({
            "bindings": [
                {"bot_name": "bot-alpha", "client_id": "acme", "knowledge_scopes": ["api-docs"]},
                {"bot_name": "bot-beta", "client_id": "acme", "knowledge_scopes": ["onboarding", "handbook"]},
            ]
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                count = cache.rebuild_from_control_plane()

        assert count == 2
        assert cache.resolve("bot-alpha") is not None
        b = cache.resolve("bot-beta")
        assert b.knowledge_scopes == ["onboarding", "handbook"]
        assert b.client_id == "acme"

    def test_replaces_existing_bindings(self, cache):
        """Full rebuild replaces stale cache."""
        cache.update("stale-bot", "old", ["old"])

        import json
        from unittest.mock import MagicMock

        payload = json.dumps({"bindings": [
            {"bot_name": "new-bot", "client_id": "c1", "knowledge_scopes": ["fresh"]},
        ]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                cache.rebuild_from_control_plane()

        assert cache.resolve("stale-bot") is None
        assert cache.resolve("new-bot") is not None

    def test_falls_back_to_k8s_when_url_not_set(self, cache):
        """Falls back to rebuild_from_k8s() when BOT_CONFIG_API_URL not set."""
        with patch("services.binding_cache._bot_config_api_url", return_value=""):
            with patch.object(cache, "rebuild_from_k8s", return_value=3) as mock_k8s:
                count = cache.rebuild_from_control_plane()
        assert count == 3
        mock_k8s.assert_called_once()

    def test_falls_back_to_k8s_on_http_error(self, cache):
        """Falls back to K8s if control-plane request fails."""
        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", side_effect=Exception("HTTP error")):
                with patch.object(cache, "rebuild_from_k8s", return_value=2) as mock_k8s:
                    count = cache.rebuild_from_control_plane()
        assert count == 2
        mock_k8s.assert_called_once()

    def test_skips_entries_missing_required_fields(self, cache):
        """Entries without bot_name, client_id, or scopes are skipped."""
        import json
        from unittest.mock import MagicMock

        payload = json.dumps({"bindings": [
            {"bot_name": "", "client_id": "c1", "knowledge_scopes": ["s"]},       # no bot_name
            {"bot_name": "bot-x", "client_id": "", "knowledge_scopes": ["s"]},     # no client_id
            {"bot_name": "bot-y", "client_id": "c1", "knowledge_scopes": []},      # empty scopes
            {"bot_name": "bot-z", "client_id": "c1", "knowledge_scopes": ["ok"]},  # valid
        ]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                count = cache.rebuild_from_control_plane()

        assert count == 1
        assert cache.resolve("bot-z") is not None

    def test_updates_last_reconcile(self, cache):
        before = cache._last_reconcile
        import json
        from unittest.mock import MagicMock

        payload = json.dumps({"bindings": []}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                cache.rebuild_from_control_plane()

        assert cache._last_reconcile > before


class TestFetchFromControlPlaneForBot:
    """_fetch_from_control_plane_for_bot() — V2 single-bot cache-miss fallback."""

    def test_returns_binding_for_enabled_bot(self, cache):
        import json
        from unittest.mock import MagicMock

        payload = json.dumps({
            "enabled": True,
            "client_id": "acme",
            "knowledge_scopes": ["api-docs", "onboarding"],
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                b = cache._fetch_from_control_plane_for_bot("bot-anna")

        assert b is not None
        assert b.client_id == "acme"
        assert "api-docs" in b.knowledge_scopes

    def test_returns_none_for_disabled_bot(self, cache):
        import json
        from unittest.mock import MagicMock

        payload = json.dumps({"enabled": False}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                b = cache._fetch_from_control_plane_for_bot("bot-unbound")

        assert b is None

    def test_falls_back_to_k8s_when_url_not_set(self, cache):
        """No BOT_CONFIG_API_URL → K8s fallback."""
        mock_binding = BotKnowledgeBinding(client_id="c1", knowledge_scopes=["s"])
        with patch("services.binding_cache._bot_config_api_url", return_value=""):
            with patch.object(cache, "_fetch_from_k8s_for_bot", return_value=mock_binding) as mock_k8s:
                b = cache._fetch_from_control_plane_for_bot("bot-x")
        assert b is not None
        mock_k8s.assert_called_once_with("bot-x")

    def test_falls_back_to_k8s_on_http_error(self, cache):
        """HTTP failure → K8s transitional fallback."""
        mock_binding = BotKnowledgeBinding(client_id="c1", knowledge_scopes=["s"])
        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
                with patch.object(cache, "_fetch_from_k8s_for_bot", return_value=mock_binding) as mock_k8s:
                    b = cache._fetch_from_control_plane_for_bot("bot-x")
        assert b is not None
        mock_k8s.assert_called_once_with("bot-x")

    def test_returns_none_when_client_id_missing(self, cache):
        import json
        from unittest.mock import MagicMock

        payload = json.dumps({"enabled": True, "client_id": "", "knowledge_scopes": ["s"]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("services.binding_cache._bot_config_api_url", return_value="http://bot-config-api"):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                b = cache._fetch_from_control_plane_for_bot("bot-bad")

        assert b is None


# ── Summary diagnostics ───────────────────────────────────────────────────────


class TestSummary:
    def test_summary_structure(self, cache):
        cache.update("bot-a", "c1", ["docs"])
        s = cache.summary()
        assert s["bot_count"] == 1
        assert "last_reconcile" in s
        assert "reconcile_interval_secs" in s
        assert "bot-a" in s["bots"]
