"""E2E integration tests for agentopia-super-rag knowledge retrieval contract.

Tests the real service boundary using:
- real Qdrant backend (via compose.e2e.yaml)
- real Postgres document store
- deterministic local embedding stub (no OpenRouter key required)
- real scope isolation enforcement

What this proves:
- service boots healthy with real backends
- documents are ingested into Qdrant and are searchable
- scoped search returns results only from the correct scope
- auth is enforced at the API boundary

What is deferred (see agentopia-super-rag#50):
- bot bearer token path (requires K8s binding cache)
- gateway → bot-config-api → knowledge-api cross-service chain

How to run:
  # Start the E2E stack
  podman compose -f compose.yaml -f compose.e2e.yaml --env-file .env.e2e up -d --build

  # Run E2E tests
  python -m pytest tests/e2e/ -m e2e -v

  # Tear down
  podman compose -f compose.yaml -f compose.e2e.yaml down -v
"""

import os
import time

import httpx
import pytest

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = os.getenv("KNOWLEDGE_API_E2E_URL", "http://localhost:8002")
INTERNAL_TOKEN = os.getenv("KNOWLEDGE_API_INTERNAL_TOKEN", "local-dev-token")

SEED_SCOPE = "e2e-org/integration-scope"
OTHER_SCOPE = "e2e-org/other-scope"

SEED_SOURCE = "e2e-seed.md"
SEED_CONTENT = b"""# Agentopia Knowledge Retrieval

Agentopia is a multi-bot Telegram platform.
Each bot has a bounded set of knowledge scopes it can query.
The service enforces scope isolation at the API level.
A bot cannot retrieve content outside its subscribed scopes regardless of query content.

## Retrieval Architecture

Knowledge is stored in Qdrant as dense vector embeddings.
Documents are ingested, chunked, embedded, and upserted per scope.
Scopes are identified by client_id/scope_name pairs.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scope_url(scope: str) -> str:
    """Convert scope 'org/name' to URL segment 'org--name'."""
    return scope.replace("/", "--")


def _internal_headers(token: str = INTERNAL_TOKEN) -> dict:
    return {"X-Internal-Token": token}


def _wait_for_service(timeout: int = 30) -> None:
    """Block until the service responds to /health or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(
        f"knowledge-api at {BASE_URL} did not become healthy within {timeout}s. "
        "Is the E2E stack running? "
        "Run: podman compose -f compose.yaml -f compose.e2e.yaml --env-file .env.e2e up -d --build"
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def wait_for_service():
    """Wait for the service to be healthy before any test in this module runs."""
    _wait_for_service(timeout=30)


@pytest.mark.e2e
class TestHealthAndReadiness:
    """Service must be healthy with real backends before any retrieval test."""

    def test_liveness(self):
        resp = httpx.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "knowledge-api"

    def test_qdrant_backend_is_real(self):
        """Internal health must show real Qdrant, not in-memory fallback."""
        resp = httpx.get(f"{BASE_URL}/internal/health", headers=_internal_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["qdrant"] == "ok", (
            f"Expected real Qdrant — got {body['qdrant']!r}. "
            "Check that compose.e2e.yaml stack is running."
        )

    def test_internal_health_requires_token(self):
        resp = httpx.get(f"{BASE_URL}/internal/health")
        assert resp.status_code in (401, 403)


@pytest.mark.e2e
class TestAuthEnforcement:
    """API must enforce auth at all read endpoints."""

    def test_missing_token_on_scopes_returns_401(self):
        resp = httpx.get(f"{BASE_URL}/api/v1/knowledge/scopes")
        assert resp.status_code in (401, 403)

    def test_wrong_token_on_scopes_returns_4xx(self):
        resp = httpx.get(
            f"{BASE_URL}/api/v1/knowledge/scopes",
            headers=_internal_headers("not-the-right-token"),
        )
        assert resp.status_code in (401, 403)

    def test_missing_token_on_search_returns_401(self):
        resp = httpx.get(
            f"{BASE_URL}/api/v1/knowledge/search",
            params={"query": "test", "scopes": SEED_SCOPE},
        )
        assert resp.status_code in (401, 403)

    def test_valid_token_on_scopes_returns_200(self):
        resp = httpx.get(
            f"{BASE_URL}/api/v1/knowledge/scopes",
            headers=_internal_headers(),
        )
        assert resp.status_code == 200


@pytest.mark.e2e
class TestIngestAndRetrieval:
    """Full ingest → real Qdrant embedding → search round-trip.

    Seed fixture ingests one document before each test, cleans up after.
    The embedding stub produces deterministic vectors — same text, same vector.
    """

    @pytest.fixture(autouse=True)
    def seed_and_cleanup(self):
        scope_url = _scope_url(SEED_SCOPE)

        resp = httpx.post(
            f"{BASE_URL}/api/v1/knowledge/{scope_url}/ingest",
            headers=_internal_headers(),
            files={"file": (SEED_SOURCE, SEED_CONTENT, "text/markdown")},
            timeout=30,
        )
        assert resp.status_code == 201, (
            f"Seed ingest failed ({resp.status_code}): {resp.text}"
        )
        body = resp.json()
        assert body["chunks_created"] > 0, "Seed document produced no chunks"

        yield

        httpx.delete(
            f"{BASE_URL}/api/v1/knowledge/{scope_url}/documents/{SEED_SOURCE}",
            headers=_internal_headers(),
            timeout=10,
        )

    def test_scope_appears_in_scope_list(self):
        resp = httpx.get(
            f"{BASE_URL}/api/v1/knowledge/scopes",
            headers=_internal_headers(),
        )
        assert resp.status_code == 200
        scope_names = [s["name"] for s in resp.json()["scopes"]]
        assert SEED_SCOPE in scope_names, (
            f"{SEED_SCOPE} not in scope list after ingest: {scope_names}"
        )

    def test_search_in_seeded_scope_returns_results(self):
        """Search within the seeded scope must return content from Qdrant."""
        resp = httpx.get(
            f"{BASE_URL}/api/v1/knowledge/search",
            params={"query": "Agentopia bot platform knowledge scopes", "scopes": SEED_SCOPE},
            headers=_internal_headers(),
            timeout=15,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] > 0, (
            "Expected at least one result after ingest. "
            "Check embedding-stub logs — embedding may have failed."
        )

    def test_search_result_source_matches_seed(self):
        """Search results must cite the seeded document."""
        resp = httpx.get(
            f"{BASE_URL}/api/v1/knowledge/search",
            params={"query": "scope isolation API level", "scopes": SEED_SCOPE},
            headers=_internal_headers(),
            timeout=15,
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        sources = [r["citation"]["source"] for r in results]
        assert SEED_SOURCE in sources, (
            f"Expected {SEED_SOURCE} in citations, got: {sources}"
        )

    def test_scope_isolation_other_scope_returns_empty(self):
        """Critical: searching a different scope must return no results.

        SEED_SCOPE has documents. OTHER_SCOPE has none.
        Any result here is a scope isolation failure.
        """
        resp = httpx.get(
            f"{BASE_URL}/api/v1/knowledge/search",
            params={"query": "Agentopia bot platform knowledge scopes", "scopes": OTHER_SCOPE},
            headers=_internal_headers(),
            timeout=15,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0, (
            f"SCOPE ISOLATION FAILURE: got {body['count']} results from {OTHER_SCOPE} "
            f"when only {SEED_SCOPE} was seeded. Results: {body['results']}"
        )

    def test_search_result_has_required_citation_fields(self):
        """Search results must include all citation fields per the API contract."""
        resp = httpx.get(
            f"{BASE_URL}/api/v1/knowledge/search",
            params={"query": "dense vector embeddings Qdrant", "scopes": SEED_SCOPE},
            headers=_internal_headers(),
            timeout=15,
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0, "Need at least one result to check citation fields"
        for result in results:
            assert "text" in result
            assert "score" in result
            cit = result.get("citation", {})
            assert "source" in cit
            assert "chunk_index" in cit
            assert "score" in cit
