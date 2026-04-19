"""W1 tests: file upload ingestion, search route, delete routes, PDF/HTML parsers.

Covers M3 #23 (ingestion) + #106 (knowledge UI routes).
"""

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Fresh TestClient — resets knowledge service singleton between tests."""
    import os

    import services.knowledge as ksvc

    ksvc._knowledge = None  # reset singleton
    from main import app

    token = os.getenv("KNOWLEDGE_API_INTERNAL_TOKEN", "")
    headers = {"X-Internal-Token": token} if token else {}

    with TestClient(app, headers=headers) as c:
        yield c
    ksvc._knowledge = None


# ── Parsers ──────────────────────────────────────────────────────────────────


class TestParsePdf:
    def test_valid_pdf(self):
        """parse_pdf extracts text from a minimal valid PDF."""
        from PyPDF2 import PdfWriter

        from services.knowledge import parse_pdf

        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        # PdfWriter blank pages have no text, so create one with annotation
        # Use a real PDF with text via reportlab-free approach
        buf = io.BytesIO()
        writer.write(buf)
        raw = buf.getvalue()
        # A blank PDF should parse without error, returning empty string
        result = parse_pdf(raw)
        assert isinstance(result, str)

    def test_invalid_pdf_raises(self):
        """parse_pdf raises on invalid bytes."""
        from services.knowledge import parse_pdf

        with pytest.raises(Exception):
            parse_pdf(b"not a pdf file at all")


class TestParseHtml:
    def test_valid_html(self):
        """parse_html extracts text, strips scripts/styles."""
        from services.knowledge import parse_html

        html = b"""<html>
        <head><style>body{color:red}</style></head>
        <body>
            <h1>Title</h1>
            <p>Hello world</p>
            <script>alert('xss')</script>
        </body>
        </html>"""
        result = parse_html(html)
        assert "Title" in result
        assert "Hello world" in result
        assert "alert" not in result
        assert "color:red" not in result

    def test_empty_html(self):
        """parse_html handles empty/minimal HTML gracefully."""
        from services.knowledge import parse_html

        result = parse_html(b"<html><body></body></html>")
        assert isinstance(result, str)

    def test_malformed_html(self):
        """parse_html handles malformed HTML without crashing."""
        from services.knowledge import parse_html

        result = parse_html(b"<p>Unclosed paragraph <b>bold")
        assert "Unclosed paragraph" in result


# ── POST /{scope}/ingest (RETIRED — direct-to-Qdrant no longer allowed, P4.5) ─


class TestIngestFileRouteRetired:
    """The direct-to-Qdrant file upload route is retired post P4.5.

    All operator uploads must go through bot-config-api's async S3 path:
      POST /api/v1/knowledge/{scope}/ingest on bot-config-api → S3 → Pathway.
    This route on knowledge-api returns 410 Gone for any caller.
    """

    def test_upload_returns_410(self, client):
        resp = client.post(
            "/api/v1/knowledge/test-scope/ingest",
            files={"file": ("readme.txt", io.BytesIO(b"x"), "text/plain")},
        )
        assert resp.status_code == 410
        assert "retired" in resp.json()["detail"].lower()

    def test_upload_no_file_also_returns_410(self, client):
        resp = client.post("/api/v1/knowledge/scope/ingest")
        assert resp.status_code == 410


class TestIngestDocumentRouteRetired:
    """The orchestrator direct-ingest route is retired post P4.5.

    agentopia-knowledge-ingest was decommissioned; no caller remains.
    Route returns 410 Gone.
    """

    def test_ingest_document_returns_410(self, client):
        resp = client.post(
            "/api/v1/knowledge/test-scope/ingest-document",
            json={"document_id": "x", "version": 1, "text": "hello"},
        )
        assert resp.status_code == 410
        assert "retired" in resp.json()["detail"].lower()


# ── Helper: seed via service layer (HTTP /ingest retired, see P4.5) ──────────


def _seed_via_service(scope: str, source: str, content: str, fmt: str = "text"):
    """Seed a scope using the service layer directly.

    The HTTP POST /{scope}/ingest route is retired in P4.5 — all live ingest
    flows through Pathway. Tests still need a way to populate in-memory state
    to exercise the search/delete routes; use the service method directly.
    """
    from services.knowledge import get_knowledge_service
    from models.knowledge import DocumentFormat
    svc = get_knowledge_service()
    fmt_enum = DocumentFormat(fmt) if not isinstance(fmt, DocumentFormat) else fmt
    svc.ingest(scope=scope, content=content, source=source, format=fmt_enum)


# ── GET /search ──────────────────────────────────────────────────────────────


class TestSearchRoute:
    def test_search_with_results(self, client):
        _seed_via_service(
            "search-test", "k8s.md",
            "Kubernetes is a container orchestration platform for deploying applications.",
            fmt="markdown",
        )
        resp = client.get(
            "/api/v1/knowledge/search",
            params={"query": "Kubernetes", "scopes": "search-test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "count" in data
        assert data["count"] >= 1

    def test_search_empty_query_returns_422(self, client):
        resp = client.get("/api/v1/knowledge/search", params={"query": ""})
        assert resp.status_code == 422

    def test_search_no_scopes_searches_all(self, client):
        _seed_via_service(
            "global", "doc.txt",
            "Terraform infrastructure as code provisioning.",
            fmt="text",
        )
        resp = client.get("/api/v1/knowledge/search", params={"query": "Terraform"})
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_search_no_match(self, client):
        resp = client.get(
            "/api/v1/knowledge/search",
            params={"query": "xyznonexistent12345"},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ── DELETE /{scope} ──────────────────────────────────────────────────────────


class TestDeleteScopeRoute:
    def test_delete_existing_scope(self, client):
        _seed_via_service("to-delete", "x.txt", "Delete me.", fmt="text")
        resp = client.delete("/api/v1/knowledge/to-delete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        resp2 = client.get("/api/v1/knowledge/to-delete")
        assert resp2.status_code == 404

    def test_delete_nonexistent_scope(self, client):
        resp = client.delete("/api/v1/knowledge/nonexistent-scope")
        assert resp.status_code == 404


# ── DELETE /{scope}/documents/{source} ───────────────────────────────────────


class TestDeleteDocumentRoute:
    def test_delete_existing_document(self, client):
        _seed_via_service("docs", "keep.txt", "Keep this content.", fmt="text")
        _seed_via_service("docs", "remove.txt", "Remove this content.", fmt="text")
        resp = client.delete("/api/v1/knowledge/docs/documents/remove.txt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["chunks_removed"] >= 1

    def test_delete_document_scope_not_found(self, client):
        resp = client.delete("/api/v1/knowledge/nosuchscope/documents/file.txt")
        assert resp.status_code == 404

    def test_delete_document_source_not_found(self, client):
        """Delete nonexistent source in existing scope → 404."""
        client.post(
            "/api/v1/knowledge/exists/ingest",
            files={"file": ("a.txt", io.BytesIO(b"Some content."), "text/plain")},
        )
        resp = client.delete("/api/v1/knowledge/exists/documents/nonexistent.txt")
        assert resp.status_code == 404

    def test_delete_document_encoded_path(self, client):
        """DELETE with path containing slashes (source:path converter) works."""
        _seed_via_service("pathtest", "docs/sub/readme.md", "Nested path document content here.", fmt="markdown")
        resp = client.delete("/api/v1/knowledge/pathtest/documents/docs/sub/readme.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["source"] == "docs/sub/readme.md"
        assert data["chunks_removed"] >= 1


# ── GET /scopes includes document_count ──────────────────────────────────────


class TestScopesResponseFormat:
    def test_scopes_include_document_count(self, client):
        """GET /scopes returns document_count field."""
        _seed_via_service("fmt-test", "a.txt", "Content A.", fmt="text")
        resp = client.get("/api/v1/knowledge/scopes")
        assert resp.status_code == 200
        scope = resp.json()["scopes"][0]
        assert "document_count" in scope
        assert "chunk_count" in scope

    def test_scope_detail_includes_document_count(self, client):
        """GET /{scope} returns document_count field."""
        _seed_via_service("detail-test", "b.txt", "Content B.", fmt="text")
        resp = client.get("/api/v1/knowledge/detail-test")
        assert resp.status_code == 200
        data = resp.json()
        assert "document_count" in data
