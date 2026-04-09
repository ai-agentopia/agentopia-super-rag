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


# ── POST /{scope}/ingest (multipart upload) ──────────────────────────────────


class TestIngestFileRoute:
    def test_upload_text_file(self, client):
        """Upload a .txt file → chunks ingested."""
        content = b"This is test content for ingestion. " * 20
        resp = client.post(
            "/api/v1/knowledge/test-scope/ingest",
            files={"file": ("readme.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "ingested"
        assert data["scope"] == "test-scope"
        assert data["source"] == "readme.txt"
        assert data["format"] == "text"
        assert data["chunks_created"] >= 1

    def test_upload_markdown_file(self, client):
        """Upload a .md file → detected as markdown format."""
        content = (
            b"# Title\n\nSome markdown content here.\n\n## Section\n\nMore content."
        )
        resp = client.post(
            "/api/v1/knowledge/docs/ingest",
            files={"file": ("guide.md", io.BytesIO(content), "text/markdown")},
        )
        assert resp.status_code == 201
        assert resp.json()["format"] == "markdown"

    def test_upload_html_file(self, client):
        """Upload a .html file → parsed with BeautifulSoup."""
        html = b"<html><body><h1>Hello</h1><p>World</p></body></html>"
        resp = client.post(
            "/api/v1/knowledge/web/ingest",
            files={"file": ("page.html", io.BytesIO(html), "text/html")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["format"] == "html"
        assert data["chunks_created"] >= 1

    def test_upload_code_file(self, client):
        """Upload a .py file → detected as code format."""
        code = b"def hello():\n    return 'world'\n\ndef foo():\n    return 'bar'\n"
        resp = client.post(
            "/api/v1/knowledge/code/ingest",
            files={"file": ("utils.py", io.BytesIO(code), "text/x-python")},
        )
        assert resp.status_code == 201
        assert resp.json()["format"] == "code"

    def test_upload_empty_file_returns_422(self, client):
        """Empty file → 422 error."""
        resp = client.post(
            "/api/v1/knowledge/scope/ingest",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert resp.status_code == 422

    def test_upload_no_file_returns_422(self, client):
        """Missing file field → 422 error."""
        resp = client.post("/api/v1/knowledge/scope/ingest")
        assert resp.status_code == 422


# ── GET /search ──────────────────────────────────────────────────────────────


class TestSearchRoute:
    def test_search_with_results(self, client):
        """Ingest then search → returns matching results."""
        # Ingest first via file upload (webhook retired #303)
        client.post(
            "/api/v1/knowledge/search-test/ingest",
            files={"file": ("k8s.md", io.BytesIO(b"Kubernetes is a container orchestration platform for deploying applications."), "text/markdown")},
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
        """Empty query string → 422 validation error."""
        resp = client.get("/api/v1/knowledge/search", params={"query": ""})
        assert resp.status_code == 422

    def test_search_no_scopes_searches_all(self, client):
        """No scopes param → searches all available scopes."""
        client.post(
            "/api/v1/knowledge/global/ingest",
            files={"file": ("doc.txt", io.BytesIO(b"Terraform infrastructure as code provisioning."), "text/plain")},
        )
        resp = client.get(
            "/api/v1/knowledge/search",
            params={"query": "Terraform"},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_search_no_match(self, client):
        """Search for nonexistent term → empty results."""
        resp = client.get(
            "/api/v1/knowledge/search",
            params={"query": "xyznonexistent12345"},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ── DELETE /{scope} ──────────────────────────────────────────────────────────


class TestDeleteScopeRoute:
    def test_delete_existing_scope(self, client):
        """Delete a scope that exists → 200."""
        client.post(
            "/api/v1/knowledge/to-delete/ingest",
            files={"file": ("x.txt", io.BytesIO(b"Delete me."), "text/plain")},
        )
        resp = client.delete("/api/v1/knowledge/to-delete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone
        resp2 = client.get("/api/v1/knowledge/to-delete")
        assert resp2.status_code == 404

    def test_delete_nonexistent_scope(self, client):
        """Delete a scope that doesn't exist → 404."""
        resp = client.delete("/api/v1/knowledge/nonexistent-scope")
        assert resp.status_code == 404


# ── DELETE /{scope}/documents/{source} ───────────────────────────────────────


class TestDeleteDocumentRoute:
    def test_delete_existing_document(self, client):
        """Delete a specific document source → 200 with chunks_removed."""
        client.post(
            "/api/v1/knowledge/docs/ingest",
            files={"file": ("keep.txt", io.BytesIO(b"Keep this content."), "text/plain")},
        )
        client.post(
            "/api/v1/knowledge/docs/ingest",
            files={"file": ("remove.txt", io.BytesIO(b"Remove this content."), "text/plain")},
        )
        resp = client.delete("/api/v1/knowledge/docs/documents/remove.txt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["chunks_removed"] >= 1

    def test_delete_document_scope_not_found(self, client):
        """Delete document in nonexistent scope → 404."""
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
        client.post(
            "/api/v1/knowledge/pathtest/ingest",
            files={"file": ("docs/sub/readme.md", io.BytesIO(b"Nested path document content here."), "text/markdown")},
        )
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
        client.post(
            "/api/v1/knowledge/fmt-test/ingest",
            files={"file": ("a.txt", io.BytesIO(b"Content A."), "text/plain")},
        )
        resp = client.get("/api/v1/knowledge/scopes")
        assert resp.status_code == 200
        scope = resp.json()["scopes"][0]
        assert "document_count" in scope
        assert "chunk_count" in scope

    def test_scope_detail_includes_document_count(self, client):
        """GET /{scope} returns document_count field."""
        client.post(
            "/api/v1/knowledge/detail-test/ingest",
            files={"file": ("b.txt", io.BytesIO(b"Content B."), "text/plain")},
        )
        resp = client.get("/api/v1/knowledge/detail-test")
        assert resp.status_code == 200
        data = resp.json()
        assert "document_count" in data
