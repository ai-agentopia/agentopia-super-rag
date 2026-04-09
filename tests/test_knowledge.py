"""Tests for knowledge base / RAG (M3 #23, #24, #25, #100, #101, #102, #106).

Covers: document chunking, ingestion, scoped search, citations,
knowledge scope management, code-aware chunking, dedup hashing.
"""

import os

from models.knowledge import (
    ChunkingStrategy,
    Citation,
    DocumentFormat,
    DocumentMetadata,
    IngestConfig,
    RepoIndexConfig,
    SearchResult,
)
from services.knowledge import (
    KnowledgeService,
    build_citations,
    chunk_document,
    compute_chunk_hash,
    format_citations,
)


# ── Models ───────────────────────────────────────────────────────────────


class TestKnowledgeModels:
    def test_ingest_config_defaults(self):
        cfg = IngestConfig()
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64

    def test_document_metadata(self):
        meta = DocumentMetadata(
            source="deploy.py",
            format=DocumentFormat.CODE,
            scope="project-alpha",
            language="python",
        )
        assert meta.language == "python"

    def test_citation(self):
        c = Citation(source="readme.md", section="Getting Started", score=0.85)
        assert c.source == "readme.md"

    def test_search_result(self):
        sr = SearchResult(
            text="Some text",
            score=0.9,
            scope="project",
            citation=Citation(source="file.md"),
        )
        data = sr.model_dump()
        assert data["score"] == 0.9

    def test_repo_index_config(self):
        cfg = RepoIndexConfig(
            repo_url="https://github.com/org/repo",
            branch="develop",
            scope="project-code",
        )
        assert cfg.branch == "develop"
        assert "**/*.py" in cfg.include_patterns


# ── Chunking ─────────────────────────────────────────────────────────────


class TestChunking:
    def test_fixed_size_chunking(self):
        content = "A" * 300
        chunks = chunk_document(
            content,
            "test.txt",
            "scope",
            DocumentFormat.TEXT,
            IngestConfig(chunk_size=100, chunk_overlap=20),
        )
        assert len(chunks) >= 3
        assert all(len(c.text) <= 100 for c in chunks)

    def test_paragraph_chunking(self):
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_document(
            content,
            "test.md",
            "scope",
            DocumentFormat.MARKDOWN,
            IngestConfig(chunking_strategy=ChunkingStrategy.PARAGRAPH, chunk_size=100),
        )
        assert len(chunks) >= 1

    def test_code_aware_chunking(self):
        content = """import os

def func_a():
    return 1

def func_b():
    return 2

class MyClass:
    pass
"""
        chunks = chunk_document(
            content,
            "test.py",
            "scope",
            DocumentFormat.CODE,
            IngestConfig(chunking_strategy=ChunkingStrategy.CODE_AWARE, chunk_size=200),
        )
        assert len(chunks) >= 1

    def test_empty_content(self):
        chunks = chunk_document(
            "",
            "empty.txt",
            "scope",
            DocumentFormat.TEXT,
            IngestConfig(chunk_size=100),
        )
        assert len(chunks) == 0

    def test_chunk_metadata_populated(self):
        chunks = chunk_document(
            "# Title\nSome content here.",
            "doc.md",
            "my-scope",
            DocumentFormat.MARKDOWN,
            IngestConfig(chunk_size=1000),
        )
        assert len(chunks) == 1
        assert chunks[0].metadata.source == "doc.md"
        assert chunks[0].metadata.scope == "my-scope"
        assert chunks[0].metadata.section == "Title"
        assert chunks[0].metadata.total_chunks == 1


# ── KnowledgeService ─────────────────────────────────────────────────────


class TestKnowledgeService:
    def setup_method(self):
        self.svc = KnowledgeService()

    def test_ingest_creates_chunks(self):
        result = self.svc.ingest(
            scope="test-scope",
            content="This is a test document with enough content to create chunks.",
            source="test.txt",
        )
        assert result.chunks_created >= 1
        assert result.scope == "test-scope"

    def test_ingest_updates_scope_stats(self):
        self.svc.ingest("scope-a", "Document one content here.", "doc1.txt")
        self.svc.ingest("scope-a", "Document two content here.", "doc2.txt")

        scope = self.svc.get_scope("scope-a")
        assert scope is not None
        assert scope.document_count == 2
        assert scope.chunk_count >= 2

    def test_search_finds_content(self):
        self.svc.ingest(
            "scope-a", "Kubernetes is a container orchestration platform.", "k8s.md"
        )
        self.svc.ingest("scope-a", "Python is a programming language.", "python.md")

        results = self.svc.search("Kubernetes", ["scope-a"])
        assert len(results) >= 1
        assert results[0].score > 0

    def test_search_scoped_isolation(self):
        """Search in scope-a should NOT return scope-b results (#100)."""
        self.svc.ingest("scope-a", "Kubernetes deployment guide.", "k8s.md")
        self.svc.ingest("scope-b", "React frontend tutorial.", "react.md")

        results = self.svc.search("Kubernetes", ["scope-a"])
        for r in results:
            assert r.scope == "scope-a"

    def test_search_multi_scope(self):
        """Search across multiple scopes returns results from all (#100)."""
        self.svc.ingest("scope-a", "Kubernetes is great.", "k8s.md")
        self.svc.ingest("scope-b", "Kubernetes on AWS.", "aws.md")

        results = self.svc.search("Kubernetes", ["scope-a", "scope-b"])
        scopes_found = {r.scope for r in results}
        assert "scope-a" in scopes_found
        assert "scope-b" in scopes_found

    def test_search_with_citations(self):
        """Results include source citations (#102)."""
        self.svc.ingest("scope", "Deploy with helm install chart.", "deploy.md")
        results = self.svc.search("helm", ["scope"])
        assert len(results) >= 1
        assert results[0].citation.source == "deploy.md"

    def test_search_no_match(self):
        self.svc.ingest("scope", "Nothing relevant here.", "doc.txt")
        results = self.svc.search("quantumcomputing", ["scope"])
        assert len(results) == 0

    def test_list_scopes(self):
        self.svc.ingest("alpha", "Content", "a.txt")
        self.svc.ingest("beta", "Content", "b.txt")
        scopes = self.svc.list_scopes()
        names = [s.name for s in scopes]
        assert "alpha" in names
        assert "beta" in names

    def test_delete_scope(self):
        self.svc.ingest("doomed", "Delete me.", "temp.txt")
        assert self.svc.delete_scope("doomed") is True
        assert self.svc.get_scope("doomed") is None
        assert self.svc.delete_scope("doomed") is False

    def test_delete_document(self):
        self.svc.ingest("scope", "Keep this.", "keep.txt")
        self.svc.ingest("scope", "Delete this.", "delete.txt")
        removed = self.svc.delete_document("scope", "delete.txt")
        assert removed >= 1
        scope = self.svc.get_scope("scope")
        assert scope.document_count == 1


# ── Citations ────────────────────────────────────────────────────────────


class TestCitations:
    def test_build_citations_from_payload(self):
        raw = [
            {
                "score": 0.9,
                "payload": {
                    "text": "Some chunk text",
                    "metadata": {
                        "source": "deploy.py",
                        "section": "Deploy Function",
                        "scope": "project",
                        "chunk_index": 3,
                    },
                },
            },
        ]
        results = build_citations(raw)
        assert len(results) == 1
        assert results[0].citation.source == "deploy.py"
        assert results[0].citation.section == "Deploy Function"

    def test_format_citations_markdown(self):
        results = [
            SearchResult(
                text="chunk",
                score=0.9,
                scope="scope",
                citation=Citation(source="file.py", section="main", score=0.9),
            ),
            SearchResult(
                text="chunk2",
                score=0.7,
                scope="scope",
                citation=Citation(source="readme.md", page=3, score=0.7),
            ),
        ]
        text = format_citations(results)
        assert "## Sources" in text
        assert "file.py" in text
        assert "readme.md" in text
        assert "p.3" in text

    def test_format_citations_empty(self):
        assert format_citations([]) == ""


# ── Dedup Hash ───────────────────────────────────────────────────────────


class TestChunkHash:
    def test_consistent_hash(self):
        h1 = compute_chunk_hash("Hello world")
        h2 = compute_chunk_hash("Hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_chunk_hash("Hello")
        h2 = compute_chunk_hash("World")
        assert h1 != h2


# ── Dedup (#101) ─────────────────────────────────────────────────────────


class TestChunkDedup:
    """compute_chunk_hash is wired into ingest — identical chunks are skipped (#101)."""

    def setup_method(self):
        self.svc = KnowledgeService()

    def test_identical_content_not_duplicated(self):
        """Ingesting the same content twice → chunks_created=0 on second call."""
        content = "The quick brown fox jumps over the lazy dog. " * 30
        r1 = self.svc.ingest("scope", content, "file.txt")
        r2 = self.svc.ingest("scope", content, "file.txt")
        assert r1.chunks_created >= 1
        assert r2.chunks_created == 0  # all chunks deduped

    def test_different_content_not_deduped(self):
        """Different content ingests fresh chunks."""
        self.svc.ingest("scope", "Content A " * 20, "a.txt")
        r2 = self.svc.ingest("scope", "Content B " * 20, "b.txt")
        assert r2.chunks_created >= 1

    def test_scope_isolation_no_cross_dedup(self):
        """Dedup is per-scope: same content in different scopes each gets chunks."""
        content = "Shared content " * 30
        r1 = self.svc.ingest("scope-x", content, "doc.txt")
        r2 = self.svc.ingest("scope-y", content, "doc.txt")
        assert r1.chunks_created >= 1
        assert r2.chunks_created >= 1  # different scope → not deduped

    def test_partial_update_deduped_correctly(self):
        """Re-indexing with one changed paragraph: only the changed chunk is new.

        Use paragraph chunking so each paragraph → exactly 1 chunk.
        After first ingest: 2 chunks (A + B). After second ingest with B changed:
        paragraph A hash already seen → skipped; paragraph B is new → 1 chunk.
        """
        from models.knowledge import IngestConfig, ChunkingStrategy

        # Each paragraph > chunk_size/2 → they cannot merge into one chunk (648 > 400)
        para_a = (
            "Paragraph A about Kubernetes deployment strategies. " * 8
        )  # ~408 chars
        para_b = (
            "Paragraph B about Terraform provisioning workflows. " * 8
        )  # ~408 chars
        original = para_a + "\n\n" + para_b

        # chunk_size=450: each paragraph ≤ 450 chars (fits alone), combined > 450 (splits)
        cfg = IngestConfig(chunking_strategy=ChunkingStrategy.PARAGRAPH, chunk_size=450)
        r1 = self.svc.ingest("scope", original, "doc.txt", config=cfg)
        assert r1.chunks_created == 2  # sanity: each paragraph → its own chunk

        # Change only paragraph B
        para_b_new = "Paragraph B UPDATED entirely different content Ansible. " * 8
        updated = para_a + "\n\n" + para_b_new
        r2 = self.svc.ingest("scope", updated, "doc.txt", config=cfg)

        # A → deduped (hash seen); B changed → 1 new chunk
        assert r2.chunks_created == 1


# ── API Endpoints ────────────────────────────────────────────────────────


class TestKnowledgeEndpoints:
    def test_list_scopes_empty(self):
        import os
        from fastapi.testclient import TestClient
        from main import app

        token = os.environ.get("KNOWLEDGE_API_INTERNAL_TOKEN", "test-internal-token-for-tests")
        with TestClient(app) as client:
            resp = client.get("/api/v1/knowledge/scopes",
                              headers={"X-Internal-Token": token})
        assert resp.status_code == 200
        data = resp.json()
        assert "scopes" in data
        assert "count" in data

    def test_webhook_endpoint_retired(self):
        """POST /api/v1/knowledge/webhook returns 410 Gone (#303 — retired in favor of file upload)."""
        from fastapi.testclient import TestClient
        from main import app

        token = os.environ.get("KNOWLEDGE_API_INTERNAL_TOKEN", "test-internal-token-for-tests")
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/knowledge/webhook",
                headers={"X-Internal-Token": token},
                params={
                    "scope": "webhook-test",
                    "source": "README.md",
                    "content": "# Project\n\nThis project does amazing things. " * 10,
                    "format": "markdown",
                },
            )
        assert resp.status_code == 410
        assert "retired" in resp.json()["detail"].lower()
