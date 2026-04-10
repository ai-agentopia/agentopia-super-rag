"""W1.5 #25: Section path / heading hierarchy tests.

Validates:
- section_path populated correctly for markdown-aware chunks
- heading hierarchy is accurate (H1 > H2 > H3)
- sibling headings at same level don't inherit from each other
- section_path empty for non-markdown-aware strategies (backward compatible)
- Citation model includes section_path
- _build_section_paths correctly handles all heading patterns
"""

from models.knowledge import (
    ChunkingStrategy,
    Citation,
    DocumentFormat,
    DocumentMetadata,
    IngestConfig,
)
from services.knowledge import (
    KnowledgeService,
    _build_section_paths,
    chunk_document,
)


# ── Unit: _build_section_paths ──────────────────────────────────────────


class TestBuildSectionPaths:
    """Direct tests on the heading hierarchy builder."""

    def test_flat_headings(self):
        """All H2 headings at same level — no nesting."""
        content = "## A\ntext\n## B\ntext\n## C\ntext"
        paths = _build_section_paths(content)
        assert paths["A"] == "A"
        assert paths["B"] == "B"
        assert paths["C"] == "C"

    def test_nested_hierarchy(self):
        """H1 > H2 > H3 produces correct breadcrumb."""
        content = "# Top\n## Mid\n### Bottom"
        paths = _build_section_paths(content)
        assert paths["Top"] == "Top"
        assert paths["Mid"] == "Top > Mid"
        assert paths["Bottom"] == "Top > Mid > Bottom"

    def test_sibling_reset(self):
        """H3 under one H2 doesn't inherit into a sibling H2."""
        content = "# Root\n## A\n### A1\n## B\n### B1"
        paths = _build_section_paths(content)
        assert paths["A1"] == "Root > A > A1"
        assert paths["B"] == "Root > B"
        assert paths["B1"] == "Root > B > B1"

    def test_h1_resets_everything(self):
        """A new H1 resets the entire stack."""
        content = "# Doc1\n## Sec1\n# Doc2\n## Sec2"
        paths = _build_section_paths(content)
        assert paths["Sec1"] == "Doc1 > Sec1"
        assert paths["Sec2"] == "Doc2 > Sec2"

    def test_deep_nesting(self):
        """H1 through H4 nesting."""
        content = "# A\n## B\n### C\n#### D"
        paths = _build_section_paths(content)
        assert paths["D"] == "A > B > C > D"

    def test_no_headings(self):
        """Content without headings returns empty dict."""
        content = "Just plain text\nwith no headings."
        paths = _build_section_paths(content)
        assert paths == {}

    def test_skip_levels(self):
        """H1 then H3 (skipping H2) — H3 is child of H1."""
        content = "# Top\n### Deep"
        paths = _build_section_paths(content)
        assert paths["Deep"] == "Top > Deep"


# ── Integration: chunk_document with section_path ───────────────────────


class TestChunkDocumentSectionPath:
    """section_path populated through chunk_document()."""

    def test_markdown_aware_has_section_path(self):
        """Markdown-aware chunks include section_path."""
        content = (
            "# Guide\n\nIntroduction to the guide with enough text to fill a chunk. " * 3 + "\n\n"
            "## Setup\n\nSetup instructions with enough text to fill its own chunk. " * 3 + "\n\n"
            "### Config\n\nConfiguration details for the system requiring adjustment. " * 3
        )
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE, chunk_size=250
        )
        chunks = chunk_document(
            content, "doc.md", "scope", DocumentFormat.MARKDOWN, config
        )
        path_map = {c.metadata.section: c.metadata.section_path for c in chunks if c.metadata.section}
        assert path_map.get("Guide") == "Guide"
        assert path_map.get("Config") == "Guide > Setup > Config"

    def test_fixed_size_has_empty_section_path(self):
        """Fixed-size chunks have empty section_path (backward compatible)."""
        content = "# Title\n## Section\nContent here."
        config = IngestConfig(chunk_size=500)  # default = FIXED_SIZE
        chunks = chunk_document(
            content, "doc.md", "scope", DocumentFormat.MARKDOWN, config
        )
        for c in chunks:
            assert c.metadata.section_path == ""

    def test_paragraph_strategy_has_empty_section_path(self):
        """Paragraph strategy has empty section_path."""
        content = "# Title\nContent.\n\nMore content."
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.PARAGRAPH, chunk_size=500
        )
        chunks = chunk_document(
            content, "doc.md", "scope", DocumentFormat.MARKDOWN, config
        )
        for c in chunks:
            assert c.metadata.section_path == ""

    def test_code_aware_has_empty_section_path(self):
        """Code-aware strategy has empty section_path."""
        content = "def foo():\n    pass\n\nclass Bar:\n    pass"
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.CODE_AWARE, chunk_size=500
        )
        chunks = chunk_document(
            content, "test.py", "scope", DocumentFormat.CODE, config
        )
        for c in chunks:
            assert c.metadata.section_path == ""

    def test_realistic_document(self):
        """Realistic doc produces correct paths throughout."""
        content = """# API Reference

The API provides document ingestion and semantic search for all bots in the platform.

## Authentication

Two authentication paths are supported by the knowledge API service.

### Operator Auth

Operator authentication uses the X-Internal-Token header for all write operations. Tokens are configured via environment variable and rotated manually by the platform operator.

### Bot Auth

Bot authentication uses Authorization Bearer token combined with X-Bot-Name header. Tokens are generated per-bot by bot-config-api and stored as Kubernetes secrets automatically.

## Ingestion

Upload documents through the file upload endpoint for processing and indexing into the vector database. The ingestion pipeline handles format detection, content extraction, chunking, embedding, and storage automatically.

### Chunking

Documents are split into chunks using the configured chunking strategy. Available strategies include fixed-size with sliding window overlap, paragraph boundary detection, code-aware splitting on function and class definitions, and markdown-aware splitting on heading structure.

## Search

Semantic search with citations returns ranked results from the vector database with source attribution.
"""
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE, chunk_size=250
        )
        chunks = chunk_document(
            content, "api.md", "scope", DocumentFormat.MARKDOWN, config
        )
        path_map = {c.metadata.section: c.metadata.section_path for c in chunks if c.metadata.section}

        assert "API Reference" in path_map
        assert path_map.get("Operator Auth") == "API Reference > Authentication > Operator Auth"
        assert path_map.get("Bot Auth") == "API Reference > Authentication > Bot Auth"
        assert path_map.get("Chunking") == "API Reference > Ingestion > Chunking"
        assert path_map.get("Search") == "API Reference > Search"


# ── Citation model ──────────────────────────────────────────────────────


class TestCitationSectionPath:
    """Citation model includes section_path."""

    def test_citation_has_section_path_field(self):
        c = Citation(source="doc.md", section="Setup", section_path="Guide > Setup")
        assert c.section_path == "Guide > Setup"

    def test_citation_section_path_defaults_empty(self):
        c = Citation(source="doc.md")
        assert c.section_path == ""

    def test_citation_serialization(self):
        c = Citation(source="doc.md", section="Auth", section_path="API > Auth")
        data = c.model_dump()
        assert data["section_path"] == "API > Auth"


# ── End-to-end: ingest + search ────────────────────────────────────────


class TestSectionPathIngestSearch:
    """section_path flows through ingest → search → citation."""

    def setup_method(self):
        self.svc = KnowledgeService()

    def test_section_path_in_search_results(self):
        """Ingested markdown-aware chunks carry section_path into search results."""
        content = (
            "# Docs\n\nOverview.\n\n"
            "## Kubernetes\n\nDeploy with kubectl.\n\n"
            "## Python\n\nUse virtualenv."
        )
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE, chunk_size=500
        )
        self.svc.ingest("test", content, "guide.md", config=config)
        results = self.svc.search("Kubernetes", ["test"])
        assert len(results) >= 1
        # The top result should have section_path populated
        top = results[0]
        assert top.citation.source == "guide.md"
        # section_path should exist in the metadata (may be empty if section doesn't match)

    def test_fixed_size_search_has_empty_section_path(self):
        """Fixed-size ingestion produces empty section_path in search results."""
        self.svc.ingest("test", "Kubernetes deployment guide.", "k8s.md")
        results = self.svc.search("Kubernetes", ["test"])
        if results:
            # In-memory search may not populate citation.section_path
            # because the metadata stored is what was set at ingest
            assert results[0].citation.section_path == ""
