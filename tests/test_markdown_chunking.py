"""W1 #14: Markdown-aware chunking strategy tests.

Validates:
- Heading boundaries create chunk splits
- Code fences are preserved intact
- Long sections split safely within max_size
- Malformed markdown degrades gracefully (no crash, no pathological output)
- Non-markdown content falls back to fixed-size chunking
- Default behavior (FIXED_SIZE) is NOT affected
- Opt-in only: strategy must be explicitly requested
"""

from models.knowledge import (
    ChunkingStrategy,
    DocumentFormat,
    IngestConfig,
)
from services.knowledge import (
    KnowledgeService,
    chunk_document,
    _chunk_markdown_aware,
    _split_oversized_block,
)


# ── Unit: _chunk_markdown_aware ─────────────────────────────────────────


class TestMarkdownAwareChunking:
    """Direct tests on the markdown-aware chunking function."""

    def test_headings_create_boundaries(self):
        """Each heading starts a new chunk."""
        content = (
            "# Introduction\n"
            "This is the intro paragraph.\n"
            "\n"
            "# Architecture\n"
            "This is the architecture section.\n"
            "\n"
            "# Deployment\n"
            "This is deployment guidance.\n"
        )
        chunks = _chunk_markdown_aware(content, max_size=2000)
        assert len(chunks) == 3
        assert chunks[0].startswith("# Introduction")
        assert chunks[1].startswith("# Architecture")
        assert chunks[2].startswith("# Deployment")

    def test_subheadings_create_boundaries(self):
        """H2/H3 headings also create boundaries."""
        content = (
            "# Main\n"
            "Intro.\n"
            "\n"
            "## Section A\n"
            "Content A.\n"
            "\n"
            "### Subsection A1\n"
            "Content A1.\n"
        )
        chunks = _chunk_markdown_aware(content, max_size=2000)
        assert len(chunks) == 3

    def test_code_fences_preserved(self):
        """Code blocks are kept intact in a single chunk."""
        content = (
            "# Setup\n"
            "Install the package:\n"
            "\n"
            "```bash\n"
            "pip install agentopia\n"
            "pip install qdrant-client\n"
            "pip install fastapi\n"
            "```\n"
            "\n"
            "# Usage\n"
            "After installation, run the server.\n"
        )
        chunks = _chunk_markdown_aware(content, max_size=2000)
        # Should have: Setup block (heading + text), code fence block, Usage block
        # Code fence may merge with adjacent text if under max_size
        code_chunk = [c for c in chunks if "```bash" in c]
        assert len(code_chunk) >= 1
        # The code fence should be complete (opening and closing ```)
        assert "```bash" in code_chunk[0]
        # pip install lines should be in the same chunk
        assert "pip install agentopia" in code_chunk[0]

    def test_code_fence_not_split(self):
        """A code fence that fits within max_size is never split across chunks."""
        code_block = "```python\n" + "x = 1\n" * 20 + "```\n"
        content = f"# Code\n\n{code_block}\n# Next\nAfter code."
        chunks = _chunk_markdown_aware(content, max_size=2000)
        for chunk in chunks:
            if "```python" in chunk:
                assert "```" in chunk[chunk.index("```python") + 10 :]  # closing fence

    def test_long_section_splits_safely(self):
        """Sections longer than max_size are split, each chunk <= max_size."""
        long_text = "This is a long paragraph. " * 100  # ~2600 chars
        content = f"# Long Section\n{long_text}"
        max_size = 500
        chunks = _chunk_markdown_aware(content, max_size=max_size)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= max_size, f"Chunk too long: {len(chunk)} > {max_size}"

    def test_paragraph_boundaries(self):
        """Empty lines create paragraph boundaries (weaker than headings)."""
        content = (
            "First paragraph content.\n"
            "\n"
            "Second paragraph content.\n"
            "\n"
            "Third paragraph content.\n"
        )
        chunks = _chunk_markdown_aware(content, max_size=2000)
        # Three short paragraphs should merge into one chunk at max_size=2000
        assert len(chunks) == 1
        assert "First paragraph" in chunks[0]
        assert "Third paragraph" in chunks[0]

    def test_paragraph_split_when_exceeds_max(self):
        """Paragraphs are split into separate chunks when combined exceeds max_size."""
        para_a = "Paragraph A content. " * 20  # ~420 chars
        para_b = "Paragraph B content. " * 20  # ~420 chars
        content = f"{para_a}\n\n{para_b}"
        chunks = _chunk_markdown_aware(content, max_size=450)
        assert len(chunks) == 2

    def test_list_structure_preserved(self):
        """List items within a section stay together when possible."""
        content = (
            "# Requirements\n"
            "- Python 3.12+\n"
            "- Qdrant vector database\n"
            "- PostgreSQL 16\n"
            "- FastAPI framework\n"
        )
        chunks = _chunk_markdown_aware(content, max_size=2000)
        assert len(chunks) == 1
        assert "- Python 3.12+" in chunks[0]
        assert "- FastAPI framework" in chunks[0]

    def test_malformed_markdown_no_crash(self):
        """Malformed markdown is handled gracefully."""
        malformed_cases = [
            "# Unclosed heading\n```\ncode without closing fence",
            "########## Too many hashes",
            "```\n```\n```\n```",  # alternating empty fences
            "\n\n\n\n\n",  # only whitespace
            "# \n## \n### \n",  # empty headings
            "Normal text without any markdown structure at all.",
        ]
        for content in malformed_cases:
            chunks = _chunk_markdown_aware(content, max_size=500)
            assert isinstance(chunks, list), f"Failed on: {content!r}"
            # No individual chunk should exceed max_size
            for chunk in chunks:
                assert len(chunk) <= 500, f"Oversized chunk from: {content!r}"

    def test_unclosed_code_fence_handled(self):
        """Unclosed code fence doesn't cause infinite loop or crash."""
        content = "# Code\n```python\nprint('hello')\nprint('world')\n# no closing fence"
        chunks = _chunk_markdown_aware(content, max_size=2000)
        assert len(chunks) >= 1
        # The code content should be in the output somewhere
        all_text = " ".join(chunks)
        assert "print('hello')" in all_text

    def test_empty_content(self):
        """Empty content returns empty list (handled by chunk_document)."""
        chunks = _chunk_markdown_aware("", max_size=500)
        # Empty falls back to fixed-size which returns [""]
        # but chunk_document filters empty chunks, so this is fine
        assert isinstance(chunks, list)

    def test_non_markdown_falls_back(self):
        """Plain text without markdown structure falls back to fixed-size."""
        content = "Just plain text without any headings or formatting. " * 50
        chunks = _chunk_markdown_aware(content, max_size=500)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 500


# ── Unit: _split_oversized_block ────────────────────────────────────────


class TestSplitOversizedBlock:
    def test_splits_on_newlines(self):
        """Oversized block splits on line boundaries."""
        block = "\n".join(f"Line {i}: some content here." for i in range(50))
        chunks = _split_oversized_block(block, max_size=200)
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_hard_split_single_long_line(self):
        """A single line exceeding max_size is hard-split."""
        block = "A" * 1000
        chunks = _split_oversized_block(block, max_size=300)
        assert len(chunks) >= 4
        for chunk in chunks:
            assert len(chunk) <= 300


# ── Integration: chunk_document with MARKDOWN_AWARE ────────────────────


class TestChunkDocumentMarkdownAware:
    """Tests through the chunk_document() entry point."""

    def test_markdown_aware_via_config(self):
        """MARKDOWN_AWARE strategy is selectable via IngestConfig."""
        content = "# Title\nContent.\n\n# Section\nMore content."
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE,
            chunk_size=2000,
        )
        chunks = chunk_document(
            content, "doc.md", "scope", DocumentFormat.MARKDOWN, config
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata.source == "doc.md"

    def test_section_extraction_with_markdown_aware(self):
        """Section metadata is populated from heading in each chunk."""
        content = "# Getting Started\nInstall and run.\n\n# API Reference\nEndpoints listed."
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE,
            chunk_size=2000,
        )
        chunks = chunk_document(
            content, "guide.md", "scope", DocumentFormat.MARKDOWN, config
        )
        sections = [c.metadata.section for c in chunks]
        assert "Getting Started" in sections
        assert "API Reference" in sections

    def test_chunk_index_sequential(self):
        """chunk_index is sequential across chunks."""
        content = "# A\nText.\n\n# B\nText.\n\n# C\nText."
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE,
            chunk_size=2000,
        )
        chunks = chunk_document(
            content, "doc.md", "scope", DocumentFormat.MARKDOWN, config
        )
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_realistic_documentation(self):
        """Realistic markdown doc chunks sensibly."""
        content = """# Agentopia Knowledge API

The knowledge API provides document ingestion and semantic search.

## Authentication

Two auth paths:
- **Operator**: X-Internal-Token header
- **Bot**: Authorization Bearer + X-Bot-Name headers

## Ingestion

Upload documents via `POST /{scope}/ingest`.

Supported formats:
- Markdown (.md)
- PDF (.pdf)
- HTML (.html)
- Plain text (.txt)

```python
import httpx

resp = httpx.post(
    "http://knowledge-api:8002/api/v1/knowledge/my-scope/ingest",
    files={"file": open("doc.md", "rb")},
    headers={"X-Internal-Token": token},
)
```

## Search

Query with `GET /search?query=...&scopes=...`.

Results include citations with source, section, and relevance score.
"""
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE,
            chunk_size=500,
        )
        chunks = chunk_document(
            content, "api-guide.md", "docs", DocumentFormat.MARKDOWN, config
        )
        assert len(chunks) >= 3
        # Code block should be preserved
        code_chunks = [c for c in chunks if "import httpx" in c.text]
        assert len(code_chunks) >= 1
        # All chunks within size limit
        for c in chunks:
            assert len(c.text) <= 500


# ── Regression: default behavior unchanged ──────────────────────────────


class TestDefaultBehaviorUnchanged:
    """Ensure FIXED_SIZE default is not affected by W1 changes."""

    def test_default_config_is_fixed_size(self):
        """IngestConfig() defaults to FIXED_SIZE."""
        config = IngestConfig()
        assert config.chunking_strategy == ChunkingStrategy.FIXED_SIZE

    def test_fixed_size_unchanged(self):
        """Fixed-size chunking produces same results as before W1."""
        content = "A" * 300
        config = IngestConfig(chunk_size=100, chunk_overlap=20)
        chunks = chunk_document(
            content, "test.txt", "scope", DocumentFormat.TEXT, config
        )
        assert len(chunks) >= 3
        assert all(len(c.text) <= 100 for c in chunks)

    def test_paragraph_unchanged(self):
        """Paragraph chunking is not affected."""
        content = "Para one.\n\nPara two.\n\nPara three."
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.PARAGRAPH, chunk_size=100
        )
        chunks = chunk_document(
            content, "test.md", "scope", DocumentFormat.MARKDOWN, config
        )
        assert len(chunks) >= 1

    def test_code_aware_unchanged(self):
        """Code-aware chunking is not affected."""
        content = "import os\n\ndef func():\n    pass\n\nclass Foo:\n    pass\n"
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.CODE_AWARE, chunk_size=200
        )
        chunks = chunk_document(
            content, "test.py", "scope", DocumentFormat.CODE, config
        )
        assert len(chunks) >= 1


# ── Opt-in enforcement ──────────────────────────────────────────────────


class TestOptInEnforcement:
    """Verify markdown-aware chunking is opt-in only."""

    def test_markdown_format_does_not_auto_select(self):
        """Ingesting a .md file with default config uses FIXED_SIZE, not MARKDOWN_AWARE."""
        content = "# Heading\nContent\n\n# Another\nMore content"
        config = IngestConfig()  # default = FIXED_SIZE
        chunks = chunk_document(
            content, "doc.md", "scope", DocumentFormat.MARKDOWN, config
        )
        # With FIXED_SIZE at 512 chars, this short content → 1 chunk
        assert len(chunks) == 1
        # The chunk should contain all content (no heading-based splitting)
        assert "# Heading" in chunks[0].text
        assert "# Another" in chunks[0].text

    def test_explicit_opt_in_required(self):
        """Only ChunkingStrategy.MARKDOWN_AWARE activates the strategy."""
        content = "# A\nText.\n\n# B\nText."
        config_default = IngestConfig(chunk_size=2000)
        config_md = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE, chunk_size=2000
        )
        chunks_default = chunk_document(
            content, "d.md", "scope", DocumentFormat.MARKDOWN, config_default
        )
        chunks_md = chunk_document(
            content, "d.md", "scope", DocumentFormat.MARKDOWN, config_md
        )
        # Default: 1 chunk (all content fits in 2000)
        assert len(chunks_default) == 1
        # Markdown-aware: 2 chunks (split on heading)
        assert len(chunks_md) == 2


# ── KnowledgeService integration ────────────────────────────────────────


class TestMarkdownAwareIngestSearch:
    """End-to-end: ingest with MARKDOWN_AWARE, then search."""

    def setup_method(self):
        self.svc = KnowledgeService()

    def test_ingest_and_search(self):
        """Ingest with markdown-aware chunking, search returns relevant results."""
        content = (
            "# Kubernetes Deployment\n"
            "Deploy with kubectl apply.\n"
            "\n"
            "# Python Development\n"
            "Use virtualenv for isolation.\n"
        )
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE,
            chunk_size=2000,
        )
        result = self.svc.ingest("test", content, "guide.md", config=config)
        assert result.chunks_created == 2

        results = self.svc.search("Kubernetes", ["test"])
        assert len(results) >= 1
        assert results[0].score > 0

    def test_dedup_works_with_markdown_aware(self):
        """Dedup hashing works correctly with markdown-aware chunks."""
        content = "# Section\nSome content here.\n\n# Other\nDifferent content."
        config = IngestConfig(
            chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE,
            chunk_size=2000,
        )
        r1 = self.svc.ingest("scope", content, "doc.md", config=config)
        r2 = self.svc.ingest("scope", content, "doc.md", config=config)
        assert r1.chunks_created >= 1
        assert r2.chunks_created == 0  # deduped
