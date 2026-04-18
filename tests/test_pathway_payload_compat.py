"""Tests for build_citations Pathway flat-payload compatibility (#99).

Verifies that build_citations correctly handles three payload schemas:
1. Legacy nested: metadata fields inside payload.metadata dict
2. Pathway flat: all fields at payload top level, document_id instead of source
3. Mixed/partial: metadata dict present but incomplete, with top-level fallbacks

The fix must not regress existing nested-metadata handling.
"""

from services.knowledge import build_citations


class TestBuildCitationsLegacyNested:
    """Legacy payload: all citation fields nested inside payload.metadata."""

    def test_all_fields_from_metadata(self):
        raw = [
            {
                "score": 0.92,
                "payload": {
                    "text": "Deploy function handles blue/green rollouts.",
                    "metadata": {
                        "source": "deploy.py",
                        "section": "Deploy Function",
                        "section_path": "Deploy > Deploy Function",
                        "scope": "project-alpha",
                        "chunk_index": 2,
                        "page": 5,
                        "ingested_at": 1711800000.0,
                        "document_hash": "abc123def456",
                    },
                },
            }
        ]
        results = build_citations(raw)
        assert len(results) == 1
        r = results[0]
        assert r.text == "Deploy function handles blue/green rollouts."
        assert r.score == 0.92
        assert r.scope == "project-alpha"
        assert r.citation.source == "deploy.py"
        assert r.citation.section == "Deploy Function"
        assert r.citation.section_path == "Deploy > Deploy Function"
        assert r.citation.chunk_index == 2
        assert r.citation.page == 5
        assert r.citation.ingested_at == 1711800000.0
        assert r.citation.document_hash == "abc123def456"

    def test_legacy_defaults_for_missing_fields(self):
        raw = [
            {
                "score": 0.5,
                "payload": {
                    "text": "some text",
                    "metadata": {
                        "source": "file.md",
                        "scope": "test/scope",
                    },
                },
            }
        ]
        results = build_citations(raw)
        r = results[0]
        assert r.citation.source == "file.md"
        assert r.citation.section == ""
        assert r.citation.chunk_index == 0
        assert r.citation.ingested_at == 0.0
        assert r.citation.document_hash == ""


class TestBuildCitationsPathwayFlat:
    """Pathway flat payload: no metadata dict, all fields at top level."""

    def test_flat_payload_with_document_id(self):
        """Pathway writes document_id (not source). Must map to citation.source."""
        raw = [
            {
                "score": 0.88,
                "payload": {
                    "document_id": "architecture/super-rag-debate.md",
                    "scope": "utop/oddspark",
                    "section": "C1. Retrieval Strategy",
                    "section_path": "architecture/super-rag-debate.md",
                    "chunk_index": 5,
                    "total_chunks": 62,
                    "text": "For retrieval, we recommend a hybrid approach...",
                    "status": "active",
                    "document_hash": "deadbeef12345678",
                    "ingested_at": 1776484426.18,
                },
            }
        ]
        results = build_citations(raw)
        assert len(results) == 1
        r = results[0]
        assert r.text == "For retrieval, we recommend a hybrid approach..."
        assert r.score == 0.88
        assert r.scope == "utop/oddspark"
        # document_id must map to citation.source
        assert r.citation.source == "architecture/super-rag-debate.md"
        assert r.citation.section == "C1. Retrieval Strategy"
        assert r.citation.chunk_index == 5
        assert r.citation.ingested_at == 1776484426.18
        assert r.citation.document_hash == "deadbeef12345678"

    def test_flat_payload_no_metadata_key(self):
        """Payload has no 'metadata' key at all — must not crash."""
        raw = [
            {
                "score": 0.75,
                "payload": {
                    "document_id": "architecture/overview.md",
                    "scope": "utop/oddspark",
                    "section": "Dual-Lane Model",
                    "chunk_index": 3,
                    "text": "The dual-lane model separates...",
                    "ingested_at": 1776000000.0,
                    "document_hash": "aabbccdd",
                },
            }
        ]
        results = build_citations(raw)
        r = results[0]
        assert r.citation.source == "architecture/overview.md"
        assert r.scope == "utop/oddspark"
        assert r.citation.section == "Dual-Lane Model"
        assert r.citation.chunk_index == 3

    def test_flat_payload_empty_metadata_dict(self):
        """Payload has metadata: {} (empty) — must fall back to top-level."""
        raw = [
            {
                "score": 0.60,
                "payload": {
                    "metadata": {},
                    "document_id": "architecture/a2a-comparison.md",
                    "scope": "utop/oddspark",
                    "section": "BeeAI vs Agentopia",
                    "chunk_index": 1,
                    "text": "Comparing implementations...",
                    "ingested_at": 1776000001.0,
                    "document_hash": "11223344",
                },
            }
        ]
        results = build_citations(raw)
        r = results[0]
        assert r.citation.source == "architecture/a2a-comparison.md"
        assert r.scope == "utop/oddspark"
        assert r.citation.section == "BeeAI vs Agentopia"

    def test_multiple_flat_results(self):
        """Multiple Pathway results in a single search response."""
        raw = [
            {
                "score": 0.90,
                "payload": {
                    "document_id": "architecture/file-a.md",
                    "scope": "utop/oddspark",
                    "section": "Section A",
                    "chunk_index": 0,
                    "text": "Content A",
                    "ingested_at": 1.0,
                    "document_hash": "hash_a",
                },
            },
            {
                "score": 0.80,
                "payload": {
                    "document_id": "architecture/file-b.md",
                    "scope": "utop/oddspark",
                    "section": "Section B",
                    "chunk_index": 2,
                    "text": "Content B",
                    "ingested_at": 2.0,
                    "document_hash": "hash_b",
                },
            },
        ]
        results = build_citations(raw)
        assert len(results) == 2
        assert results[0].citation.source == "architecture/file-a.md"
        assert results[1].citation.source == "architecture/file-b.md"
        assert results[0].citation.chunk_index == 0
        assert results[1].citation.chunk_index == 2


class TestBuildCitationsMixedPayload:
    """Edge cases: partial metadata, both source and document_id present."""

    def test_metadata_with_source_takes_precedence(self):
        """When metadata.source exists, it wins over top-level document_id."""
        raw = [
            {
                "score": 0.85,
                "payload": {
                    "document_id": "should-not-be-used.md",
                    "text": "text",
                    "metadata": {
                        "source": "correct-source.md",
                        "section": "Correct",
                        "scope": "test/scope",
                        "chunk_index": 7,
                    },
                },
            }
        ]
        results = build_citations(raw)
        r = results[0]
        # metadata.source must win — legacy path takes precedence
        assert r.citation.source == "correct-source.md"
        assert r.citation.section == "Correct"
        assert r.citation.chunk_index == 7

    def test_empty_payload(self):
        """Completely empty payload — must not crash, returns empty defaults."""
        raw = [{"score": 0.1, "payload": {}}]
        results = build_citations(raw)
        r = results[0]
        assert r.citation.source == ""
        assert r.citation.section == ""
        assert r.citation.chunk_index == 0
        assert r.scope == ""

    def test_no_payload_key(self):
        """Result with no payload key at all — must not crash."""
        raw = [{"score": 0.1, "text": "fallback text"}]
        results = build_citations(raw)
        r = results[0]
        assert r.text == "fallback text"
        assert r.citation.source == ""
