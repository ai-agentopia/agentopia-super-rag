"""Regression tests for orchestrator ingest scope statistics.

Bug: ingest_from_orchestrator() was updating last_indexed but NOT document_count
or chunk_count. As a result list_scopes() reported zero counts even after
successful ingest, causing the KB page to show "Not indexed".

These tests verify the fix maintains scope stats correctly for:
- First ingest: increments document_count and chunk_count
- Replacement (same document_id, new version): document_count unchanged, chunk_count adjusts
- Multiple documents in same scope: document_count reflects distinct documents
- Idempotent skip (same version): counts unchanged
"""

from models.knowledge import OrchestratorIngestRequest, OrchestratorIngestMetadata, ChunkingStrategy
from services.knowledge import KnowledgeService
from services.document_store import InMemoryDocumentStore


def _svc() -> KnowledgeService:
    """Fresh KnowledgeService with an in-memory doc_store for each test."""
    s = KnowledgeService()
    s._doc_store = InMemoryDocumentStore()
    return s


def _make_request(doc_id: str, version: int, text: str, fmt: str = "markdown") -> OrchestratorIngestRequest:
    return OrchestratorIngestRequest(
        document_id=doc_id,
        version=version,
        text=text,
        metadata=OrchestratorIngestMetadata(format=fmt),
        chunking_strategy=ChunkingStrategy.MARKDOWN_AWARE if fmt == "markdown" else ChunkingStrategy.FIXED_SIZE,
    )


class TestOrchestratorIngestScopeStats:
    """Scope stats must be maintained on ingest_from_orchestrator path."""

    def test_first_ingest_increments_document_and_chunk_counts(self):
        svc = _svc()
        scope = "acme/api-docs"

        text = "# Title\n\nFirst paragraph with content.\n\n## Section\n\nMore content here that will be chunked appropriately.\n"
        req = _make_request(doc_id="doc-1", version=1, text=text)

        resp = svc.ingest_from_orchestrator(scope, req)

        assert resp.status == "indexed"
        assert resp.chunk_count > 0

        # Scope stats must reflect the ingested document
        scopes = svc.list_scopes()
        scope_obj = next((s for s in scopes if s.name == scope), None)
        assert scope_obj is not None, "scope must be present in list_scopes"
        assert scope_obj.document_count == 1, "document_count must be 1 after first ingest"
        assert scope_obj.chunk_count == resp.chunk_count, "chunk_count must match response"
        assert scope_obj.last_indexed is not None

    def test_replacement_keeps_document_count_stable(self):
        svc = _svc()
        scope = "acme/api-docs"

        # Initial ingest
        req1 = _make_request(doc_id="doc-1", version=1, text="# V1\n\nOriginal content here with enough text to chunk.\n")
        svc.ingest_from_orchestrator(scope, req1)
        initial_scope = next(s for s in svc.list_scopes() if s.name == scope)
        assert initial_scope.document_count == 1
        initial_chunks = initial_scope.chunk_count

        # Replace with v2
        req2 = _make_request(doc_id="doc-1", version=2, text="# V2\n\nRevised content with different text and more detail added.\n\n## Extra section\n\nAdditional content.\n")
        resp2 = svc.ingest_from_orchestrator(scope, req2)

        # document_count must not double-count — same logical document
        replaced_scope = next(s for s in svc.list_scopes() if s.name == scope)
        assert replaced_scope.document_count == 1, (
            f"document_count must stay 1 on replacement, got {replaced_scope.document_count}"
        )
        # chunk_count must reflect the new version's chunks (not added on top)
        assert replaced_scope.chunk_count == resp2.chunk_count, (
            f"chunk_count must equal new version's chunks ({resp2.chunk_count}), got {replaced_scope.chunk_count}"
        )

    def test_multiple_documents_increment_document_count(self):
        svc = _svc()
        scope = "acme/api-docs"

        r1 = svc.ingest_from_orchestrator(scope, _make_request("doc-a", 1, "# A\n\nContent for document A.\n"))
        r2 = svc.ingest_from_orchestrator(scope, _make_request("doc-b", 1, "# B\n\nContent for document B.\n"))
        r3 = svc.ingest_from_orchestrator(scope, _make_request("doc-c", 1, "# C\n\nContent for document C.\n"))

        scope_obj = next(s for s in svc.list_scopes() if s.name == scope)
        assert scope_obj.document_count == 3, f"document_count must be 3, got {scope_obj.document_count}"
        assert scope_obj.chunk_count == r1.chunk_count + r2.chunk_count + r3.chunk_count

    def test_idempotent_same_version_does_not_change_counts(self):
        svc = _svc()
        scope = "acme/api-docs"

        text = "# Same\n\nIdentical content for idempotency check.\n"
        svc.ingest_from_orchestrator(scope, _make_request("doc-x", 1, text))
        before = next(s for s in svc.list_scopes() if s.name == scope)
        doc_count_before = before.document_count
        chunk_count_before = before.chunk_count

        # Re-ingest same (document_id, version)
        resp = svc.ingest_from_orchestrator(scope, _make_request("doc-x", 1, text))
        assert resp.status == "skipped"

        after = next(s for s in svc.list_scopes() if s.name == scope)
        assert after.document_count == doc_count_before, "document_count must not change on skipped re-ingest"
        assert after.chunk_count == chunk_count_before, "chunk_count must not change on skipped re-ingest"

    def test_list_scopes_reports_indexed_state_after_orchestrator_ingest(self):
        """The KB page reads /api/v1/knowledge/scopes — must reflect truth, not zeros."""
        svc = _svc()
        scope = "client/kb"

        # Before ingest — scope may not exist or have zero counts
        before = next((s for s in svc.list_scopes() if s.name == scope), None)
        assert before is None or (before.document_count == 0 and before.chunk_count == 0)

        # After ingest via orchestrator path — list_scopes must reflect indexed state
        resp = svc.ingest_from_orchestrator(scope, _make_request("doc-1", 1, "# Title\n\nContent.\n"))
        assert resp.chunk_count > 0

        after = next((s for s in svc.list_scopes() if s.name == scope), None)
        assert after is not None
        assert after.document_count > 0, "KB page would show 'Not indexed' if this is 0"
        assert after.chunk_count > 0, "chunk_count must be non-zero after successful ingest"

    def test_replacement_chunk_count_adjusts_on_bigger_new_version(self):
        svc = _svc()
        scope = "acme/api-docs"

        # V1: small content → few chunks
        svc.ingest_from_orchestrator(scope, _make_request("doc-1", 1, "# Small\n\nShort content.\n"))
        v1_chunks = next(s for s in svc.list_scopes() if s.name == scope).chunk_count

        # V2: much larger content → more chunks
        big_text = "# Big\n\n" + "\n\n".join(f"## Section {i}\n\n" + ("word " * 200) for i in range(5))
        resp2 = svc.ingest_from_orchestrator(scope, _make_request("doc-1", 2, big_text))
        v2_scope = next(s for s in svc.list_scopes() if s.name == scope)

        # chunk_count should be the new version's count (not v1+v2)
        assert v2_scope.chunk_count == resp2.chunk_count, (
            f"After replace, chunk_count={v2_scope.chunk_count} must equal new version chunks {resp2.chunk_count}"
        )
        assert v2_scope.document_count == 1

    def test_non_orchestrator_ingest_path_still_works(self):
        """Regression: ensure we didn't break the manual ingest() path."""
        from models.knowledge import DocumentFormat
        svc = _svc()
        scope = "acme/manual"

        result = svc.ingest(
            scope=scope,
            content="# Manual\n\nContent from manual path.\n",
            source="manual.md",
            format=DocumentFormat.MARKDOWN,
        )
        assert result.chunks_created > 0

        scope_obj = next(s for s in svc.list_scopes() if s.name == scope)
        assert scope_obj.document_count == 1
        assert scope_obj.chunk_count == result.chunks_created
