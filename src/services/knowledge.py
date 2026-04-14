"""Knowledge base service — ingestion, chunking, retrieval (#330).

Owned by knowledge-api. This is the sole retrieval data plane.
Bot search: gateway → knowledge-api directly (#328).
Operator reads/writes: bot-config-api → knowledge-api proxy (#320).
"""

import hashlib
import logging
import os
import time
from typing import Any

from models.knowledge import (
    ChunkingStrategy,
    Citation,
    DocumentChunk,
    DocumentFormat,
    DocumentMetadata,
    DocumentRecord,
    DocumentRecordStatus,
    IngestConfig,
    IngestResult,
    KnowledgeScope,
    OrchestratorIngestRequest,
    OrchestratorIngestResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


def parse_pdf(raw_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    from PyPDF2 import PdfReader
    import io

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def parse_html(raw_bytes: bytes) -> str:
    """Extract text from HTML bytes using BeautifulSoup."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_bytes, "html.parser")
    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def chunk_document(
    content: str,
    source: str,
    scope: str,
    format: DocumentFormat,
    config: IngestConfig,
    document_hash: str = "",
    ingested_at: float = 0.0,
) -> list[DocumentChunk]:
    """Split document into chunks based on chunking strategy.

    Returns list of DocumentChunks ready for embedding.
    """
    if config.chunking_strategy == ChunkingStrategy.MARKDOWN_AWARE:
        texts = _chunk_markdown_aware(content, config.chunk_size)
    elif config.chunking_strategy == ChunkingStrategy.PARAGRAPH:
        texts = _chunk_by_paragraph(content, config.chunk_size)
    elif config.chunking_strategy == ChunkingStrategy.CODE_AWARE:
        texts = _chunk_code_aware(content, config.chunk_size)
    else:
        texts = _chunk_fixed_size(content, config.chunk_size, config.chunk_overlap)

    # For markdown-aware chunks, compute section_path per-chunk using a
    # heading stack. This avoids the duplicate-heading-name collision that
    # a global dict[heading_text -> path] would have.
    use_section_paths = config.chunking_strategy == ChunkingStrategy.MARKDOWN_AWARE
    heading_stack: list[tuple[int, str]] = []  # (level, heading_text)

    # Track the active section name and path for continuation chunks
    active_section = ""
    active_section_path = ""

    chunks = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        section = _extract_section(text)
        section_path = ""
        if use_section_paths:
            if section:
                # Chunk starts with a heading — update the stack
                level = _extract_heading_level(text)
                if level > 0:
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, section))
                    section_path = " > ".join(h for _, h in heading_stack)
                active_section = section
                active_section_path = section_path
            else:
                # Continuation chunk (no heading) — inherit from active section
                section = active_section
                section_path = active_section_path
        chunks.append(
            DocumentChunk(
                text=text,
                metadata=DocumentMetadata(
                    source=source,
                    format=format,
                    scope=scope,
                    section=section,
                    section_path=section_path,
                    chunk_index=i,
                    total_chunks=len(texts),
                    ingested_at=ingested_at,
                    document_hash=document_hash,
                ),
            )
        )
    return chunks


def _chunk_fixed_size(content: str, size: int, overlap: int) -> list[str]:
    """Fixed-size chunking with overlap."""
    chunks = []
    start = 0
    while start < len(content):
        end = start + size
        chunks.append(content[start:end])
        start = end - overlap
        if start >= len(content):
            break
    return chunks


def _chunk_by_paragraph(content: str, max_size: int) -> list[str]:
    """Chunk by paragraph boundaries, merging small paragraphs."""
    paragraphs = content.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= max_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para[:max_size] if len(para) > max_size else para
    if current:
        chunks.append(current)
    return chunks


def _chunk_code_aware(content: str, max_size: int) -> list[str]:
    """Code-aware chunking: split on function/class boundaries."""
    import re

    # Split on function/class definitions
    pattern = r"(?=\n(?:def |class |function |async function |export ))"
    parts = re.split(pattern, content)
    chunks = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= max_size:
            current += part
        else:
            if current:
                chunks.append(current)
            current = part[:max_size] if len(part) > max_size else part
    if current:
        chunks.append(current)
    return chunks or [content[:max_size]]


def _chunk_markdown_aware(content: str, max_size: int) -> list[str]:
    """Markdown-aware chunking (W1 #14): split on structural boundaries.

    Break-point priority: heading > code fence > paragraph > list item > newline.
    Preserves semantic units for documentation-heavy corpora.

    Falls back to fixed-size splitting when no markdown structure is detected
    (i.e., content has no headings and no code fences — only plain paragraphs).
    """
    import re

    # Phase 1: Parse content into typed blocks.
    # A "section" starts at each heading and includes all content until the next heading.
    # Code fences are standalone blocks. Paragraph groups (between empty lines) are blocks.
    #
    # Block types: "heading_section" (strong boundary — never merge across),
    #              "code_fence", "paragraph"
    blocks: list[tuple[str, str]] = []  # (type, content)
    lines = content.split("\n")
    current_lines: list[str] = []
    current_type = "paragraph"
    in_code_fence = False

    for line in lines:
        stripped = line.strip()

        # Track code fence state
        if stripped.startswith("```"):
            if in_code_fence:
                # Closing fence — finish the code block
                current_lines.append(line)
                blocks.append(("code_fence", "\n".join(current_lines)))
                current_lines = []
                current_type = "paragraph"
                in_code_fence = False
                continue
            else:
                # Opening fence — flush current block first
                if current_lines:
                    blocks.append((current_type, "\n".join(current_lines)))
                    current_lines = []
                current_lines.append(line)
                current_type = "code_fence"
                in_code_fence = True
                continue

        if in_code_fence:
            current_lines.append(line)
            continue

        # Heading → strong boundary: starts a new heading_section
        if re.match(r"^#{1,6}\s+", stripped):
            if current_lines:
                blocks.append((current_type, "\n".join(current_lines)))
                current_lines = []
            current_lines.append(line)
            current_type = "heading_section"
            continue

        # Empty line → paragraph boundary (weaker)
        if not stripped:
            if current_lines:
                blocks.append((current_type, "\n".join(current_lines)))
                current_lines = []
                current_type = "paragraph"
            continue

        current_lines.append(line)

    # Flush remaining
    if current_lines:
        blocks.append((current_type, "\n".join(current_lines)))

    # Fall back to fixed-size if content has no markdown structure.
    # "No markdown structure" = no heading_section blocks AND no code_fence blocks.
    # Pure paragraph blocks mean plain text — markdown-aware adds no value.
    has_markdown = any(bt in ("heading_section", "code_fence") for bt, _ in blocks)
    if not blocks or not has_markdown:
        return _chunk_fixed_size(content, max_size, max_size // 8)

    # Phase 2: Filter and merge blocks.
    # - Drop horizontal-rule-only blocks ("---") — visual separators, not content.
    # - heading_section → strong boundary: always starts a new chunk.
    #   But heading always absorbs following weak-boundary blocks (paragraph,
    #   code_fence) until max_size — heading is never emitted alone if content follows.
    # - code_fence/paragraph → weak boundary: merge into current chunk if fits.

    # Filter out horizontal rules
    blocks = [(bt, txt) for bt, txt in blocks if txt.strip() not in ("---", "----", "-----")]

    chunks: list[str] = []
    current = ""
    # Track whether the current accumulator starts with a heading.
    # If it does and the next block doesn't fit, the heading must stay
    # attached to the first part of the next block — never emitted alone.
    current_is_heading_only = False

    for i, (block_type, block_text) in enumerate(blocks):
        block_text = block_text.strip()
        if not block_text:
            continue

        # Strong boundary: heading_section always starts a new chunk.
        # Invariant: the heading line is NEVER emitted as a standalone
        # chunk — it must always be attached to body content.
        if block_type == "heading_section":
            if current:
                chunks.append(current)
            if len(block_text) > max_size:
                # Oversized heading section. Extract the heading line and
                # split the body, keeping the heading attached to the
                # first body sub-chunk.
                first_nl = block_text.find("\n")
                if first_nl > 0:
                    heading_line = block_text[:first_nl]
                    body = block_text[first_nl + 1:].lstrip("\n")
                    heading_prefix = heading_line + "\n"
                    body_budget = max(max_size - len(heading_prefix), 1)
                    body_sub = _split_oversized_block(body, body_budget)
                    if body_sub:
                        body_sub[0] = heading_prefix + body_sub[0]
                    else:
                        body_sub = [heading_prefix.strip()]
                    chunks.extend(body_sub[:-1])
                    current = body_sub[-1] if body_sub else ""
                else:
                    # Single-line heading that exceeds max_size — hard-split
                    sub_chunks = _split_oversized_block(block_text, max_size)
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""
                current_is_heading_only = False
            else:
                current = block_text
                current_is_heading_only = "\n" not in block_text.strip()
            continue

        # If current is a heading-only stub, force-attach this block as
        # its body content — even if it exceeds max_size. The heading
        # must remain attached to the first content chunk, never standalone.
        if current_is_heading_only:
            combined = f"{current}\n\n{block_text}"
            if len(combined) <= max_size:
                current = combined
            else:
                # Split the BODY only, then prepend the heading to the
                # first body sub-chunk. This guarantees the heading is
                # never emitted alone, even when the body is a single
                # line longer than max_size.
                heading_prefix = current + "\n\n"
                body_budget = max(max_size - len(heading_prefix), 1)
                body_sub = _split_oversized_block(block_text, body_budget)
                if body_sub:
                    body_sub[0] = heading_prefix + body_sub[0]
                else:
                    body_sub = [heading_prefix.strip()]
                chunks.extend(body_sub[:-1])
                current = body_sub[-1] if body_sub else ""
            current_is_heading_only = False
            continue

        # If the block itself exceeds max_size, split it
        if len(block_text) > max_size:
            if current:
                chunks.append(current)
                current = ""
            sub_chunks = _split_oversized_block(block_text, max_size)
            chunks.extend(sub_chunks)
            current_is_heading_only = False
            continue

        # Try to merge with current accumulator
        separator = "\n\n"
        merged_len = len(current) + len(separator) + len(block_text) if current else len(block_text)
        if merged_len <= max_size:
            current = f"{current}{separator}{block_text}" if current else block_text
        else:
            if current:
                chunks.append(current)
            current = block_text

    if current:
        chunks.append(current)

    return chunks


def _split_oversized_block(block: str, max_size: int) -> list[str]:
    """Split an oversized block on internal boundaries (newline, list items).

    Used by _chunk_markdown_aware when a single structural block exceeds max_size.
    """
    lines = block.split("\n")
    chunks: list[str] = []
    current = ""

    for line in lines:
        if not current:
            current = line
        elif len(current) + 1 + len(line) <= max_size:
            current = f"{current}\n{line}"
        else:
            chunks.append(current)
            current = line

    if current:
        chunks.append(current)

    # Final safety: if any chunk still exceeds max_size, hard-split
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_size:
            result.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                result.append(chunk[start : start + max_size])
                start += max_size

    return result


def _build_section_paths(content: str) -> dict[str, str]:
    """Build heading hierarchy paths from document content (W1.5 #25).

    Scans all headings in the document and builds a breadcrumb path for each.
    Returns a dict mapping heading text → full path (e.g., "Auth > Token Mgmt").

    Heading levels (H1-H6) define the hierarchy:
    - H1 resets the entire path
    - H2 under H1 produces "H1 > H2"
    - H3 under H2 produces "H1 > H2 > H3"
    - A heading at the same or higher level pops the stack
    """
    import re

    paths: dict[str, str] = {}
    # Stack of (level, heading_text) tracking current hierarchy
    stack: list[tuple[int, str]] = []

    for line in content.split("\n"):
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if not match:
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()

        # Pop headings at same or deeper level (new sibling or parent)
        while stack and stack[-1][0] >= level:
            stack.pop()

        stack.append((level, heading))

        # Build path from stack
        path = " > ".join(h for _, h in stack)
        paths[heading] = path

    return paths


def _extract_section(text: str) -> str:
    """Extract section heading from chunk text (H1-H6)."""
    import re

    match = re.search(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_heading_level(text: str) -> int:
    """Extract heading level (1-6) from the first heading in chunk text. 0 if none."""
    import re

    match = re.search(r"^(#{1,6})\s+", text, re.MULTILINE)
    return len(match.group(1)) if match else 0


def build_citations(results: list[dict[str, Any]]) -> list[SearchResult]:
    """Build search results with citations from Qdrant payload (#102, ADR-011)."""
    search_results = []
    for r in results:
        payload = r.get("payload", {})
        metadata = payload.get("metadata", {})
        search_results.append(
            SearchResult(
                text=payload.get("text", r.get("text", "")),
                score=r.get("score", 0.0),
                scope=metadata.get("scope", ""),
                citation=Citation(
                    source=metadata.get("source", ""),
                    section=metadata.get("section", ""),
                    section_path=metadata.get("section_path", ""),
                    page=metadata.get("page"),
                    chunk_index=metadata.get("chunk_index", 0),
                    score=r.get("score", 0.0),
                    ingested_at=metadata.get("ingested_at", 0.0),
                    document_hash=metadata.get("document_hash", ""),
                ),
            )
        )
    return search_results


def format_citations(results: list[SearchResult]) -> str:
    """Format citations as markdown references for LLM context."""
    if not results:
        return ""
    lines = ["## Sources"]
    for i, r in enumerate(results, 1):
        c = r.citation
        ref = f"[{i}] {c.source}"
        if c.section:
            ref += f", {c.section}"
        if c.page is not None:
            ref += f", p.{c.page}"
        ref += f" (score: {c.score:.2f})"
        lines.append(ref)
    return "\n".join(lines)


def compute_chunk_hash(text: str) -> str:
    """MD5 hash for dedup/incremental indexing (#101)."""
    return hashlib.md5(text.encode()).hexdigest()


def compute_document_hash(content: str) -> str:
    """SHA-256 hash of full document content (ADR-011 §1)."""
    return hashlib.sha256(content.encode()).hexdigest()


def compute_point_id(scope: str, source: str, chunk_index: int) -> int:
    """Composite provenance-safe Qdrant point ID (ADR-011 §2).

    Deterministic integer from SHA-256 of scope:source:chunk_index.
    Avoids collisions for identical text across different documents/scopes.
    """
    composite = f"{scope}:{source}:{chunk_index}"
    return int(hashlib.sha256(composite.encode()).hexdigest()[:16], 16)


class KnowledgeService:
    """Manages knowledge scopes and provides ingestion/search operations.

    In production, connects to Qdrant for vector storage.
    For testing, uses in-memory storage.
    Document lifecycle persisted via DocumentStore (#303).
    """

    def __init__(self) -> None:
        self._scopes: dict[str, KnowledgeScope] = {}
        self._chunks: dict[str, list[DocumentChunk]] = {}  # scope → chunks
        self._chunk_hashes: dict[str, set[str]] = {}  # scope → seen hashes (#101 dedup)
        self._qdrant: "QdrantBackend | None" = (
            None  # injected by get_knowledge_service()
        )
        self._doc_store: "DocumentStore | None" = None  # injected by get_knowledge_service()
        # W3a: per-scope query expansion allowlist. Only scopes in this set
        # may use query expansion. Populated from QUERY_EXPANSION_SCOPES env
        # var (comma-separated) or via enable_query_expansion() method.
        self._expansion_allowed_scopes: set[str] = set()
        _env_scopes = os.getenv("QUERY_EXPANSION_SCOPES", "")
        if _env_scopes:
            self._expansion_allowed_scopes = {s.strip() for s in _env_scopes.split(",") if s.strip()}

        # W3b: per-scope HyDE allowlist. Only scopes in this set may use HyDE.
        # Populated from HYDE_SCOPES env var (comma-separated) or via
        # enable_hyde() method.
        self._hyde_allowed_scopes: set[str] = set()
        _env_hyde = os.getenv("HYDE_SCOPES", "")
        if _env_hyde:
            self._hyde_allowed_scopes = {s.strip() for s in _env_hyde.split(",") if s.strip()}

        # W4: per-scope reranking allowlist. Only scopes in this set may use
        # LLM listwise reranking. Populated from RERANK_SCOPES env var
        # (comma-separated) or via enable_reranking() method.
        self._reranking_allowed_scopes: set[str] = set()
        _env_rerank = os.getenv("RERANK_SCOPES", "")
        if _env_rerank:
            self._reranking_allowed_scopes = {s.strip() for s in _env_rerank.split(",") if s.strip()}

    def enable_query_expansion(self, scope: str) -> None:
        """Enable query expansion for a specific scope (W3a #16)."""
        self._expansion_allowed_scopes.add(scope)
        logger.info("W3a: query expansion enabled for scope '%s'", scope)

    def disable_query_expansion(self, scope: str) -> None:
        """Disable query expansion for a specific scope (W3a #16)."""
        self._expansion_allowed_scopes.discard(scope)
        logger.info("W3a: query expansion disabled for scope '%s'", scope)

    def is_expansion_allowed(self, scopes: list[str]) -> bool:
        """Check if query expansion is allowed for any of the given scopes."""
        return any(s in self._expansion_allowed_scopes for s in scopes)

    def enable_hyde(self, scope: str) -> None:
        """Enable HyDE for a specific scope (W3b #17)."""
        self._hyde_allowed_scopes.add(scope)
        logger.info("W3b: HyDE enabled for scope '%s'", scope)

    def disable_hyde(self, scope: str) -> None:
        """Disable HyDE for a specific scope (W3b #17)."""
        self._hyde_allowed_scopes.discard(scope)
        logger.info("W3b: HyDE disabled for scope '%s'", scope)

    def is_hyde_allowed(self, scopes: list[str]) -> bool:
        """Check if HyDE is allowed for any of the given scopes."""
        return any(s in self._hyde_allowed_scopes for s in scopes)

    def enable_reranking(self, scope: str) -> None:
        """Enable LLM listwise reranking for a specific scope (W4 #18)."""
        self._reranking_allowed_scopes.add(scope)
        logger.info("W4: reranking enabled for scope '%s'", scope)

    def disable_reranking(self, scope: str) -> None:
        """Disable LLM listwise reranking for a specific scope (W4 #18)."""
        self._reranking_allowed_scopes.discard(scope)
        logger.info("W4: reranking disabled for scope '%s'", scope)

    def is_reranking_allowed(self, scopes: list[str]) -> bool:
        """Check if reranking is allowed for any of the given scopes."""
        return any(s in self._reranking_allowed_scopes for s in scopes)

    def ingest(
        self,
        scope: str,
        content: str,
        source: str,
        format: DocumentFormat = DocumentFormat.TEXT,
        config: IngestConfig | None = None,
    ) -> IngestResult:
        """Ingest a document with lifecycle management (#303, ADR-011/012).

        Same-hash re-upload → return unchanged, no churn.
        Different-hash re-upload → two-phase replace:
          Phase 1 (prepare): compute new chunks
          Phase 2 (commit): supersede old record, delete old chunks, upsert new, create new record
        """
        config = config or IngestConfig()
        doc_hash = compute_document_hash(content)
        ingested_at = time.time()

        # ── Same-hash short circuit (ADR-012 §2) ────────────────────
        if self._doc_store:
            existing = self._doc_store.get_active(scope, source)
            if existing and existing.document_hash == doc_hash:
                logger.info(
                    "Same-hash re-upload for '%s' in '%s' — unchanged",
                    source, scope,
                )
                return IngestResult(
                    scope=scope,
                    source=source,
                    chunks_created=0,
                    chunks_skipped=0,
                    format=format,
                    document_hash=doc_hash,
                    ingested_at=existing.ingested_at,
                )

        # ── Phase 1: Prepare — compute new chunks ───────────────────
        raw_chunks = chunk_document(
            content, source, scope, format, config,
            document_hash=doc_hash, ingested_at=ingested_at,
        )

        if scope not in self._scopes:
            self._scopes[scope] = KnowledgeScope(name=scope)
        if scope not in self._chunks:
            self._chunks[scope] = []

        # Dedup within scope
        seen = self._chunk_hashes.setdefault(scope, set())
        chunks = []
        for chunk in raw_chunks:
            h = compute_chunk_hash(chunk.text)
            if h not in seen:
                seen.add(h)
                chunks.append(chunk)

        # ── Phase 2: Commit — atomic swap of active version ──────────
        # Contract (ADR-012): old document remains authoritative until commit succeeds.
        # replace_active() atomically supersedes old + creates new in one transaction.
        # If it fails, old active record stays untouched.
        is_replace = False
        if self._doc_store:
            existing = self._doc_store.get_active(scope, source)
            if existing:
                is_replace = True

                new_record = DocumentRecord(
                    scope=scope,
                    source=source,
                    document_hash=doc_hash,
                    format=format,
                    chunk_count=len(raw_chunks),
                    ingested_at=ingested_at,
                    status=DocumentRecordStatus.ACTIVE,
                )
                # Atomic commit: old stays active until this succeeds
                self._doc_store.replace_active(scope, source, new_record)

                # Commit succeeded — now safe to clean up old chunks
                if self._qdrant:
                    try:
                        self._qdrant.delete_by_source(scope, source)
                    except Exception as exc:
                        logger.warning("Qdrant delete_by_source failed during replace: %s", exc)
                # Remove old chunks from memory
                self._chunks[scope] = [
                    c for c in self._chunks[scope] if c.metadata.source != source
                ]
                # Reset dedup hashes for this scope
                remaining_hashes = {
                    compute_chunk_hash(c.text) for c in self._chunks.get(scope, [])
                }
                self._chunk_hashes[scope] = remaining_hashes
                # Re-compute new chunks against clean dedup state
                seen = self._chunk_hashes[scope]
                chunks = []
                for chunk in raw_chunks:
                    h = compute_chunk_hash(chunk.text)
                    if h not in seen:
                        seen.add(h)
                        chunks.append(chunk)

        # Add new chunks to memory
        self._chunks[scope].extend(chunks)
        if is_replace:
            self._scopes[scope].chunk_count = len(self._chunks.get(scope, []))
        else:
            self._scopes[scope].document_count += 1
            self._scopes[scope].chunk_count += len(chunks)
        self._scopes[scope].last_indexed = time.time()

        # Upsert new chunks to Qdrant
        if self._qdrant and chunks:
            try:
                self._qdrant.ingest_chunks(scope, chunks)
            except Exception as exc:
                logger.warning("Qdrant ingest failed for scope '%s': %s", scope, exc)

        # First ingest (not replace): create new active record
        if self._doc_store and not is_replace:
            self._doc_store.create(DocumentRecord(
                scope=scope,
                source=source,
                document_hash=doc_hash,
                format=format,
                chunk_count=len(chunks),
                ingested_at=ingested_at,
                status=DocumentRecordStatus.ACTIVE,
            ))

        action = "Replaced" if is_replace else "Ingested"
        skipped = len(raw_chunks) - len(chunks)
        logger.info(
            "%s %d chunks from '%s' into scope '%s' (%d deduped/skipped)",
            action, len(chunks), source, scope, skipped,
        )
        return IngestResult(
            scope=scope,
            source=source,
            chunks_created=len(chunks),
            chunks_skipped=skipped,
            format=format,
            document_hash=doc_hash,
            ingested_at=ingested_at,
        )

    def ingest_from_orchestrator(
        self,
        scope: str,
        request: OrchestratorIngestRequest,
    ) -> OrchestratorIngestResponse:
        """Ingest pre-parsed text from agentopia-knowledge-ingest Orchestrator.

        Source key model
        ----------------
        source = f"orchestrator:{doc_id}" — stable per logical document, version-agnostic.
        This is the identity key used in document_records.  Version is stored separately
        in document_records.metadata["version"].

        Prior version detection
        -----------------------
        get_active(scope, source) finds the existing ACTIVE record for this logical
        document (regardless of version). Its metadata["version"] field gives the prior
        version number that needs to be superseded in Qdrant.

        Idempotency on (document_id, version)
        --------------------------------------
        Before embedding, check if the active record's stored version == request.version.
        If so, the same version is already indexed → return without re-embedding (skipped).

        Replacement flow
        ----------------
        1. Chunk + embed new version (new chunks, document_id, version=N, status=active)
        2. Update document_records: replace_active supersedes old record, creates new one
        3. Supersede Qdrant chunks for the prior version (payload update, no re-embed)
        """
        doc_id = request.document_id
        version = request.version
        ingested_at = time.time()

        # Stable source key — version-agnostic so get_active finds the prior version
        source = f"orchestrator:{doc_id}"

        fmt_map = {
            "pdf": DocumentFormat.PDF,
            "docx": DocumentFormat.TEXT,
            "html": DocumentFormat.HTML,
            "markdown": DocumentFormat.MARKDOWN,
            "md": DocumentFormat.MARKDOWN,
            "txt": DocumentFormat.TEXT,
            "text": DocumentFormat.TEXT,
        }
        fmt = fmt_map.get(request.metadata.format.lower(), DocumentFormat.TEXT)

        config = IngestConfig(chunking_strategy=request.chunking_strategy)
        doc_hash = compute_document_hash(request.text)

        # ── Idempotency check: already indexed this exact version? ────────────
        if self._doc_store:
            existing = self._doc_store.get_active(scope, source)
            if existing:
                stored_version = (existing.metadata or {}).get("version") if hasattr(existing, "metadata") else None
                # Fall back to parsing from source if metadata not present
                if stored_version is None:
                    # Attempt to read version stored as source suffix (legacy)
                    stored_version = None
                if stored_version == version:
                    logger.info(
                        "ingest_from_orchestrator: skipping — scope=%s document_id=%s version=%d already active",
                        scope, doc_id, version,
                    )
                    return OrchestratorIngestResponse(
                        document_id=doc_id,
                        scope=scope,
                        version=version,
                        chunk_count=existing.chunk_count,
                        status="skipped",
                    )

        raw_chunks = chunk_document(
            content=request.text,
            source=source,
            scope=scope,
            format=fmt,
            config=config,
            document_hash=doc_hash,
            ingested_at=ingested_at,
        )

        # Propagate section_path from extracted metadata for non-markdown strategies
        if request.metadata.section_path and config.chunking_strategy != ChunkingStrategy.MARKDOWN_AWARE:
            path_str = " > ".join(request.metadata.section_path)
            for chunk in raw_chunks:
                if not chunk.metadata.section_path:
                    chunk.metadata.section_path = path_str

        chunk_count = 0

        # Detect prior version BEFORE upsert/storage so we can supersede after commit
        prior_version: int | None = None
        if self._doc_store:
            existing = self._doc_store.get_active(scope, source)
            if existing:
                prior_meta = getattr(existing, "metadata", None) or {}
                prior_version = prior_meta.get("version")

        if self._qdrant:

            # Upsert new version chunks (status=active in payload)
            chunk_count = self._qdrant.ingest_chunks_versioned(
                scope=scope,
                chunks=raw_chunks,
                document_id=doc_id,
                version=version,
            )

            # Supersede prior version in Qdrant (if any) — after new chunks are committed
            if prior_version is not None and prior_version != version:
                try:
                    self._qdrant.supersede_document_version(scope, doc_id, prior_version)
                except Exception as exc:
                    logger.warning(
                        "Failed to supersede prior version %d for document_id=%s: %s",
                        prior_version, doc_id, exc,
                    )
        else:
            # In-memory fallback (dev/test without Qdrant)
            if scope not in self._chunks:
                self._chunks[scope] = []
            # Remove prior version chunks for this document_id (in-memory replacement)
            self._chunks[scope] = [
                c for c in self._chunks[scope]
                if not (hasattr(c.metadata, "document_hash") and c.metadata.source == source)
            ]
            self._chunks[scope].extend(raw_chunks)
            chunk_count = len(raw_chunks)

        # Update document_records with stable source and version stored in metadata
        existing_record = None
        if self._doc_store:
            new_record = DocumentRecord(
                scope=scope,
                source=source,
                document_hash=doc_hash,
                format=fmt,
                chunk_count=chunk_count,
                ingested_at=ingested_at,
                status=DocumentRecordStatus.ACTIVE,
                metadata={"version": version},
            )
            existing_record = self._doc_store.get_active(scope, source)
            if existing_record:
                self._doc_store.replace_active(scope, source, new_record)
            else:
                self._doc_store.create(new_record)

        # Update in-memory scope metadata — must maintain document_count and chunk_count
        # so list_scopes() reflects the actual indexed state.
        is_replacement = existing_record is not None

        if scope not in self._scopes:
            self._scopes[scope] = KnowledgeScope(name=scope)
        if is_replacement:
            # Replace: document_count stays the same, chunk_count adjusts
            old_count = existing_record.chunk_count if existing_record else 0
            self._scopes[scope].chunk_count += chunk_count - old_count
        else:
            # New document: increment both
            self._scopes[scope].document_count += 1
            self._scopes[scope].chunk_count += chunk_count
        self._scopes[scope].last_indexed = time.time()

        logger.info(
            "ingest_from_orchestrator: scope=%s document_id=%s version=%d chunks=%d doc_count=%d chunk_count=%d",
            scope, doc_id, version, chunk_count,
            self._scopes[scope].document_count, self._scopes[scope].chunk_count,
        )

        # ── Regression check on replacement ──────────────────────────────────
        # Only triggered when a prior version was superseded.
        # Runs after commit so retrieval is never blocked.
        # Failures are logged but never propagate — ingest is already complete.
        version_superseded = prior_version is not None and prior_version != version
        if version_superseded:
            try:
                from services.evaluation import check_regression
                reg_result = check_regression(
                    scope=scope,
                    svc=self,
                    trigger="replacement",
                    document_id=doc_id,
                    document_version=version,
                )
                if reg_result.verdict == "blocked":
                    logger.warning(
                        "ingest_from_orchestrator: REGRESSION BLOCKED scope=%s "
                        "document_id=%s version=%d ndcg_5=%.4f delta=%+.4f result_id=%s",
                        scope, doc_id, version,
                        reg_result.ndcg_5 or 0.0,
                        reg_result.delta_ndcg_5 or 0.0,
                        reg_result.result_id,
                    )
            except Exception as exc:
                logger.warning(
                    "ingest_from_orchestrator: regression check failed scope=%s: %s "
                    "(document is active — evaluation failure does not roll back ingest)",
                    scope, exc,
                )

        return OrchestratorIngestResponse(
            document_id=doc_id,
            scope=scope,
            version=version,
            chunk_count=chunk_count,
            status="indexed",
        )

    def search(
        self,
        query: str,
        scopes: list[str],
        limit: int = 5,
        min_score: float = 0.0,
        query_expansion_enabled: bool = False,
        query_expansion_n: int = 3,
        hyde_enabled: bool = False,
        rerank_enabled: bool = False,
        rerank_candidate_k: int = 0,
    ) -> list[SearchResult]:
        """Search across knowledge scopes.

        Routes to Qdrant vector similarity search when backend is configured (#23),
        otherwise falls back to in-memory text matching (tests / local dev).
        Emits per-result provenance audit log (ADR-011 §5).

        When query_expansion_enabled=True (W3a #16):
        1. Generate N alternative phrasings via LLM
        2. Run retrieval for original + N expansions
        3. Merge results via RRF (Reciprocal Rank Fusion)
        4. On LLM failure, fall back silently to dense-only search

        When hyde_enabled=True (W3b #17):
        1. Generate one hypothetical answer via LLM
        2. Retrieve using the hypothesis embedding
        3. Merge HyDE results with original query results via RRF
        4. On LLM failure, fall back silently to dense-only search
        Per-scope allowlist enforced — hyde_enabled=True ignored for non-allowed scopes.
        W3a and W3b are independent; enabling both is not supported without combined eval.

        When rerank_enabled=True (W4 #18):
        1. Retrieve candidate_k candidates from Qdrant (candidate_k > limit)
        2. Rerank via LLM listwise scoring (single LLM call for all candidates)
        3. Return top-limit from reranked order
        4. On LLM failure, fall back silently to vector-ranked order
        Per-scope allowlist enforced. Cannot be combined with W3a or W3b without
        a combined evaluation run.
        """
        if query_expansion_enabled and hyde_enabled:
            raise ValueError("query_expansion and hyde cannot both be enabled")
        if rerank_enabled and query_expansion_enabled:
            raise ValueError("rerank and query_expansion cannot both be enabled")
        if rerank_enabled and hyde_enabled:
            raise ValueError("rerank and hyde cannot both be enabled")

        # Qdrant path: fan-out across scopes, merge, sort, limit
        if self._qdrant:
            if query_expansion_enabled:
                # W3a: split scopes into allowed (expansion) vs blocked (dense-only)
                allowed = [s for s in scopes if s in self._expansion_allowed_scopes]
                blocked = [s for s in scopes if s not in self._expansion_allowed_scopes]

                ranked_lists: list[list[SearchResult]] = []

                # Expanded retrieval for allowed scopes only
                if allowed:
                    ranked_lists.append(
                        self._search_with_expansion(
                            query, allowed, limit, min_score, query_expansion_n
                        )
                    )

                # Dense-only retrieval for blocked scopes
                if blocked:
                    ranked_lists.append(
                        self._search_dense_across_scopes(
                            query, blocked, limit, min_score
                        )
                    )

                if blocked:
                    logger.info(
                        "W3a: expansion applied to %s, dense-only for %s",
                        allowed, blocked,
                    )

                final = self._merge_ranked_search_results(ranked_lists, limit)
                self._audit_search_results(final)
                return final

            if hyde_enabled:
                # W3b: split scopes into allowed (HyDE) vs blocked (dense-only)
                hyde_allowed = [s for s in scopes if s in self._hyde_allowed_scopes]
                hyde_blocked = [s for s in scopes if s not in self._hyde_allowed_scopes]

                ranked_lists: list[list[SearchResult]] = []

                if hyde_allowed:
                    ranked_lists.append(
                        self._search_with_hyde(query, hyde_allowed, limit, min_score)
                    )

                if hyde_blocked:
                    ranked_lists.append(
                        self._search_dense_across_scopes(
                            query, hyde_blocked, limit, min_score
                        )
                    )

                if hyde_blocked:
                    logger.info(
                        "W3b: HyDE applied to %s, dense-only for %s",
                        hyde_allowed, hyde_blocked,
                    )

                final = self._merge_ranked_search_results(ranked_lists, limit)
                self._audit_search_results(final)
                return final

            if rerank_enabled:
                # W4: dense retrieve candidate_k, rerank, return top-limit
                rerank_allowed = [s for s in scopes if s in self._reranking_allowed_scopes]
                rerank_blocked = [s for s in scopes if s not in self._reranking_allowed_scopes]

                if rerank_allowed:
                    from services.reranker import DEFAULT_CANDIDATE_K
                    candidate_k = rerank_candidate_k or DEFAULT_CANDIDATE_K
                    # Retrieve more candidates than needed for reranking
                    candidates = self._search_dense_across_scopes(
                        query, rerank_allowed, max(candidate_k, limit), min_score
                    )
                    reranked_results = self._rerank_search_results(query, candidates, limit)
                else:
                    reranked_results = []

                # Dense-only for blocked scopes
                dense_results = (
                    self._search_dense_across_scopes(query, rerank_blocked, limit, min_score)
                    if rerank_blocked
                    else []
                )

                if rerank_blocked:
                    logger.info(
                        "W4: reranking applied to %s, dense-only for %s",
                        rerank_allowed, rerank_blocked,
                    )

                # Merge allowed (reranked) and blocked (dense) via RRF
                ranked_lists: list[list[SearchResult]] = []
                if reranked_results:
                    ranked_lists.append(reranked_results)
                if dense_results:
                    ranked_lists.append(dense_results)

                final = self._merge_ranked_search_results(ranked_lists, limit)
                self._audit_search_results(final)
                return final

            all_results_dense = self._search_dense_across_scopes(
                query, scopes, limit, min_score
            )
            final = all_results_dense[:limit]
            self._audit_search_results(final)
            return final

        # In-memory fallback path
        results = []
        query_lower = query.lower()
        for scope in scopes:
            for chunk in self._chunks.get(scope, []):
                # Simple text matching score (production: cosine similarity)
                text_lower = chunk.text.lower()
                if query_lower in text_lower:
                    score = 1.0
                else:
                    words = query_lower.split()
                    matched = sum(1 for w in words if w in text_lower)
                    score = matched / len(words) if words else 0.0
                if score > 0:
                    results.append(
                        SearchResult(
                            text=chunk.text[:500],
                            score=score,
                            scope=scope,
                            citation=Citation(
                                source=chunk.metadata.source,
                                section=chunk.metadata.section,
                                page=chunk.metadata.page,
                                chunk_index=chunk.metadata.chunk_index,
                                score=score,
                                ingested_at=chunk.metadata.ingested_at,
                                document_hash=chunk.metadata.document_hash,
                            ),
                        )
                    )
        results.sort(key=lambda r: r.score, reverse=True)
        final = results[:limit]
        self._audit_search_results(final)
        return final

    def _search_dense_across_scopes(
        self,
        query: str,
        scopes: list[str],
        limit: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Dense-only Qdrant search across scopes, sorted by backend score."""
        if not self._qdrant:
            return []

        results: list[SearchResult] = []
        for scope in scopes:
            try:
                results.extend(
                    self._qdrant.search_scope(query, scope, limit, min_score=min_score)
                )
            except Exception as exc:
                logger.warning("Qdrant search failed for scope '%s': %s", scope, exc)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    @staticmethod
    def _search_results_to_ranked_dicts(results: list[SearchResult]) -> list[dict[str, Any]]:
        """Convert SearchResult objects into rank-fusion input dictionaries."""
        ranked: list[dict[str, Any]] = []
        for sr in results:
            ranked.append({
                "text": sr.text,
                "score": sr.score,
                "scope": sr.scope,
                "citation": {
                    "source": sr.citation.source,
                    "section": sr.citation.section,
                    "section_path": sr.citation.section_path,
                    "page": sr.citation.page,
                    "chunk_index": sr.citation.chunk_index,
                    "ingested_at": sr.citation.ingested_at,
                    "document_hash": sr.citation.document_hash,
                },
            })
        return ranked

    @staticmethod
    def _ranked_dicts_to_search_results(results: list[dict[str, Any]]) -> list[SearchResult]:
        """Convert rank-fusion dictionaries back into SearchResult objects."""
        final: list[SearchResult] = []
        for result in results:
            cit = result.get("citation", {})
            final.append(SearchResult(
                text=result.get("text", ""),
                score=result.get("score", 0.0),
                scope=result.get("scope", ""),
                citation=Citation(
                    source=cit.get("source", ""),
                    section=cit.get("section", ""),
                    section_path=cit.get("section_path", ""),
                    page=cit.get("page"),
                    chunk_index=cit.get("chunk_index", 0),
                    score=result.get("score", 0.0),
                    ingested_at=cit.get("ingested_at", 0.0),
                    document_hash=cit.get("document_hash", ""),
                ),
            ))
        return final

    def _merge_ranked_search_results(
        self,
        ranked_lists: list[list[SearchResult]],
        limit: int,
    ) -> list[SearchResult]:
        """Fuse already-ranked result lists in one coherent rank space."""
        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return ranked_lists[0][:limit]

        from services.query_expansion import rrf_merge

        merged = rrf_merge(
            [
                self._search_results_to_ranked_dicts(results)
                for results in ranked_lists
                if results
            ],
            limit,
        )
        return self._ranked_dicts_to_search_results(merged)

    def _search_with_expansion(
        self,
        query: str,
        scopes: list[str],
        limit: int,
        min_score: float,
        n: int,
    ) -> list[SearchResult]:
        """Search with query expansion (W3a #16).

        1. Generate N alternative phrasings via LLM
        2. Run retrieval for original + N expansions
        3. Merge results via RRF
        4. On LLM failure, fall back to dense-only search
        """
        from services.query_expansion import expand_query, rrf_merge

        t0 = time.monotonic()
        phrasings = expand_query(query, n=n)
        expansion_ms = (time.monotonic() - t0) * 1000

        all_queries = [query] + phrasings
        logger.info(
            "W3a: searching with %d queries (1 original + %d expansions, %.0fms expansion)",
            len(all_queries), len(phrasings), expansion_ms,
        )

        # Run retrieval for each query
        ranked_lists: list[list[dict]] = []
        for q in all_queries:
            results_for_q: list[dict] = []
            for scope in scopes:
                try:
                    sr_list = self._qdrant.search_scope(q, scope, limit, min_score=min_score)
                    for sr in sr_list:
                        results_for_q.append({
                            "text": sr.text,
                            "score": sr.score,
                            "scope": sr.scope,
                            "citation": {
                                "source": sr.citation.source,
                                "section": sr.citation.section,
                                "section_path": sr.citation.section_path,
                                "page": sr.citation.page,
                                "chunk_index": sr.citation.chunk_index,
                                "ingested_at": sr.citation.ingested_at,
                                "document_hash": sr.citation.document_hash,
                            },
                        })
                except Exception as exc:
                    logger.warning("W3a search failed for scope '%s': %s", scope, exc)
            ranked_lists.append(results_for_q)

        # Merge via RRF
        merged = rrf_merge(ranked_lists, limit)

        # Convert back to SearchResult
        final = []
        for r in merged:
            cit = r.get("citation", {})
            final.append(SearchResult(
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                scope=r.get("scope", ""),
                citation=Citation(
                    source=cit.get("source", ""),
                    section=cit.get("section", ""),
                    section_path=cit.get("section_path", ""),
                    page=cit.get("page"),
                    chunk_index=cit.get("chunk_index", 0),
                    score=r.get("score", 0.0),
                    ingested_at=cit.get("ingested_at", 0.0),
                    document_hash=cit.get("document_hash", ""),
                ),
            ))

        total_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "W3a: completed in %.0fms (expansion=%.0fms, retrieval=%.0fms, results=%d)",
            total_ms, expansion_ms, total_ms - expansion_ms, len(final),
        )
        self._audit_search_results(final)
        return final

    def _search_with_hyde(
        self,
        query: str,
        scopes: list[str],
        limit: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Search with HyDE (W3b #17).

        1. Generate one hypothetical answer via LLM
        2. Retrieve using the hypothesis embedding
        3. Merge HyDE results with original query results via RRF
        4. On LLM failure, fall back to dense-only search
        """
        from services.hyde import generate_hypothesis
        from services.query_expansion import rrf_merge

        t0 = time.monotonic()
        hypothesis = generate_hypothesis(query)
        hypothesis_ms = (time.monotonic() - t0) * 1000

        def _retrieve(q: str) -> list[dict]:
            results_for_q: list[dict] = []
            for scope in scopes:
                try:
                    sr_list = self._qdrant.search_scope(q, scope, limit, min_score=min_score)
                    for sr in sr_list:
                        results_for_q.append({
                            "text": sr.text,
                            "score": sr.score,
                            "scope": sr.scope,
                            "citation": {
                                "source": sr.citation.source,
                                "section": sr.citation.section,
                                "section_path": sr.citation.section_path,
                                "page": sr.citation.page,
                                "chunk_index": sr.citation.chunk_index,
                                "ingested_at": sr.citation.ingested_at,
                                "document_hash": sr.citation.document_hash,
                            },
                        })
                except Exception as exc:
                    logger.warning("W3b: search failed for scope '%s': %s", scope, exc)
            return results_for_q

        # Always include original query results
        original_results = _retrieve(query)

        if hypothesis:
            # Merge: [original, HyDE] via RRF
            hyde_results = _retrieve(hypothesis)
            ranked_lists = [original_results, hyde_results]
            logger.info(
                "W3b: RRF merge — original=%d results, HyDE=%d results (%.0fms generation)",
                len(original_results), len(hyde_results), hypothesis_ms,
            )
        else:
            # Fallback: hypothesis generation failed, dense-only
            ranked_lists = [original_results]
            logger.info(
                "W3b: falling back to dense-only (hypothesis generation failed, %.0fms)",
                hypothesis_ms,
            )

        merged = rrf_merge(ranked_lists, limit)

        final = []
        for r in merged:
            cit = r.get("citation", {})
            final.append(SearchResult(
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                scope=r.get("scope", ""),
                citation=Citation(
                    source=cit.get("source", ""),
                    section=cit.get("section", ""),
                    section_path=cit.get("section_path", ""),
                    page=cit.get("page"),
                    chunk_index=cit.get("chunk_index", 0),
                    score=r.get("score", 0.0),
                    ingested_at=cit.get("ingested_at", 0.0),
                    document_hash=cit.get("document_hash", ""),
                ),
            ))

        total_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "W3b: completed in %.0fms (hypothesis=%.0fms, retrieval=%.0fms, results=%d)",
            total_ms, hypothesis_ms, total_ms - hypothesis_ms, len(final),
        )
        self._audit_search_results(final)
        return final

    def _rerank_search_results(
        self,
        query: str,
        candidates: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        """Rerank candidate SearchResults via LLM listwise scoring (W4 #18).

        Converts SearchResult → dicts for reranker, calls rerank_results(),
        overwrites scores to reflect reranked position, converts back.
        On LLM failure, returns top-limit from vector-ranked order with
        coherent rank-based scores.

        Score semantics: score = 1/(1+rank) — rank 0 → 1.0, rank 1 → 0.5,
        rank 2 → 0.333 … Original vector cosine scores are discarded because
        they reflect pre-rerank similarity order, which may no longer match
        the reranked order returned by the LLM.
        """
        from services.reranker import rerank_results

        t0 = time.monotonic()
        candidate_dicts = self._search_results_to_ranked_dicts(candidates)
        reranked_dicts = rerank_results(query, candidate_dicts)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Overwrite scores with rank-based values so SearchResult.score and
        # Citation.score reflect the reranked position, not stale cosine scores.
        # _ranked_dicts_to_search_results() reads d["score"] for both fields.
        top = reranked_dicts[:limit]
        for rank, d in enumerate(top):
            d["score"] = round(1.0 / (1 + rank), 6)

        final = self._ranked_dicts_to_search_results(top)
        logger.info(
            "W4: rerank complete in %.0fms (candidates=%d, returned=%d)",
            elapsed_ms, len(candidates), len(final),
        )
        return final

    @staticmethod
    def _audit_search_results(results: list[SearchResult]) -> None:
        """Emit per-result provenance audit log (ADR-011 §5)."""
        for r in results:
            logger.info(
                "knowledge_search_result: source=%s chunk_index=%d scope=%s "
                "document_hash=%s score=%.4f",
                r.citation.source,
                r.citation.chunk_index,
                r.scope,
                r.citation.document_hash,
                r.score,
            )

    def list_scopes(self) -> list[KnowledgeScope]:
        """List all knowledge scopes with stats (#106)."""
        return list(self._scopes.values())

    def get_scope(self, name: str) -> KnowledgeScope | None:
        """Get a specific scope's metadata."""
        return self._scopes.get(name)

    def delete_scope(self, name: str) -> bool:
        """Delete a knowledge scope and all its chunks (#106, ADR-011, #303)."""
        if name in self._scopes:
            del self._scopes[name]
            self._chunks.pop(name, None)
            self._chunk_hashes.pop(name, None)
            # Delete Qdrant collection (ADR-011 — real deletion)
            if self._qdrant:
                try:
                    self._qdrant.delete_collection(name)
                except Exception as exc:
                    logger.warning(
                        "Qdrant delete_collection failed for '%s': %s", name, exc,
                    )
            # Mark all active documents in scope as deleted (#303)
            if self._doc_store:
                self._doc_store.mark_scope_deleted(name)
            logger.info("Deleted knowledge scope '%s'", name)
            return True
        return False

    def delete_document(self, scope: str, source: str) -> int:
        """Delete all chunks from a specific source document (#106, ADR-011, #303)."""
        if scope not in self._chunks:
            return 0
        before = len(self._chunks[scope])
        self._chunks[scope] = [
            c for c in self._chunks[scope] if c.metadata.source != source
        ]
        removed = before - len(self._chunks[scope])
        if removed > 0 and scope in self._scopes:
            self._scopes[scope].chunk_count -= removed
            self._scopes[scope].document_count = max(
                0, self._scopes[scope].document_count - 1
            )
        # Invalidate deleted chunk hashes so re-ingest is treated as new (#101)
        if scope in self._chunk_hashes and removed > 0:
            remaining_hashes = {
                compute_chunk_hash(c.text) for c in self._chunks.get(scope, [])
            }
            self._chunk_hashes[scope] = remaining_hashes
        # Delete from Qdrant (ADR-011 — real deletion, not orphan)
        if self._qdrant and removed > 0:
            try:
                self._qdrant.delete_by_source(scope, source)
            except Exception as exc:
                logger.warning(
                    "Qdrant delete_by_source failed for '%s/%s': %s",
                    scope, source, exc,
                )
        # Mark document as deleted in lifecycle store (#303)
        if self._doc_store:
            self._doc_store.mark_deleted(scope, source)
        return removed

    def list_documents(self, scope: str) -> list[dict]:
        """List documents in a scope (#106, #303).

        Uses DocumentStore if available (durable, includes provenance).
        Falls back to in-memory chunk scan.
        """
        if self._doc_store:
            records = self._doc_store.list_active(scope)
            return [
                {
                    "source": r.source,
                    "chunk_count": r.chunk_count,
                    "scope": r.scope,
                    "document_hash": r.document_hash,
                    "ingested_at": r.ingested_at,
                    "format": r.format.value,
                }
                for r in records
            ]
        # Fallback: in-memory chunk scan
        chunks = self._chunks.get(scope, [])
        sources: dict[str, int] = {}
        for chunk in chunks:
            src = chunk.metadata.source or ""
            sources[src] = sources.get(src, 0) + 1
        return [
            {"source": src, "chunk_count": count, "scope": scope}
            for src, count in sorted(sources.items())
        ]

    def reindex(self, scope: str) -> dict:
        """Reset chunk hashes so next ingest re-indexes all content (#101).

        Does NOT remove existing chunks — clears the dedup cache so that
        re-submitted content is not silently skipped.
        """
        if scope not in self._scopes:
            return {"scope": scope, "status": "not_found"}
        chunk_count = len(self._chunks.get(scope, []))
        self._chunk_hashes.pop(scope, None)
        logger.info(
            "Reindex triggered for scope '%s': cleared %d chunk hashes",
            scope,
            chunk_count,
        )
        return {
            "scope": scope,
            "status": "reindex_triggered",
            "chunks_cleared_for_reindex": chunk_count,
        }

    def list_stale_scopes(self, max_age_secs: float = 86400.0) -> list[str]:
        """Return names of scopes whose last_indexed exceeds max_age_secs (#101)."""
        return [s.name for s in self._scopes.values() if s.is_stale(max_age_secs)]


class QdrantBackend:
    """Qdrant-backed knowledge storage (#23, #100).

    One Qdrant collection per knowledge scope (collection-per-scope pattern).
    Embedding is done via the configured embedding model.

    Falls back silently if qdrant-client is not installed or QDRANT_URL is unset.
    Use KnowledgeService for all access — do not instantiate directly in tests.
    """

    def __init__(self, url: str, api_key: str = "") -> None:
        import os
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        self._client = QdrantClient(url=url, api_key=api_key or None)
        self._Distance = Distance
        self._VectorParams = VectorParams
        self._collection_cache: set[str] = set()

        # ── Model-variable embedding config (#331) ────────────────────────
        # All embedding parameters are config-driven. Changing model or
        # dimension requires Qdrant collection reindex.
        self.VECTOR_SIZE = int(os.getenv("EMBEDDING_VECTOR_DIMENSION", "1536"))
        self._embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self._embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1/embeddings")
        self._embedding_timeout = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
        self._embedding_api_key = (
            os.getenv("EMBEDDING_API_KEY", "")
            or os.getenv("OPENROUTER_API_KEY", "")
        )

        # Hybrid retrieval config (#319, frozen)
        self._hybrid_enabled = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
        self._hybrid_prefetch_limit = int(os.getenv("HYBRID_PREFETCH_LIMIT", "20"))

        mode = "hybrid (dense+sparse+RRF)" if self._hybrid_enabled else "dense-only"
        logger.info(
            "QdrantBackend: connected to %s (model=%s, dims=%d, timeout=%ds, mode=%s)",
            url, self._embedding_model, self.VECTOR_SIZE, self._embedding_timeout, mode,
        )

        # ── Startup dimension validation ─────────────────────────────────
        # Fail-fast if existing Qdrant collection dimension != configured
        # dimension. This catches model/dimension drift before silent corruption.
        self._validate_collection_dimensions()

    def _validate_collection_dimensions(self) -> None:
        """Check existing Qdrant collections for dimension mismatch (#331).

        If any kb-* collection has a different vector dimension than the
        configured EMBEDDING_VECTOR_DIMENSION, log a warning. This is a
        signal that reindex is required after model/dimension change.

        Does NOT fail-fast on mismatch (allows graceful migration) but
        logs a clear warning for operator visibility.
        """
        try:
            collections = self._client.get_collections().collections
            for col in collections:
                if not col.name.startswith("kb-"):
                    continue
                try:
                    info = self._client.get_collection(col.name)
                    config = info.config
                    params = config.params
                    vectors = params.vectors
                    if isinstance(vectors, dict):
                        # Named vectors (hybrid): check "dense" key
                        dense_cfg = vectors.get("dense")
                        if dense_cfg and dense_cfg.size != self.VECTOR_SIZE:
                            logger.warning(
                                "DIMENSION_MISMATCH: collection '%s' has dense dim=%d, "
                                "config expects dim=%d. Reindex required (#331).",
                                col.name, dense_cfg.size, self.VECTOR_SIZE,
                            )
                    elif hasattr(vectors, "size") and vectors.size != self.VECTOR_SIZE:
                        # Unnamed vector (dense-only)
                        logger.warning(
                            "DIMENSION_MISMATCH: collection '%s' has dim=%d, "
                            "config expects dim=%d. Reindex required (#331).",
                            col.name, vectors.size, self.VECTOR_SIZE,
                        )
                except Exception:
                    pass  # Collection info unavailable — skip
        except Exception as exc:
            logger.debug("Collection dimension validation skipped: %s", exc)

    @staticmethod
    def _qdrant_collection_name(scope: str) -> str:
        """Map canonical scope identity to a collision-safe Qdrant collection name (#327).

        Canonical scope identity uses '/' as the client/scope separator
        (e.g. 'acme-corp/api-docs'). Qdrant collection names must not
        contain '/'.

        We use a SHA-256 hash of the full canonical identity to produce
        a deterministic, collision-resistant physical name:
            kb-{sha256_hex[:16]}

        Examples:
            'acme-corp/api-docs'  → 'kb-<16-hex-chars>'
            'acme/foo--bar'       → 'kb-<different-16-hex-chars>'
            'acme--foo/bar'       → 'kb-<different-16-hex-chars>'

        The old '/' → '--' mapping was NOT injective: 'acme/foo--bar' and
        'acme--foo/bar' both mapped to 'acme--foo--bar'. SHA-256 with 64
        bits (16 hex chars) is collision-resistant for practical scope counts.

        The canonical form (with '/') is always preserved in annotations,
        logs, public API responses, and provenance metadata — only Qdrant
        sees the hashed physical name.
        """
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
        return f"kb-{digest}"

    def health_check(self) -> dict[str, str]:
        """Check Qdrant connectivity. Returns {"status": "ok"} or {"status": "...", "message": "..."}."""
        try:
            collections = self._client.get_collections()
            return {
                "status": "ok",
                "collections": str(len(collections.collections)),
            }
        except Exception as exc:
            return {"status": "unavailable", "message": str(exc)}

    def has_collection(self, scope: str) -> bool:
        """Return True if the Qdrant collection for scope exists and has at least one point."""
        cname = self._qdrant_collection_name(scope)
        try:
            info = self._client.get_collection(cname)
            return (info.points_count or 0) > 0
        except Exception:
            return False

    def _ensure_collection(self, scope: str) -> None:
        """Create Qdrant collection for scope if it does not exist (#100, #319 hybrid)."""
        cname = self._qdrant_collection_name(scope)
        if cname in self._collection_cache:
            return
        try:
            self._client.get_collection(cname)
        except Exception:
            if self._hybrid_enabled:
                from qdrant_client.models import SparseVectorParams, SparseIndexParams  # type: ignore
                self._client.create_collection(
                    collection_name=cname,
                    vectors_config={
                        "dense": self._VectorParams(
                            size=self.VECTOR_SIZE,
                            distance=self._Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                        ),
                    },
                )
                logger.info("QdrantBackend: created hybrid collection '%s' (dense+sparse)", cname)
            else:
                self._client.create_collection(
                    collection_name=cname,
                    vectors_config=self._VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=self._Distance.COSINE,
                    ),
                )
                logger.info("QdrantBackend: created collection '%s' (dense-only)", cname)
        self._collection_cache.add(cname)

    # ── BM25 sparse encoding for hybrid search (#319) ──────────────────────

    # BM25 hyperparameters (Okapi BM25 defaults)
    _bm25_k1: float = 1.2
    _bm25_b: float = 0.75

    # Corpus statistics — rebuilt per-scope at index/search time
    _bm25_stats: dict[str, dict] = {}  # scope → {N, df, avgdl}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text: lowercase, split on non-alphanumeric."""
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _term_id(word: str) -> int:
        """Map word to sparse vector index (stable across processes, int32 range).

        Uses SHA-256 truncated to 31 bits. Python's built-in hash() is
        randomized per process (PYTHONHASHSEED), making it unsuitable for
        sparse vector coordinate identity that must be consistent across
        indexing and querying — which may happen in different processes,
        pods, or after restarts.
        """
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (2**31)

    def _build_bm25_stats(self, scope: str) -> dict:
        """Build BM25 corpus statistics from a Qdrant collection.

        Computes:
          N     — total number of documents
          df    — dict of term → number of documents containing that term
          avgdl — average document length (in tokens)
        """
        cname = self._qdrant_collection_name(scope)
        all_texts: list[str] = []

        # Scroll through all documents in the collection
        offset = None
        while True:
            kwargs: dict = {
                "collection_name": cname,
                "limit": 100,
                "with_payload": True,
                "with_vectors": False,
            }
            if offset is not None:
                kwargs["offset"] = offset
            result = self._client.scroll(**kwargs)
            points, next_offset = result
            for pt in points:
                text = pt.payload.get("text", "") if pt.payload else ""
                all_texts.append(text)
            if next_offset is None:
                break
            offset = next_offset

        N = len(all_texts)
        if N == 0:
            return {"N": 0, "df": {}, "avgdl": 1.0}

        from collections import Counter
        df: Counter = Counter()
        total_length = 0
        for text in all_texts:
            words = self._tokenize(text)
            total_length += len(words)
            unique_terms = set(words)
            for term in unique_terms:
                df[term] += 1

        avgdl = total_length / N
        stats = {"N": N, "df": dict(df), "avgdl": avgdl}
        self._bm25_stats[scope] = stats
        logger.info(
            "BM25 stats built for scope '%s': N=%d, vocab=%d, avgdl=%.1f",
            scope, N, len(df), avgdl,
        )
        return stats

    def _get_bm25_stats(self, scope: str) -> dict:
        """Get or build BM25 corpus statistics for a scope."""
        if scope not in self._bm25_stats:
            self._build_bm25_stats(scope)
        return self._bm25_stats[scope]

    def _bm25_idf(self, term: str, stats: dict) -> float:
        """Compute IDF for a term: log((N - n + 0.5) / (n + 0.5) + 1)."""
        import math
        N = stats["N"]
        n = stats["df"].get(term, 0)
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    def _encode_bm25_document(self, text: str, stats: dict) -> tuple[list[int], list[float]]:
        """Encode a document into a BM25-weighted sparse vector.

        Each term gets weight: IDF(t) * (f(t,d) * (k1+1)) / (f(t,d) + k1*(1-b+b*|d|/avgdl))

        This is the full Okapi BM25 score per term, stored in the sparse vector.
        At query time, query terms use IDF-only weights. Qdrant dot product
        approximates BM25 ranking.
        """
        from collections import Counter

        words = self._tokenize(text)
        if not words:
            return [], []

        doc_len = len(words)
        avgdl = stats["avgdl"] or 1.0
        k1 = self._bm25_k1
        b = self._bm25_b

        counts = Counter(words)
        indices = []
        values = []
        for term, tf in counts.items():
            idf = self._bm25_idf(term, stats)
            if idf <= 0:
                continue  # Skip terms that appear in all documents
            # BM25 TF component with length normalization
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            weight = idf * tf_norm
            indices.append(self._term_id(term))
            values.append(weight)
        return indices, values

    def _encode_bm25_query(self, text: str, stats: dict) -> tuple[list[int], list[float]]:
        """Encode a query into a BM25 sparse vector.

        Query terms use IDF weights only (no TF normalization needed for
        short queries). Qdrant dot product with document BM25 vectors
        produces the approximate BM25 ranking.
        """
        words = self._tokenize(text)
        if not words:
            return [], []

        seen = set()
        indices = []
        values = []
        for term in words:
            if term in seen:
                continue
            seen.add(term)
            idf = self._bm25_idf(term, stats)
            if idf <= 0:
                continue
            indices.append(self._term_id(term))
            values.append(idf)
        return indices, values

    def _tokenize_sparse(self, text: str, scope: str = "", mode: str = "document") -> tuple[list[int], list[float]]:
        """Encode text into BM25-weighted sparse vector (#319).

        Args:
            text: Document or query text.
            scope: Scope for corpus statistics lookup.
            mode: 'document' for indexing, 'query' for search.

        If no BM25 stats are available (no scope or empty corpus),
        falls back to IDF=1.0 (equivalent to TF-only, but with BM25
        length normalization).
        """
        stats = self._bm25_stats.get(scope, {"N": 0, "df": {}, "avgdl": 1.0})
        if mode == "query":
            return self._encode_bm25_query(text, stats)
        return self._encode_bm25_document(text, stats)

    # ── Circuit breaker state ──────────────────────────────────────────────
    _cb_failures: int = 0
    _cb_threshold: int = 5
    _cb_cooldown: float = 300.0  # 5 minutes
    _cb_open_until: float = 0.0

    def _cb_record_success(self) -> None:
        self._cb_failures = 0

    def _cb_record_failure(self) -> None:
        self._cb_failures += 1
        if self._cb_failures >= self._cb_threshold:
            self._cb_open_until = time.time() + self._cb_cooldown
            logger.error(
                "EmbeddingCircuitBreaker: OPEN — %d consecutive failures, fast-failing for %.0fs",
                self._cb_failures, self._cb_cooldown,
            )

    def _cb_is_open(self) -> bool:
        if self._cb_open_until == 0.0:
            return False
        if time.time() >= self._cb_open_until:
            # Cooldown expired — allow half-open probe
            logger.info("EmbeddingCircuitBreaker: cooldown expired, allowing probe")
            self._cb_open_until = 0.0
            self._cb_failures = 0
            return False
        return True

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via configured embedding API with retry + circuit breaker."""
        import httpx

        if self._cb_is_open():
            raise RuntimeError(
                f"EmbeddingCircuitBreaker: OPEN — fast-failing until "
                f"{self._cb_open_until - time.time():.0f}s remaining"
            )

        max_attempts = 3
        backoff = 1.0  # seconds, doubles each retry
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                resp = httpx.post(
                    self._embedding_base_url,
                    headers={"Authorization": f"Bearer {self._embedding_api_key}"},
                    json={"model": self._embedding_model, "input": texts},
                    timeout=self._embedding_timeout,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                self._cb_record_success()
                return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Embedding API attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt, max_attempts, exc, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(
                        "Embedding API attempt %d/%d failed: %s — exhausted retries",
                        attempt, max_attempts, exc,
                    )

        self._cb_record_failure()
        raise RuntimeError(f"Embedding API failed after {max_attempts} attempts: {last_exc}")

    def ingest_chunks(self, scope: str, chunks: list[DocumentChunk]) -> None:
        """Embed and upsert chunks into Qdrant collection for scope (#23, ADR-011, #319 hybrid).

        Uses composite point ID (scope:source:chunk_index) instead of text-hash
        to avoid collisions for identical text across different documents/scopes.

        When hybrid enabled: stores both dense (named "dense") and sparse (named "sparse")
        vectors per point. When dense-only: stores unnamed dense vector (legacy).
        """
        if not chunks:
            return
        from qdrant_client.models import PointStruct  # type: ignore

        cname = self._qdrant_collection_name(scope)
        self._ensure_collection(scope)

        texts = [c.text for c in chunks]
        dense_vectors = self._embed(texts)

        points = []
        for chunk, dense_vec in zip(chunks, dense_vectors):
            point_id = compute_point_id(
                scope, chunk.metadata.source, chunk.metadata.chunk_index,
            )
            payload = {
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump(),
            }

            if self._hybrid_enabled:
                from qdrant_client.models import SparseVector  # type: ignore
                indices, values = self._tokenize_sparse(chunk.text, scope=scope, mode="document")
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vec,
                            "sparse": SparseVector(indices=indices, values=values),
                        },
                        payload=payload,
                    )
                )
            else:
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=dense_vec,
                        payload=payload,
                    )
                )
        self._client.upsert(collection_name=cname, points=points)

    def ingest_chunks_versioned(
        self,
        scope: str,
        chunks: list[DocumentChunk],
        document_id: str,
        version: int,
    ) -> int:
        """Embed and upsert chunks with document_id, version, status=active in payload.

        Used by the Orchestrator ingest path (POST /{scope}/ingest-document).
        Top-level payload fields `document_id`, `version`, `status` are written
        alongside the existing `text` and `metadata` fields.

        Returns number of chunks indexed.
        """
        if not chunks:
            return 0
        from qdrant_client.models import PointStruct  # type: ignore

        cname = self._qdrant_collection_name(scope)
        self._ensure_collection(scope)

        texts = [c.text for c in chunks]
        dense_vectors = self._embed(texts)

        points = []
        for chunk, dense_vec in zip(chunks, dense_vectors):
            point_id = compute_point_id(scope, chunk.metadata.source, chunk.metadata.chunk_index)
            payload = {
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump(),
                # Top-level fields for orchestrator ingest path (foundation contract)
                "document_id": document_id,
                "version": version,
                "status": "active",
            }
            if self._hybrid_enabled:
                from qdrant_client.models import SparseVector  # type: ignore
                indices, values = self._tokenize_sparse(chunk.text, scope=scope, mode="document")
                points.append(PointStruct(
                    id=point_id,
                    vector={"dense": dense_vec, "sparse": SparseVector(indices=indices, values=values)},
                    payload=payload,
                ))
            else:
                points.append(PointStruct(id=point_id, vector=dense_vec, payload=payload))

        self._client.upsert(collection_name=cname, points=points)
        return len(points)

    def supersede_document_version(self, scope: str, document_id: str, version: int) -> None:
        """Set status=superseded on all Qdrant chunks for (document_id, prior_version).

        Called after a new version is successfully indexed to retire the old version.
        Chunks are retained in Qdrant (rollback possible) but excluded from retrieval.
        """
        from qdrant_client.models import (  # type: ignore
            FieldCondition,
            Filter,
            FilterSelector,
            MatchAny,
            MatchValue,
        )

        cname = self._qdrant_collection_name(scope)
        try:
            self._client.set_payload(
                collection_name=cname,
                payload={"status": "superseded"},
                points=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                            FieldCondition(key="version", match=MatchValue(value=version)),
                        ]
                    )
                ),
            )
            logger.info(
                "QdrantBackend: superseded document_id=%s version=%d in scope=%s",
                document_id, version, scope,
            )
        except Exception as exc:
            logger.warning(
                "QdrantBackend: supersede_document_version failed document_id=%s version=%d scope=%s: %s",
                document_id, version, scope, exc,
            )
            raise

    def delete_by_source(self, scope: str, source: str) -> None:
        """Delete all Qdrant points for a source within a scope (ADR-011)."""
        from qdrant_client.models import (  # type: ignore
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        cname = self._qdrant_collection_name(scope)
        try:
            self._client.delete(
                collection_name=cname,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.source",
                                match=MatchValue(value=source),
                            )
                        ]
                    )
                ),
            )
            logger.info(
                "QdrantBackend: deleted points for source '%s' in scope '%s'",
                source, scope,
            )
        except Exception as exc:
            logger.warning(
                "QdrantBackend: delete_by_source failed for '%s/%s': %s",
                scope, source, exc,
            )
            raise

    def delete_collection(self, scope: str) -> bool:
        """Delete an entire Qdrant collection for a scope (ADR-011)."""
        cname = self._qdrant_collection_name(scope)
        try:
            self._client.delete_collection(collection_name=cname)
            self._collection_cache.discard(cname)
            logger.info("QdrantBackend: deleted collection '%s'", scope)
            return True
        except Exception as exc:
            logger.warning(
                "QdrantBackend: delete_collection failed for '%s': %s", scope, exc,
            )
            raise

    def search_scope(
        self,
        query: str,
        scope: str,
        limit: int,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search within a single Qdrant collection (#319: hybrid or dense-only)."""
        try:
            self._ensure_collection(scope)
        except Exception:
            return []

        t_start = time.time()
        query_vector = self._embed([query])[0]
        t_embed = time.time()

        if self._hybrid_enabled:
            results = self._search_hybrid(query, query_vector, scope, limit, min_score)
            mode = "hybrid"
        else:
            results = self._search_dense_only(query_vector, scope, limit, min_score)
            mode = "dense-only"

        t_search = time.time()
        logger.debug(
            "search_scope: scope=%s mode=%s results=%d embed=%.3fs search=%.3fs total=%.3fs",
            scope, mode, len(results), t_embed - t_start, t_search - t_embed, t_search - t_start,
        )
        return results

    def _search_dense_only(
        self, query_vector: list[float], scope: str, limit: int, min_score: float,
    ) -> list[SearchResult]:
        """Dense-only search (Phase 0). Uses query_points for qdrant-client v1.17+.

        Applies status filter: returns only chunks with status=active OR no status field
        (backward compatibility for chunks indexed before the orchestrator ingest path).
        """
        from qdrant_client.models import (  # type: ignore
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchValue,
            PayloadField,
        )

        cname = self._qdrant_collection_name(scope)
        # status = "active" OR status field absent (backward compat for pre-orchestrator chunks)
        # IsEmptyCondition matches absent/empty fields; IsNullCondition matches only explicit nulls.
        status_filter = Filter(
            should=[
                FieldCondition(key="status", match=MatchValue(value="active")),
                IsEmptyCondition(is_empty=PayloadField(key="status")),
            ]
        )
        kwargs: dict[str, Any] = {
            "collection_name": cname,
            "query": query_vector,
            "query_filter": status_filter,
            "limit": limit,
        }
        if min_score > 0.0:
            kwargs["score_threshold"] = min_score
        query_response = self._client.query_points(**kwargs)
        return build_citations(
            [{"payload": p.payload, "score": p.score if p.score is not None else 0.0}
             for p in query_response.points]
        )

    def _search_hybrid(
        self, query: str, query_vector: list[float], scope: str, limit: int, min_score: float,
    ) -> list[SearchResult]:
        """Hybrid search: dense + sparse prefetch → RRF fusion (#319).

        Applies same status filter as dense-only path.
        """
        from qdrant_client.models import (  # type: ignore
            FieldCondition,
            Filter,
            FusionQuery,
            Fusion,
            IsEmptyCondition,
            MatchValue,
            PayloadField,
            Prefetch,
            SparseVector,
        )

        cname = self._qdrant_collection_name(scope)
        sparse_indices, sparse_values = self._tokenize_sparse(query, scope=scope, mode="query")
        prefetch_limit = self._hybrid_prefetch_limit

        status_filter = Filter(
            should=[
                FieldCondition(key="status", match=MatchValue(value="active")),
                IsEmptyCondition(is_empty=PayloadField(key="status")),
            ]
        )

        prefetches = [
            Prefetch(query=query_vector, using="dense", limit=prefetch_limit),
        ]
        if sparse_indices:
            prefetches.append(
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=prefetch_limit,
                ),
            )

        query_response = self._client.query_points(
            collection_name=cname,
            prefetch=prefetches,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=status_filter,
            limit=limit,
        )

        results_raw = []
        for point in query_response.points:
            score = point.score if point.score is not None else 0.0
            if min_score > 0.0 and score < min_score:
                continue
            results_raw.append({"payload": point.payload, "score": score})

        return build_citations(results_raw)


# Singleton
_knowledge: KnowledgeService | None = None


def get_knowledge_service() -> KnowledgeService:
    """Return knowledge service singleton.

    Auto-selects backends:
    - QDRANT_URL set → QdrantBackend (production, #23)
    - DATABASE_URL set → PostgresDocumentStore (#303)
    - Otherwise → in-memory (tests / local dev)
    """
    import os

    global _knowledge
    if _knowledge is None:
        _knowledge = KnowledgeService()
        qdrant_url = os.getenv("QDRANT_URL", "")
        if qdrant_url:
            try:
                _knowledge._qdrant = QdrantBackend(
                    url=qdrant_url,
                    api_key=os.getenv("QDRANT_API_KEY", ""),
                )
                logger.info("KnowledgeService: using Qdrant backend at %s", qdrant_url)
            except Exception as exc:
                logger.warning(
                    "KnowledgeService: Qdrant init failed (%s) — falling back to in-memory",
                    exc,
                )
        # Document lifecycle store (#303)
        # If Postgres init fails (auth, connectivity), log LOUDLY and leave
        # _doc_store=None so /internal/health reports degraded — do not let a
        # silent exception half-initialize the singleton.
        from services.document_store import get_document_store
        try:
            _knowledge._doc_store = get_document_store()
            store_type = type(_knowledge._doc_store).__name__
            logger.info("KnowledgeService: document store = %s", store_type)
        except Exception as exc:
            logger.error(
                "KnowledgeService: DocumentStore init FAILED (%s: %s) — "
                "service will run degraded: document listing and dedup will not work. "
                "Check DATABASE_URL, agentopia-postgres-auth secret, and Postgres availability.",
                type(exc).__name__, exc,
            )
            _knowledge._doc_store = None
    return _knowledge
