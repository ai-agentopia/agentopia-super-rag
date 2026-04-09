"""Knowledge base / RAG models (M3 #23, #24, #25, #100, #102).

Document ingestion, chunking, retrieval, citations, and scoping.
"""

from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Knowledge source type (Super RAG Phase 0, Track C foundation)."""

    BUSINESS_DOC = "business_doc"  # Uploaded business documents (default)
    CODE_FILE = "code_file"  # Repository source code (Phase 3a)
    FEATURE_ARTIFACT = "feature_artifact"  # Issues/PRs/specs (conditional)


class DocumentFormat(str, Enum):
    """Supported document formats for ingestion."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    CODE = "code"


class ChunkingStrategy(str, Enum):
    """How to split documents into chunks."""

    FIXED_SIZE = "fixed_size"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    CODE_AWARE = "code_aware"


class IngestConfig(BaseModel):
    """Configuration for document ingestion (#23)."""

    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=512)
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document."""

    source: str = ""  # Filename, URL, or repo path
    format: DocumentFormat = DocumentFormat.TEXT
    source_type: SourceType = SourceType.BUSINESS_DOC  # Phase 0 foundation for Track C
    scope: str = ""  # Knowledge scope (Qdrant collection)
    page: int | None = None  # PDF page number
    section: str = ""  # Section heading
    language: str = ""  # Code language (for code files)
    chunk_index: int = 0  # Position within document
    total_chunks: int = 0
    ingested_at: float = 0.0  # Unix timestamp — when chunk entered KB (ADR-011)
    document_hash: str = ""  # SHA-256 of original document content (ADR-011)


class DocumentChunk(BaseModel):
    """A single chunk of a document ready for embedding."""

    text: str
    metadata: DocumentMetadata
    embedding: list[float] = []  # Populated after embedding


class IngestRequest(BaseModel):
    """Request to ingest a document into a knowledge scope (#23)."""

    scope: str  # Target knowledge scope
    content: str  # Document text content
    source: str = ""  # Source filename/URL
    format: DocumentFormat = DocumentFormat.TEXT
    config: IngestConfig = IngestConfig()


class IngestResult(BaseModel):
    """Result of document ingestion."""

    scope: str
    source: str
    chunks_created: int
    chunks_skipped: int = 0
    format: DocumentFormat
    document_hash: str = ""  # SHA-256 of ingested content (ADR-011)
    ingested_at: float = 0.0  # Unix timestamp (ADR-011)


class SearchRequest(BaseModel):
    """Knowledge base search request."""

    query: str
    scopes: list[str]  # Which knowledge scopes to search
    limit: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class Citation(BaseModel):
    """Source attribution for a retrieved chunk (#102)."""

    source: str
    section: str = ""
    page: int | None = None
    chunk_index: int = 0
    score: float = 0.0
    ingested_at: float = 0.0  # Provenance: Unix timestamp when chunk was ingested (ADR-011)
    document_hash: str = ""  # Provenance: SHA-256 of source document (ADR-011)


class SearchResult(BaseModel):
    """A single search result with citation."""

    text: str
    score: float
    citation: Citation
    scope: str


class KnowledgeScope(BaseModel):
    """Knowledge scope metadata (#100)."""

    name: str
    document_count: int = 0
    chunk_count: int = 0
    last_indexed: float = 0.0  # Unix timestamp

    def is_stale(self, max_age_secs: float = 86400.0) -> bool:
        """Return True if the scope has not been indexed within max_age_secs (#101).

        A scope with last_indexed=0 (never indexed) is always considered stale.
        Default max age: 24 hours.
        """
        if self.last_indexed == 0.0:
            return True
        import time

        return (time.time() - self.last_indexed) > max_age_secs


class DocumentRecordStatus(str, Enum):
    """Document lifecycle states (ADR-012)."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class DocumentRecord(BaseModel):
    """Durable document lifecycle record (ADR-011/012, #303).

    Persisted in Postgres. One active record per (scope, source).
    Tombstones retained for superseded/deleted.
    """

    id: int = 0
    scope: str
    source: str
    document_hash: str  # SHA-256
    format: DocumentFormat = DocumentFormat.TEXT
    source_type: SourceType = SourceType.BUSINESS_DOC  # Phase 0 foundation for Track C
    chunk_count: int = 0
    ingested_at: float = 0.0  # Unix timestamp
    status: DocumentRecordStatus = DocumentRecordStatus.ACTIVE
    superseded_at: float | None = None
    deleted_at: float | None = None


class RepoIndexConfig(BaseModel):
    """Configuration for code repository indexing (#24)."""

    repo_url: str
    branch: str = "main"
    include_patterns: list[str] = ["**/*.py", "**/*.ts", "**/*.js", "**/*.go"]
    exclude_patterns: list[str] = ["**/node_modules/**", "**/.git/**", "**/vendor/**"]
    scope: str = ""  # Target knowledge scope
