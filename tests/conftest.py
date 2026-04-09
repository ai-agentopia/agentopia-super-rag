"""knowledge-api test fixtures (#330).

knowledge-api owns all retrieval runtime code directly:
  - models/knowledge.py (pure Pydantic models)
  - services/knowledge.py (QdrantBackend, KnowledgeService)
  - services/document_store.py (DocumentStore, PostgresDocumentStore)

No PYTHONPATH coupling to bot-config-api. No Dockerfile COPY stubs needed.
"""

import os

# ── Default env for tests ─────────────────────────────────────────────────────
os.environ.setdefault("KNOWLEDGE_API_INTERNAL_TOKEN", "test-internal-token-for-tests")
os.environ.setdefault("K8S_NAMESPACE", "agentopia")
