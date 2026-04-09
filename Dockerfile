# agentopia-super-rag — knowledge-api service
#
# BUILD CONTEXT: this repo root (NOT the agentopia-protocol monorepo root).
#   docker build -t ghcr.io/ai-agentopia/knowledge-api:dev-local .
#
# NOTE: Image push to GHCR is gated by atomic cutover (agentopia-super-rag#24).
# Do NOT enable push in CI until Phase 3 cutover sequence is initiated.

FROM python:3.12-slim

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# DB migrations (document_records schema)
COPY db/ db/

# knowledge-api application code
COPY src/ .

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["python", "main.py"]
