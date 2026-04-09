# agentopia-super-rag — knowledge-api service
#
# Phase 1 — PLACEHOLDER
# Service source has not been extracted yet (Phase 2).
# This Dockerfile is structurally correct for the future extracted service
# but will not produce a runnable container until Phase 2 populates src/.
#
# Build context: this repo root (NOT the agentopia-protocol monorepo root).
#   docker build -t ghcr.io/ai-agentopia/knowledge-api:dev-local .
#
# NOTE: Image push to GHCR is gated by atomic cutover (agentopia-super-rag#24).
# Do NOT enable push in CI until Phase 3 cutover sequence is initiated.

FROM python:3.12-slim

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# DB migrations
# Phase 1: placeholder — migration files will be included in Phase 2.
# In agentopia-protocol, these live in bot-config-api/db/.
# Extraction destination to be determined before Phase 2 completes.
# COPY db/ db/

# Service source
# Phase 1: placeholder — source will be copied in Phase 2.
# COPY src/ .

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

# Phase 1: placeholder entrypoint — replace with real CMD in Phase 2
CMD ["python", "-c", "print('Phase 1 placeholder — source not yet extracted'); import sys; sys.exit(1)"]
