"""HyDE: Hypothetical Document Embedding (W3b #17).

Generates one short hypothetical answer to the search query via llm-proxy,
then retrieves against the hypothetical answer embedding. Merges HyDE results
with the original query results via RRF.

This module is only called when HyDE is explicitly enabled for a scope.
Default search behavior is unchanged.
"""

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

LLM_PROXY_URL = os.getenv("LLM_PROXY_URL", "")
LLM_PROXY_MODEL = os.getenv(
    "HYDE_MODEL", "openai/gpt-4o-mini"
)
LLM_PROXY_TIMEOUT = int(os.getenv("HYDE_TIMEOUT_MS", "3000"))

HYDE_PROMPT = """Write a short hypothetical document that would directly answer the following search query.
The document should be 2-4 sentences, factual in tone, and contain the kind of content that would appear in a knowledge base.
Return ONLY the hypothetical document text, no preamble, no explanation.

Query: {query}"""


# ── Hypothesis generation ────────────────────────────────────────────────


def generate_hypothesis(
    query: str,
    api_key: str = "",
) -> str:
    """Generate one hypothetical answer document for the query via LLM.

    Returns the hypothesis text. On failure, returns empty string
    (caller falls back to dense-only — no error surfaced).

    Args:
        query: Search query to generate a hypothesis for.
        api_key: API key for llm-proxy / OpenRouter.

    Returns:
        Hypothesis text, or empty string on failure.
    """
    if not api_key:
        api_key = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("HyDE: no API key configured, skipping")
        return ""

    base_url = LLM_PROXY_URL or os.getenv(
        "EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"
    )
    if base_url.endswith("/embeddings"):
        base_url = base_url[: -len("/embeddings")]
    chat_url = f"{base_url}/chat/completions"

    import json
    import urllib.request

    prompt = HYDE_PROMPT.format(query=query)
    payload = json.dumps({
        "model": f"openrouter/{LLM_PROXY_MODEL}",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 150,
    }).encode()

    req = urllib.request.Request(
        chat_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=LLM_PROXY_TIMEOUT / 1000)
        data = json.loads(resp.read().decode())
        elapsed_ms = (time.monotonic() - t0) * 1000

        hypothesis = data["choices"][0]["message"]["content"].strip()
        logger.info(
            "HyDE: generated hypothesis in %.0fms (model=%s, len=%d chars)",
            elapsed_ms, LLM_PROXY_MODEL, len(hypothesis),
        )
        return hypothesis

    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.warning(
            "HyDE: generation failed (%.0fms): %s — falling back to dense-only",
            elapsed_ms, exc,
        )
        return ""
