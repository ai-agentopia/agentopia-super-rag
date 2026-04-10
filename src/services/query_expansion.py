"""Query expansion via LLM alternative phrasings (W3a #16).

Generates N alternative phrasings of a search query via llm-proxy,
then merges retrieval results from all phrasings via RRF.

This module is only called when query expansion is explicitly enabled.
Default search behavior is unchanged.
"""

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_EXPANSION_COUNT = 3  # N alternative phrasings
MAX_EXPANSION_COUNT = 5
LLM_PROXY_URL = os.getenv("LLM_PROXY_URL", "")
LLM_PROXY_MODEL = os.getenv(
    "QUERY_EXPANSION_MODEL", "openai/gpt-4o-mini"
)
LLM_PROXY_TIMEOUT = int(os.getenv("QUERY_EXPANSION_TIMEOUT_MS", "3000"))

EXPANSION_PROMPT = """Generate {n} alternative phrasings of the following search query.
Each phrasing should express the same information need using different words or structure.
Return ONLY the alternative phrasings, one per line, no numbering, no explanation.

Query: {query}"""


# ── Query expansion ─────────────────────────────────────────────────────


def expand_query(
    query: str,
    n: int = DEFAULT_EXPANSION_COUNT,
    api_key: str = "",
) -> list[str]:
    """Generate N alternative phrasings of a query via LLM.

    Returns list of alternative phrasings. On failure, returns empty list
    (caller falls back to original query only — no error surfaced).

    Args:
        query: Original search query.
        n: Number of alternative phrasings to generate.
        api_key: API key for llm-proxy / OpenRouter.

    Returns:
        List of alternative phrasings (may be empty on failure).
    """
    n = min(n, MAX_EXPANSION_COUNT)
    if not api_key:
        api_key = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("Query expansion: no API key configured, skipping")
        return []

    base_url = LLM_PROXY_URL or os.getenv(
        "EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"
    )
    # Strip /embeddings suffix if present — we need /chat/completions
    if base_url.endswith("/embeddings"):
        base_url = base_url[: -len("/embeddings")]
    chat_url = f"{base_url}/chat/completions"

    import json
    import urllib.request
    import urllib.error

    prompt = EXPANSION_PROMPT.format(n=n, query=query)
    payload = json.dumps({
        "model": LLM_PROXY_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 200,
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

        content = data["choices"][0]["message"]["content"].strip()
        phrasings = [
            line.strip().lstrip("0123456789.-) ")
            for line in content.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        # Filter empty and duplicates of the original
        phrasings = [
            p for p in phrasings
            if p and p.lower() != query.lower()
        ][:n]

        logger.info(
            "Query expansion: generated %d phrasings in %.0fms (model=%s)",
            len(phrasings), elapsed_ms, LLM_PROXY_MODEL,
        )
        return phrasings

    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.warning(
            "Query expansion failed (%.0fms): %s — falling back to dense-only",
            elapsed_ms, exc,
        )
        return []


# ── RRF merge ────────────────────────────────────────────────────────────

RRF_K = 60  # Standard RRF constant


def rrf_merge(
    ranked_lists: list[list[dict[str, Any]]],
    limit: int,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Merge multiple ranked result lists via Reciprocal Rank Fusion.

    Each result is identified by (scope, source, chunk_index).
    RRF score = sum(1 / (k + rank)) across all lists where the result appears.

    Args:
        ranked_lists: List of ranked result lists. Each result must have
            citation.source, citation.chunk_index, and scope.
        limit: Maximum number of merged results to return.
        rrf_k: RRF constant (default 60).

    Returns:
        Merged results sorted by RRF score descending, limited to `limit`.
    """
    scores: dict[str, float] = {}
    results_by_key: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            cit = result.get("citation", {})
            key = f"{result.get('scope', '')}:{cit.get('source', '')}:{cit.get('chunk_index', 0)}"
            rrf_score = 1.0 / (rrf_k + rank + 1)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in results_by_key:
                results_by_key[key] = result

    # Sort by RRF score descending
    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)

    merged = []
    for key in sorted_keys[:limit]:
        result = results_by_key[key].copy()
        result["score"] = round(scores[key], 6)
        merged.append(result)

    return merged
