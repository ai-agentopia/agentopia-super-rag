"""W4: LLM listwise reranking (per-scope opt-in, #18).

Reranks a candidate set by sending query + all candidates to the LLM in a
single call. The LLM returns a relevance-ordered list of candidate indices.

This module is only called when reranking is explicitly enabled for a scope.
Default search behavior is unchanged.
"""

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

LLM_PROXY_URL = os.getenv("LLM_PROXY_URL", "")
LLM_PROXY_MODEL = os.getenv("RERANK_MODEL", "openai/gpt-4o-mini")
LLM_PROXY_TIMEOUT = int(os.getenv("RERANK_TIMEOUT_MS", "5000"))
DEFAULT_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "20"))
CANDIDATE_TEXT_LIMIT = 300  # chars per candidate to keep token cost bounded

RERANK_PROMPT = """You are a relevance ranking assistant.
Given a search query and a numbered list of document excerpts, \
return the indices of the most relevant documents in order of relevance, \
most relevant first.

Return ONLY a comma-separated list of indices (e.g. "2,0,4,1,3"). \
Do not explain. Do not repeat indices. Include only the indices you consider relevant.

Query: {query}

Documents:
{documents}"""


# ── Reranking ────────────────────────────────────────────────────────────


def rerank_results(
    query: str,
    candidates: list[dict[str, Any]],
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Rerank candidates by LLM listwise relevance scoring.

    Sends the query and all candidate excerpts to the LLM in one call.
    The LLM returns a comma-separated list of candidate indices in relevance
    order. On failure, returns the original vector-ranked order (no 5xx).

    Args:
        query: The original search query.
        candidates: Vector-ranked result dicts (text, score, scope, citation).
        api_key: API key for llm-proxy / OpenRouter.

    Returns:
        Reranked candidate list. Candidates not mentioned by LLM are appended
        in their original vector order. Returns original order on failure.
    """
    if not candidates:
        return candidates

    if not api_key:
        api_key = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("W4: no API key configured, skipping rerank — returning vector order")
        return candidates

    base_url = LLM_PROXY_URL or os.getenv(
        "EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"
    )
    if base_url.endswith("/embeddings"):
        base_url = base_url[: -len("/embeddings")]
    chat_url = f"{base_url}/chat/completions"

    import json
    import urllib.request

    doc_lines = []
    for i, c in enumerate(candidates):
        text = c.get("text", "")[:CANDIDATE_TEXT_LIMIT].replace("\n", " ")
        doc_lines.append(f"[{i}] {text}")

    prompt = RERANK_PROMPT.format(query=query, documents="\n".join(doc_lines))
    payload = json.dumps({
        "model": f"openrouter/{LLM_PROXY_MODEL}",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100,
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

        # Parse comma-separated indices
        rerank_order: list[int] = []
        seen: set[int] = set()
        for part in content.replace(";", ",").split(","):
            part = part.strip().strip("[]().")
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(candidates) and idx not in seen:
                    rerank_order.append(idx)
                    seen.add(idx)

        # Append candidates not mentioned by LLM in original vector order
        remaining = [i for i in range(len(candidates)) if i not in seen]
        final_order = rerank_order + remaining

        reranked = [candidates[i] for i in final_order]

        logger.info(
            "W4: reranked %d candidates in %.0fms (model=%s, llm_order=%s)",
            len(candidates), elapsed_ms, LLM_PROXY_MODEL, rerank_order,
        )
        return reranked

    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.warning(
            "W4: reranking failed (%.0fms): %s — falling back to vector order",
            elapsed_ms, exc,
        )
        return candidates
