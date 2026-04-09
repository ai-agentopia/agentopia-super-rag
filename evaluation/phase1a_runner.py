"""Phase 1a RAGAS evaluation runner (#317).

Evaluates the ACTUAL retrieval + response path — not pre-baked fixtures.
Each sample query is executed against KnowledgeService.search(), a response
is generated from the retrieved contexts, and RAGAS scores the real output.

This means: if retrieval code changes, contexts change, scores change.
That is what makes this a directional regression harness.

Usage:
    # Live evaluation against running knowledge service
    cd bot-config-api/src
    pip install -r evaluation/requirements-eval.txt
    OPENROUTER_API_KEY=... QDRANT_URL=http://localhost:6333 \
        python -m evaluation.phase1a_runner

    # Custom options
    python -m evaluation.phase1a_runner \
        --dataset evaluation/datasets/phase1a_sample.json \
        --judge-model google/gemini-2.0-flash-001 \
        --response-model google/gemini-2.0-flash-001 \
        --max-samples 3 \
        --output-dir evaluation/results

IMPORTANT: This is Phase 1a — EARLY SIGNAL ONLY.
    - No nDCG, MRR, or Precision@K (those require Phase 1b #318)
    - No authoritative retrieval-ranking gate
    - Metrics are reference-free LLM-as-judge (inherent biases)
    - Use for directional regression checks between code changes
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Protocol

logger = logging.getLogger(__name__)

# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_DATASET = "evaluation/datasets/phase1a_sample.json"
DEFAULT_JUDGE_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_RESPONSE_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_MAX_SAMPLES = 5
DEFAULT_OUTPUT_DIR = "evaluation/results"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

WARN_THRESHOLDS = {
    "faithfulness": 0.7,
    "context_utilization": 0.6,
}


# ── Retrieval adapter ────────────────────────────────────────────────────────


class RetrievalAdapter(Protocol):
    """Interface for retrieval backends. Implementations call the actual system."""

    def retrieve(self, query: str, scope: str, limit: int) -> list[str]:
        """Return list of retrieved text chunks for a query."""
        ...


class KnowledgeServiceAdapter:
    """Calls KnowledgeService.search() — the actual retrieval path under test.

    Requires QDRANT_URL set and knowledge data indexed.
    Code changes in retrieval logic will change the returned contexts.
    """

    def __init__(self):
        from services.knowledge import get_knowledge_service
        self._svc = get_knowledge_service()
        if not self._svc._qdrant:
            raise RuntimeError(
                "KnowledgeServiceAdapter requires QDRANT_URL and indexed data. "
                "Set QDRANT_URL=http://... and ensure scopes have documents."
            )
        logger.info("KnowledgeServiceAdapter: using live KnowledgeService with Qdrant")

    def retrieve(self, query: str, scope: str, limit: int) -> list[str]:
        results = self._svc.search(query, scopes=[scope], limit=limit)
        return [r.text for r in results]


class HttpAdapter:
    """Calls bot-config-api HTTP search endpoint — for remote evaluation.

    SCOPE SEMANTICS WARNING:
    This adapter uses bot-auth mode (Bearer token + X-Bot-Name). The server
    resolves scopes from the bot's subscription, NOT from the per-sample scope
    field. This means:
    - All samples are evaluated against the bot's subscribed scopes
    - The per-sample `scope` field is IGNORED by the server
    - This is NOT semantically equivalent to KnowledgeServiceAdapter

    Use this adapter only when:
    - All dataset samples target scopes the bot is subscribed to
    - You accept that scope resolution is bot-subscription-scoped

    For exact per-sample scope control, use KnowledgeServiceAdapter instead.
    """

    def __init__(self, base_url: str, token: str, bot_name: str = "eval-bot"):
        import httpx
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._bot_name = bot_name
        self._httpx = httpx
        self._warned_scopes: set[str] = set()
        logger.info("HttpAdapter: targeting %s (bot=%s)", self._base_url, bot_name)
        logger.warning(
            "HttpAdapter: scopes are resolved server-side from bot '%s' subscription. "
            "Per-sample scope field is NOT used for routing. "
            "Use KnowledgeServiceAdapter for exact per-sample scope control.",
            bot_name,
        )

    def retrieve(self, query: str, scope: str, limit: int) -> list[str]:
        if scope and scope not in self._warned_scopes:
            logger.warning(
                "HttpAdapter: sample scope '%s' is not used for routing — "
                "server resolves scopes from bot '%s' subscription",
                scope, self._bot_name,
            )
            self._warned_scopes.add(scope)

        resp = self._httpx.get(
            f"{self._base_url}/api/v1/knowledge/search",
            params={"query": query, "limit": limit},
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-Bot-Name": self._bot_name,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return [r["text"] for r in data.get("results", [])]


def validate_dataset_for_adapter(samples: list[dict], adapter: RetrievalAdapter) -> None:
    """Validate dataset is compatible with the chosen adapter.

    Raises ValueError if dataset has multiple scopes and adapter is HttpAdapter
    (which cannot route per-sample scope).
    """
    if not isinstance(adapter, HttpAdapter):
        return  # KnowledgeServiceAdapter supports per-sample scopes

    scopes = {s.get("scope", "") for s in samples if s.get("scope")}
    if len(scopes) > 1:
        raise ValueError(
            f"Dataset has {len(scopes)} distinct scopes {scopes}, but HttpAdapter "
            f"resolves scopes from bot subscription (ignores per-sample scope). "
            f"Use KnowledgeServiceAdapter for multi-scope datasets, or ensure all "
            f"samples target scopes the bot is subscribed to."
        )


# ── Response generation ──────────────────────────────────────────────────────


async def generate_response(
    query: str,
    contexts: list[str],
    model: str,
    base_url: str,
    api_key: str,
) -> str:
    """Generate a response from query + retrieved contexts via LLM.

    This exercises a realistic answer-generation path: the LLM must
    synthesize an answer grounded in the provided contexts.
    """
    from openai import AsyncOpenAI

    if not contexts:
        return "I don't have relevant knowledge to answer this question."

    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# ── Dataset loading ──────────────────────────────────────────────────────────


def load_dataset(path: str, max_samples: int) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", [])
    if max_samples > 0:
        samples = samples[:max_samples]
    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


# ── Core evaluation ──────────────────────────────────────────────────────────


async def run_evaluation(
    samples: list[dict],
    retrieval: RetrievalAdapter,
    judge_model: str,
    response_model: str,
    base_url: str,
    api_key: str,
) -> dict:
    """Run Phase 1a evaluation: retrieve → generate → score.

    1. For each sample: call retrieval adapter (system under test)
    2. Generate response from real contexts via LLM
    3. Score with RAGAS reference-free metrics
    """
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            ContextUtilization,
            Faithfulness,
        )
    except ImportError as e:
        logger.error(
            "RAGAS deps not installed. Run: pip install -r evaluation/requirements-eval.txt"
        )
        raise SystemExit(1) from e

    judge_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    llm = llm_factory(judge_model, client=judge_client)

    # NOTE: AnswerRelevancy requires an embeddings model (extra dependency).
    # Using Faithfulness + ContextUtilization only — both reference-free, LLM-only.
    scorers = {
        "faithfulness": Faithfulness(llm=llm),
        "context_utilization": ContextUtilization(llm=llm),
    }

    results = []
    for i, sample in enumerate(samples):
        sample_id = sample.get("id", f"s{i:02d}")
        scope = sample.get("scope", "")
        limit = sample.get("search_limit", 5)

        # Step 1: ACTUAL RETRIEVAL — this is the system under test
        logger.info("[%d/%d] %s — retrieving from scope '%s'...", i + 1, len(samples), sample_id, scope)
        try:
            contexts = retrieval.retrieve(sample["user_input"], scope, limit)
        except Exception as exc:
            logger.warning("Retrieval failed for %s: %s", sample_id, exc)
            contexts = []

        # Step 2: GENERATE RESPONSE from real contexts
        logger.info("[%d/%d] %s — generating response (%d contexts)...", i + 1, len(samples), sample_id, len(contexts))
        try:
            response = await generate_response(
                sample["user_input"], contexts, response_model, base_url, api_key,
            )
        except Exception as exc:
            logger.warning("Response generation failed for %s: %s", sample_id, exc)
            response = f"Error generating response: {exc}"

        # Step 3: SCORE with RAGAS
        logger.info("[%d/%d] %s — scoring...", i + 1, len(samples), sample_id)
        scores = {}
        for metric_name, scorer in scorers.items():
            try:
                result = await scorer.ascore(
                    user_input=sample["user_input"],
                    response=response,
                    retrieved_contexts=contexts if contexts else ["No context retrieved."],
                )
                scores[metric_name] = round(result.value, 4)
            except Exception as exc:
                logger.warning("Metric %s failed for %s: %s", metric_name, sample_id, exc)
                scores[metric_name] = None

        results.append({
            "id": sample_id,
            "query": sample["user_input"][:80],
            "scenario": sample.get("scenario", ""),
            "scope": scope,
            "contexts_retrieved": len(contexts),
            "response_preview": response[:200],
            "scores": scores,
        })

    return _build_summary(results)


def _build_summary(results: list[dict]) -> dict:
    metric_names = ["faithfulness", "context_utilization"]
    aggregates = {}
    for metric in metric_names:
        values = [r["scores"].get(metric) for r in results if r["scores"].get(metric) is not None]
        if values:
            avg = round(sum(values) / len(values), 4)
            aggregates[metric] = {
                "mean": avg,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
                "warn": avg < WARN_THRESHOLDS.get(metric, 0.0),
            }
        else:
            aggregates[metric] = {"mean": None, "count": 0, "warn": True}

    any_warn = any(a.get("warn", False) for a in aggregates.values())
    total_contexts = sum(r.get("contexts_retrieved", 0) for r in results)

    return {
        "per_sample": results,
        "aggregates": aggregates,
        "total_contexts_retrieved": total_contexts,
        "status": "WARN" if any_warn else "OK",
    }


def write_artifacts(
    summary: dict,
    output_dir: str,
    judge_model: str,
    response_model: str,
    dataset_path: str,
    sample_count: int,
    adapter_type: str,
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    artifact = {
        "phase": "1a",
        "timestamp": ts,
        "judge_model": judge_model,
        "response_model": response_model,
        "adapter": adapter_type,
        "dataset": dataset_path,
        "sample_count": sample_count,
        "total_contexts_retrieved": summary.get("total_contexts_retrieved", 0),
        "note": "Phase 1a EARLY SIGNAL ONLY. No nDCG/MRR. Not an authoritative retrieval gate. Contexts and responses produced by system under test.",
        **summary,
    }
    json_path = os.path.join(output_dir, f"phase1a_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2)

    md_lines = [
        f"# Phase 1a Evaluation — {ts}",
        "",
        "> **EARLY SIGNAL ONLY** — No nDCG, MRR, or Precision@K.",
        "> Contexts and responses produced by system under test (not fixtures).",
        "> See #318 for Phase 1b authoritative gate.",
        "",
        f"- **Judge model**: `{judge_model}`",
        f"- **Response model**: `{response_model}`",
        f"- **Adapter**: `{adapter_type}`",
        f"- **Dataset**: `{dataset_path}`",
        f"- **Samples evaluated**: {sample_count}",
        f"- **Total contexts retrieved**: {summary.get('total_contexts_retrieved', 0)}",
        f"- **Status**: **{summary['status']}**",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Min | Max | Count | Warning |",
        "|---|---|---|---|---|---|",
    ]
    for name, agg in summary["aggregates"].items():
        mean = f"{agg['mean']:.4f}" if agg["mean"] is not None else "N/A"
        mn = f"{agg.get('min', 'N/A')}" if agg.get("min") is not None else "N/A"
        mx = f"{agg.get('max', 'N/A')}" if agg.get("max") is not None else "N/A"
        warn = "YES" if agg.get("warn") else "no"
        md_lines.append(f"| {name} | {mean} | {mn} | {mx} | {agg['count']} | {warn} |")

    md_lines.extend(["", "## Per-Sample Results", "",
        "| ID | Scenario | Contexts | Faithfulness | Utilization |",
        "|---|---|---|---|---|"])
    for r in summary["per_sample"]:
        s = r["scores"]
        f_v = f"{s.get('faithfulness')}" if s.get("faithfulness") is not None else "N/A"
        c_v = f"{s.get('context_utilization')}" if s.get("context_utilization") is not None else "N/A"
        md_lines.append(f"| {r['id']} | {r['scenario']} | {r['contexts_retrieved']} | {f_v} | {c_v} |")

    md_lines.extend(["", "---", "",
        "*Generated by Phase 1a evaluation runner (#317). Directional signal only.*"])

    md_path = os.path.join(output_dir, f"phase1a_{ts}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1a RAGAS evaluation — calls actual retrieval + generation path (#317)"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--response-model", default=DEFAULT_RESPONSE_MODEL)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--adapter", choices=["knowledge-service", "http"], default="knowledge-service",
        help=(
            "Retrieval adapter. knowledge-service: in-process via KnowledgeService.search(), "
            "evaluates exact per-sample scope (requires QDRANT_URL). "
            "http: remote via bot-config-api endpoint, scope resolved from bot subscription "
            "(per-sample scope field is NOT used for routing). "
            "Multi-scope datasets require knowledge-service adapter."
        ),
    )
    parser.add_argument("--http-url", default="", help="Base URL for http adapter")
    parser.add_argument("--http-token", default="", help="Bearer token for http adapter")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Build retrieval adapter — this is the system under test
    if args.adapter == "knowledge-service":
        adapter = KnowledgeServiceAdapter()
        adapter_type = "knowledge-service (in-process)"
    else:
        adapter = HttpAdapter(args.http_url, args.http_token)
        adapter_type = f"http ({args.http_url})"

    samples = load_dataset(args.dataset, args.max_samples)
    if not samples:
        logger.error("No samples")
        sys.exit(1)

    validate_dataset_for_adapter(samples, adapter)

    logger.info("Phase 1a: %d samples, judge=%s, response=%s, adapter=%s",
                len(samples), args.judge_model, args.response_model, adapter_type)

    t0 = time.monotonic()
    summary = asyncio.run(run_evaluation(
        samples, adapter, args.judge_model, args.response_model, args.base_url, api_key,
    ))
    elapsed = time.monotonic() - t0

    json_path, md_path = write_artifacts(
        summary, args.output_dir, args.judge_model, args.response_model,
        args.dataset, len(samples), adapter_type,
    )

    logger.info("Done in %.1fs — %d contexts retrieved total", elapsed, summary.get("total_contexts_retrieved", 0))
    logger.info("Status: %s", summary["status"])
    for name, agg in summary["aggregates"].items():
        mean = f"{agg['mean']:.4f}" if agg["mean"] is not None else "N/A"
        warn = " ⚠ WARN" if agg.get("warn") else ""
        logger.info("  %s: %s%s", name, mean, warn)
    logger.info("Artifacts: %s, %s", json_path, md_path)


if __name__ == "__main__":
    main()
