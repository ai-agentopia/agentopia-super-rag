"""Per-scope evaluation service.

Provides:
- Golden question set storage and retrieval
- Per-scope baseline establishment (run golden questions, store nDCG/MRR/P/R)
- Benchmark execution (run golden questions against live retrieval, return metrics)
- Regression detection on document replacement (compare vs per-scope baseline)
- Regression gate logic: pass / warning / blocked / overridden / no_baseline
- Operator notification on regression block (structured log + evaluation_results record)
- Evaluation result history (append-only PostgreSQL)

Gate thresholds:
    delta >= 0               → verdict = "passed"
    -0.02 <= delta < 0       → verdict = "warning"  (allowed, operator informed)
    delta < -0.02            → verdict = "blocked"   (operator must explicitly override)

Integration:
    Called from KnowledgeService.ingest_from_orchestrator() after a successful
    document replacement. Runs asynchronously post-commit so retrieval is never
    blocked on evaluation.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.knowledge import KnowledgeService

logger = logging.getLogger(__name__)

# Gate thresholds — matches the agreed system design
_WARN_THRESHOLD = -0.02   # delta < this → blocked
_PASS_THRESHOLD = 0.0     # delta >= this → passed; between: warning

K = 5  # metric cutoff


@dataclass
class EvaluationMetrics:
    ndcg_5: float
    mrr: float
    p_5: float
    r_5: float
    question_count: int

    def to_dict(self) -> dict:
        return {
            "ndcg_5": self.ndcg_5,
            "mrr": self.mrr,
            "p_5": self.p_5,
            "r_5": self.r_5,
            "question_count": self.question_count,
        }


@dataclass
class RegressionResult:
    scope: str
    document_id: str | None
    document_version: int | None
    verdict: str          # passed | warning | blocked | no_baseline | no_questions
    ndcg_5: float | None
    delta_ndcg_5: float | None
    baseline_ndcg_5: float | None
    result_id: str | None
    message: str


def _get_db_conn():
    """Return a psycopg connection from DATABASE_URL. Returns None if unset."""
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        return None
    try:
        import psycopg
        return psycopg.connect(database_url, autocommit=False)
    except Exception as exc:
        logger.error("evaluation: failed to connect to DB: %s", exc)
        return None


# ── Golden question set ───────────────────────────────────────────────────────


def add_golden_question(
    scope: str,
    query: str,
    expected_sources: list[dict],
    weight: float = 1.0,
    created_by: str = "",
) -> str:
    """Insert a golden question for a scope. Returns the question id.

    expected_sources format: [{"source": "filename.pdf", "relevance": 2}, ...]
    relevance: 2=fully relevant, 1=partially relevant, 0=not relevant
    """
    conn = _get_db_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL not configured")
    qid = str(uuid.uuid4())
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO golden_questions (id, scope, query, expected_sources, weight, created_by)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (qid, scope, query, json.dumps(expected_sources), weight, created_by or None),
            )
        conn.commit()
    finally:
        conn.close()
    return qid


def list_golden_questions(scope: str) -> list[dict]:
    """Return all golden questions for a scope."""
    conn = _get_db_conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, scope, query, expected_sources, weight, created_by, created_at
                FROM golden_questions WHERE scope = %s ORDER BY created_at
                """,
                (scope,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "id": str(r[0]),
            "scope": r[1],
            "query": r[2],
            "expected_sources": r[3] if isinstance(r[3], list) else json.loads(r[3] or "[]"),
            "weight": r[4],
            "created_by": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]


def delete_golden_question(question_id: str) -> bool:
    """Delete a golden question by id. Returns True if deleted."""
    conn = _get_db_conn()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM golden_questions WHERE id = %s", (question_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    finally:
        conn.close()
    return deleted


# ── Benchmark execution ───────────────────────────────────────────────────────


def run_benchmark(scope: str, svc: "KnowledgeService") -> EvaluationMetrics | None:
    """Run all golden questions for scope against live retrieval. Return aggregate metrics.

    Aggregation uses question weight (default 1.0) as a multiplier in a
    weighted average so high-weight questions contribute proportionally more
    to the final score.

    Returns None if there are no golden questions for this scope.
    """
    from evaluation.retrieval_metrics import compute_query_metrics

    questions = list_golden_questions(scope)
    if not questions:
        logger.info("evaluation: no golden questions for scope=%s — skipping benchmark", scope)
        return None

    weighted_ndcg: list[tuple[float, float]] = []   # (value, weight)
    weighted_mrr:  list[tuple[float, float]] = []
    weighted_p:    list[tuple[float, float]] = []
    weighted_r:    list[tuple[float, float]] = []

    for q in questions:
        weight = float(q.get("weight") or 1.0)
        if weight <= 0:
            weight = 1.0

        try:
            results = svc.search(query=q["query"], scopes=[scope], limit=K)
        except Exception as exc:
            logger.warning("evaluation: search failed for query '%s' scope=%s: %s", q["query"], scope, exc)
            continue

        expected = {e["source"]: e["relevance"] for e in q["expected_sources"]}
        total_relevant = sum(1 for rel in expected.values() if rel >= 1)

        # Build relevance list in retrieval rank order; pad to K with zeros
        relevances = [float(expected.get(r.citation.source, 0)) for r in results]
        while len(relevances) < K:
            relevances.append(0.0)

        m = compute_query_metrics(relevances, total_relevant=total_relevant, k=K)
        weighted_ndcg.append((m["ndcg"],     weight))
        weighted_mrr.append((m["mrr"],       weight))
        weighted_p.append((m["precision"],   weight))
        weighted_r.append((m["recall"],      weight))

    if not weighted_ndcg:
        return None

    def _wavg(pairs: list[tuple[float, float]]) -> float:
        total_w = sum(w for _, w in pairs)
        if total_w == 0:
            return 0.0
        return round(sum(v * w for v, w in pairs) / total_w, 4)

    return EvaluationMetrics(
        ndcg_5=_wavg(weighted_ndcg),
        mrr=_wavg(weighted_mrr),
        p_5=_wavg(weighted_p),
        r_5=_wavg(weighted_r),
        question_count=len(weighted_ndcg),
    )


# ── Baseline management ───────────────────────────────────────────────────────


def establish_baseline(scope: str, svc: "KnowledgeService", notes: str = "") -> dict:
    """Run golden questions and store the result as the per-scope baseline.

    Replaces any existing baseline for this scope.
    Raises RuntimeError if no golden questions exist or DB unavailable.
    """
    metrics = run_benchmark(scope, svc)
    if metrics is None:
        raise RuntimeError(f"No golden questions for scope '{scope}' — cannot establish baseline")

    conn = _get_db_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL not configured")

    bid = str(uuid.uuid4())
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluation_baselines
                    (id, scope, ndcg_5, mrr, p_5, r_5, golden_question_count, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scope) DO UPDATE SET
                    ndcg_5                = EXCLUDED.ndcg_5,
                    mrr                   = EXCLUDED.mrr,
                    p_5                   = EXCLUDED.p_5,
                    r_5                   = EXCLUDED.r_5,
                    golden_question_count = EXCLUDED.golden_question_count,
                    established_at        = NOW(),
                    notes                 = EXCLUDED.notes
                RETURNING id
                """,
                (bid, scope, metrics.ndcg_5, metrics.mrr, metrics.p_5, metrics.r_5,
                 metrics.question_count, notes or None),
            )
            row = conn.cursor().fetchone() if False else cur.fetchone()
            if row:
                bid = str(row[0])
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "evaluation: baseline established scope=%s ndcg_5=%.4f mrr=%.4f questions=%d",
        scope, metrics.ndcg_5, metrics.mrr, metrics.question_count,
    )
    return {"baseline_id": bid, "scope": scope, **metrics.to_dict()}


def get_baseline(scope: str) -> dict | None:
    """Return the current per-scope baseline, or None if not established."""
    conn = _get_db_conn()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, scope, ndcg_5, mrr, p_5, r_5, golden_question_count,
                       established_at, notes
                FROM evaluation_baselines WHERE scope = %s
                """,
                (scope,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "baseline_id": str(row[0]),
        "scope": row[1],
        "ndcg_5": row[2],
        "mrr": row[3],
        "p_5": row[4],
        "r_5": row[5],
        "golden_question_count": row[6],
        "established_at": row[7].isoformat() if row[7] else None,
        "notes": row[8],
    }


def list_baselines() -> list[dict]:
    """Return all per-scope baselines."""
    conn = _get_db_conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, scope, ndcg_5, mrr, p_5, r_5, golden_question_count,
                       established_at, notes
                FROM evaluation_baselines ORDER BY scope
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "baseline_id": str(r[0]),
            "scope": r[1],
            "ndcg_5": r[2],
            "mrr": r[3],
            "p_5": r[4],
            "r_5": r[5],
            "golden_question_count": r[6],
            "established_at": r[7].isoformat() if r[7] else None,
            "notes": r[8],
        }
        for r in rows
    ]


# ── Regression detection ──────────────────────────────────────────────────────


def check_regression(
    scope: str,
    svc: "KnowledgeService",
    trigger: str = "replacement",
    document_id: str | None = None,
    document_version: int | None = None,
) -> RegressionResult:
    """Run benchmark for scope, compare against baseline, apply gate logic.

    Gate thresholds:
        delta >= 0               → verdict = "passed"
        -0.02 <= delta < 0       → verdict = "warning"
        delta < -0.02            → verdict = "blocked"

    Special verdicts (no comparison possible):
        - "no_baseline"   → baseline not established for this scope
        - "no_questions"  → golden question set is empty
        - "eval_error"    → benchmark execution failed

    Always writes a record to evaluation_results (append-only).
    Never raises — failures are logged and recorded as "eval_error".
    """
    result_id = str(uuid.uuid4())

    try:
        baseline = get_baseline(scope)
        if baseline is None:
            logger.info("evaluation: no baseline for scope=%s — skipping gate", scope)
            _write_result(
                result_id=result_id, scope=scope, document_id=document_id,
                document_version=document_version, trigger=trigger,
                ndcg_5=None, delta=None, verdict="no_baseline",
            )
            return RegressionResult(
                scope=scope, document_id=document_id, document_version=document_version,
                verdict="no_baseline", ndcg_5=None, delta_ndcg_5=None,
                baseline_ndcg_5=None, result_id=result_id,
                message=f"No baseline established for scope '{scope}'",
            )

        metrics = run_benchmark(scope, svc)
        if metrics is None:
            _write_result(
                result_id=result_id, scope=scope, document_id=document_id,
                document_version=document_version, trigger=trigger,
                ndcg_5=None, delta=None, verdict="no_questions",
            )
            return RegressionResult(
                scope=scope, document_id=document_id, document_version=document_version,
                verdict="no_questions", ndcg_5=None, delta_ndcg_5=None,
                baseline_ndcg_5=baseline["ndcg_5"], result_id=result_id,
                message=f"No golden questions for scope '{scope}' — gate skipped",
            )

        baseline_ndcg = baseline["ndcg_5"]
        delta = round(metrics.ndcg_5 - baseline_ndcg, 4)

        if delta >= _PASS_THRESHOLD:
            verdict = "passed"
            msg = f"nDCG@5 {metrics.ndcg_5:.4f} (delta={delta:+.4f}) — no regression"
        elif delta >= _WARN_THRESHOLD:
            verdict = "warning"
            msg = f"nDCG@5 {metrics.ndcg_5:.4f} (delta={delta:+.4f}) — minor regression, within warning threshold"
        else:
            verdict = "blocked"
            msg = (
                f"nDCG@5 {metrics.ndcg_5:.4f} (delta={delta:+.4f}) — regression exceeds threshold "
                f"({_WARN_THRESHOLD}). Operator override required."
            )

        _write_result(
            result_id=result_id, scope=scope, document_id=document_id,
            document_version=document_version, trigger=trigger,
            ndcg_5=metrics.ndcg_5, mrr=metrics.mrr,
            p_5=metrics.p_5, r_5=metrics.r_5,
            delta=delta, verdict=verdict,
        )

        log_fn = logger.warning if verdict in ("warning", "blocked") else logger.info
        log_fn(
            "evaluation: scope=%s trigger=%s verdict=%s ndcg_5=%.4f delta=%+.4f",
            scope, trigger, verdict, metrics.ndcg_5, delta,
        )

        if verdict == "blocked":
            logger.warning(
                "evaluation: REGRESSION BLOCKED scope=%s document_id=%s version=%s "
                "ndcg_5=%.4f baseline=%.4f delta=%+.4f result_id=%s",
                scope, document_id, document_version,
                metrics.ndcg_5, baseline_ndcg, delta, result_id,
            )

        return RegressionResult(
            scope=scope, document_id=document_id, document_version=document_version,
            verdict=verdict, ndcg_5=metrics.ndcg_5, delta_ndcg_5=delta,
            baseline_ndcg_5=baseline_ndcg, result_id=result_id, message=msg,
        )

    except Exception as exc:
        logger.error("evaluation: check_regression failed scope=%s: %s", scope, exc)
        try:
            _write_result(
                result_id=result_id, scope=scope, document_id=document_id,
                document_version=document_version, trigger=trigger,
                ndcg_5=None, delta=None, verdict="eval_error",
            )
        except Exception:
            pass
        return RegressionResult(
            scope=scope, document_id=document_id, document_version=document_version,
            verdict="eval_error", ndcg_5=None, delta_ndcg_5=None,
            baseline_ndcg_5=None, result_id=result_id,
            message=f"Evaluation failed: {exc}",
        )


def record_operator_override(
    result_id: str,
    operator_note: str,
    operator_identity: str = "",
) -> bool:
    """Mark an evaluation result as operator-overridden. Returns True on success.

    operator_identity: who performed the override (X-Internal-Token actor,
    username, or any operator identifier the caller supplies). Stored in
    evaluation_results.operator_identity for audit attribution.
    """
    conn = _get_db_conn()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluation_results
                SET operator_override  = TRUE,
                    operator_note      = %s,
                    operator_identity  = %s,
                    verdict            = 'overridden'
                WHERE id = %s AND verdict = 'blocked'
                """,
                (operator_note, operator_identity or None, result_id),
            )
            updated = cur.rowcount > 0
        conn.commit()
    finally:
        conn.close()
    if updated:
        logger.info(
            "evaluation: operator override recorded result_id=%s identity=%s",
            result_id, operator_identity or "(unset)",
        )
    return updated


# ── Evaluation results ────────────────────────────────────────────────────────


def get_evaluation_results(scope: str, limit: int = 50) -> list[dict]:
    """Return recent evaluation results for a scope, newest first."""
    conn = _get_db_conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, scope, document_id, document_version, run_at,
                       trigger, ndcg_5, mrr, p_5, r_5, delta_ndcg_5,
                       verdict, operator_override, operator_note, operator_identity
                FROM evaluation_results
                WHERE scope = %s
                ORDER BY run_at DESC
                LIMIT %s
                """,
                (scope, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_result_row_to_dict(r) for r in rows]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _write_result(
    *,
    result_id: str,
    scope: str,
    document_id: str | None,
    document_version: int | None,
    trigger: str,
    ndcg_5: float | None,
    mrr: float | None = None,
    p_5: float | None = None,
    r_5: float | None = None,
    delta: float | None,
    verdict: str,
) -> None:
    """Append one row to evaluation_results with the full metric set.

    All four metrics (ndcg_5, mrr, p_5, r_5) are persisted so historical
    rows contain a complete quality snapshot, not just the gate metric.
    Never raises — logs errors instead.
    """
    conn = _get_db_conn()
    if conn is None:
        logger.warning("evaluation: DB unavailable — result not persisted verdict=%s scope=%s", verdict, scope)
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluation_results
                    (id, scope, document_id, document_version, trigger,
                     ndcg_5, mrr, p_5, r_5, delta_ndcg_5, verdict)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (result_id, scope, document_id, document_version, trigger,
                 ndcg_5, mrr, p_5, r_5, delta, verdict),
            )
        conn.commit()
    except Exception as exc:
        logger.error("evaluation: failed to write result scope=%s verdict=%s: %s", scope, verdict, exc)
    finally:
        conn.close()


def _result_row_to_dict(row) -> dict:
    def _iso(dt):
        return dt.isoformat() if dt is not None else None
    return {
        "id": str(row[0]),
        "scope": row[1],
        "document_id": row[2],
        "document_version": row[3],
        "run_at": _iso(row[4]),
        "trigger": row[5],
        "ndcg_5": row[6],
        "mrr": row[7],
        "p_5": row[8],
        "r_5": row[9],
        "delta_ndcg_5": row[10],
        "verdict": row[11],
        "operator_override": row[12],
        "operator_note": row[13],
        "operator_identity": row[14],
    }
