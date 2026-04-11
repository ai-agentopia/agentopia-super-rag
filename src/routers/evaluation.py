"""Evaluation API router.

Endpoints:
  GET  /evaluation/baselines            — list all per-scope baselines
  GET  /evaluation/baselines/{scope}    — get baseline for one scope
  POST /evaluation/baselines/{scope}    — establish (or re-establish) baseline by running golden questions
  GET  /evaluation/results              — list evaluation run history (filterable by scope)
  GET  /evaluation/results/{result_id} — get one result
  POST /evaluation/results/{result_id}/override — operator override on a blocked result
  GET  /evaluation/questions/{scope}   — list golden questions for scope
  POST /evaluation/questions/{scope}   — add a golden question
  DELETE /evaluation/questions/{question_id} — remove a golden question
  POST /evaluation/run/{scope}          — run benchmark and check regression (manual trigger)

All endpoints require internal token auth (X-Internal-Token).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.guards import require_internal_auth
from services import evaluation as eval_svc
from services.knowledge import get_knowledge_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/evaluation",
    tags=["evaluation"],
    dependencies=[Depends(require_internal_auth)],
)


# ── Baselines ─────────────────────────────────────────────────────────────────


@router.get("/baselines")
async def list_baselines() -> dict:
    """List all per-scope evaluation baselines."""
    return {"baselines": eval_svc.list_baselines()}


@router.get("/baselines/{scope:path}")
async def get_baseline(scope: str) -> dict:
    """Get evaluation baseline for a specific scope."""
    b = eval_svc.get_baseline(scope)
    if b is None:
        raise HTTPException(
            status_code=404,
            detail=f"No baseline established for scope '{scope}'. POST /evaluation/baselines/{scope} to create one.",
        )
    return b


class EstablishBaselineRequest(BaseModel):
    notes: str = ""


@router.post("/baselines/{scope:path}", status_code=201)
async def establish_baseline(scope: str, body: EstablishBaselineRequest = EstablishBaselineRequest()) -> dict:
    """Run golden questions for this scope and store the result as the baseline.

    Replaces any existing baseline for this scope.
    Requires golden questions to exist for the scope — see POST /evaluation/questions/{scope}.
    """
    svc = get_knowledge_service()
    try:
        result = eval_svc.establish_baseline(scope=scope, svc=svc, notes=body.notes)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("establish_baseline failed scope=%s: %s", scope, exc)
        raise HTTPException(status_code=500, detail=f"Baseline establishment failed: {exc}")
    logger.info("baseline established scope=%s ndcg_5=%.4f", scope, result.get("ndcg_5"))
    return result


# ── Results ───────────────────────────────────────────────────────────────────


@router.get("/results")
async def list_results(scope: str = "", limit: int = 50) -> dict:
    """List evaluation results, optionally filtered by scope."""
    if not scope:
        raise HTTPException(status_code=422, detail="scope query parameter is required")
    results = eval_svc.get_evaluation_results(scope=scope, limit=limit)
    return {"scope": scope, "results": results}


class OverrideRequest(BaseModel):
    operator_note: str = ""
    operator_identity: str = ""
    """Who is performing the override. Stored in evaluation_results for audit attribution.
    Should be an operator identifier (email, username, or service account name).
    Required when operator accountability is needed — empty string is accepted but not recommended."""


@router.post("/results/{result_id}/override")
async def override_result(result_id: str, body: OverrideRequest) -> dict:
    """Record an operator override for a blocked evaluation result.

    The verdict changes from 'blocked' to 'overridden' in the evaluation_results table.
    operator_identity is stored for audit attribution.
    This does not change the document's active status — it only records that the operator
    has explicitly accepted the regression.
    """
    success = eval_svc.record_operator_override(
        result_id=result_id,
        operator_note=body.operator_note,
        operator_identity=body.operator_identity,
    )
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Result '{result_id}' not found or not in 'blocked' state.",
        )
    logger.info(
        "evaluation: operator override result_id=%s identity=%s",
        result_id, body.operator_identity or "(unset)",
    )
    return {
        "result_id": result_id,
        "verdict": "overridden",
        "operator_note": body.operator_note,
        "operator_identity": body.operator_identity,
    }


# ── Golden questions ──────────────────────────────────────────────────────────


@router.get("/questions/{scope:path}")
async def list_questions(scope: str) -> dict:
    """List golden questions for a scope."""
    return {"scope": scope, "questions": eval_svc.list_golden_questions(scope)}


class AddQuestionRequest(BaseModel):
    query: str
    expected_sources: list[dict]
    """
    Format: [{"source": "filename.pdf", "relevance": 2}, ...]
    relevance: 2=fully relevant, 1=partially relevant, 0=not relevant
    "source" must match the document source key used in document_records
    (typically the filename passed during ingest).
    """
    weight: float = 1.0
    created_by: str = ""


@router.post("/questions/{scope:path}", status_code=201)
async def add_question(scope: str, body: AddQuestionRequest) -> dict:
    """Add a golden question to a scope's evaluation set."""
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if not body.expected_sources:
        raise HTTPException(
            status_code=422,
            detail="expected_sources must not be empty — provide at least one source with relevance grade",
        )
    try:
        qid = eval_svc.add_golden_question(
            scope=scope,
            query=body.query,
            expected_sources=body.expected_sources,
            weight=body.weight,
            created_by=body.created_by,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"question_id": qid, "scope": scope, "query": body.query}


@router.delete("/questions/{question_id}")
async def delete_question(question_id: str) -> dict:
    """Remove a golden question by id."""
    deleted = eval_svc.delete_golden_question(question_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Question '{question_id}' not found")
    return {"question_id": question_id, "deleted": True}


# ── Manual benchmark trigger ──────────────────────────────────────────────────


@router.post("/run/{scope:path}")
async def run_evaluation(scope: str) -> dict:
    """Manually trigger a benchmark run for a scope and check regression against baseline.

    Returns the regression result including verdict and metrics.
    Does not alter the active document — only records the evaluation result.
    """
    svc = get_knowledge_service()
    result = eval_svc.check_regression(scope=scope, svc=svc, trigger="manual")
    return {
        "scope": result.scope,
        "verdict": result.verdict,
        "ndcg_5": result.ndcg_5,
        "delta_ndcg_5": result.delta_ndcg_5,
        "baseline_ndcg_5": result.baseline_ndcg_5,
        "result_id": result.result_id,
        "message": result.message,
    }
