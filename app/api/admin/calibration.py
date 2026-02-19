"""
Harmonia V3 — Admin Calibration API

Endpoints for managing the ground-truth calibration pipeline:
  - Reviewing pending Gemini scoring examples
  - Approving, correcting, or rejecting examples
  - Bulk review operations
  - Coverage and drift statistics
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.calibration import CalibrationExample
from app.schemas.report import (
    CalibrationBulkReviewRequest,
    CalibrationExampleResponse,
    CalibrationReviewRequest,
    CalibrationStatsResponse,
)

logger = structlog.get_logger("harmonia.api.admin.calibration")

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# GET /queue — Pending review queue
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/queue",
    response_model=list[CalibrationExampleResponse],
    summary="Get pending review queue",
)
async def get_review_queue(
    review_status: str = Query(
        "pending",
        alias="status",
        description="Filter by review status: pending, approved, corrected, rejected",
    ),
    question_number: int | None = Query(
        None,
        ge=1,
        le=6,
        description="Filter by question number (1-6)",
    ),
    sin: str | None = Query(
        None,
        description="Filter by sin dimension",
    ),
    limit: int = Query(50, ge=1, le=200, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    db: AsyncSession = Depends(get_db),
) -> list[CalibrationExample]:
    """Return calibration examples filtered by status, question, and sin.

    Defaults to showing pending items, which are the primary review
    queue for human validators.
    """
    log = logger.bind(
        review_status=review_status,
        question_number=question_number,
        sin=sin,
    )
    log.info("get_review_queue", limit=limit, offset=offset)

    stmt = select(CalibrationExample).where(
        CalibrationExample.review_status == review_status
    )

    if question_number is not None:
        stmt = stmt.where(CalibrationExample.question_number == question_number)

    if sin is not None:
        stmt = stmt.where(CalibrationExample.sin == sin)

    stmt = (
        stmt.order_by(CalibrationExample.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    result = await db.execute(stmt)
    examples = result.scalars().all()

    log.info("get_review_queue_complete", count=len(examples))
    return list(examples)


# ──────────────────────────────────────────────────────────────────────────────
# POST /{example_id}/review — Submit review action
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/{example_id}/review",
    response_model=CalibrationExampleResponse,
    summary="Submit review action for a calibration example",
)
async def review_example(
    example_id: uuid.UUID,
    payload: CalibrationReviewRequest,
    db: AsyncSession = Depends(get_db),
) -> CalibrationExample:
    """Apply a review action (approve / correct / reject) to a single
    calibration example.

    For ``correct`` actions, a ``validated_score`` must be provided.
    The reviewer identity and optional notes are recorded for audit.
    """
    log = logger.bind(
        example_id=str(example_id),
        action=payload.action,
        reviewer=payload.reviewer,
    )
    log.info("review_example_start")

    # Validate action
    valid_actions = {"approve", "correct", "reject"}
    if payload.action not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action '{payload.action}'. Must be one of: {valid_actions}",
        )

    # Correction requires a validated_score
    if payload.action == "correct" and payload.validated_score is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A validated_score is required for 'correct' actions.",
        )

    # Fetch the example
    stmt = select(CalibrationExample).where(CalibrationExample.id == example_id)
    result = await db.execute(stmt)
    example = result.scalar_one_or_none()

    if example is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Calibration example {example_id} not found.",
        )

    # Apply review
    if payload.action == "approve":
        example.review_status = "approved"
        example.validated_score = example.gemini_raw_score  # Keep original
    elif payload.action == "correct":
        example.review_status = "corrected"
        example.validated_score = payload.validated_score
    elif payload.action == "reject":
        example.review_status = "rejected"

    example.validated_by = payload.reviewer
    example.validated_at = datetime.now(timezone.utc)
    example.review_notes = payload.notes

    await db.flush()

    log.info(
        "review_example_complete",
        new_status=example.review_status,
        validated_score=example.validated_score,
    )
    return example


# ──────────────────────────────────────────────────────────────────────────────
# POST /bulk-review — Batch review multiple examples
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/bulk-review",
    summary="Batch review multiple calibration examples",
)
async def bulk_review(
    payload: CalibrationBulkReviewRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Apply review actions to multiple calibration examples in a single
    request.

    Returns a summary of how many examples were successfully reviewed
    versus how many failed (e.g. not found, invalid action).
    """
    log = logger.bind(
        reviewer=payload.reviewer,
        review_count=len(payload.reviews),
    )
    log.info("bulk_review_start")

    valid_actions = {"approve", "correct", "reject"}
    results: list[dict] = []
    success_count = 0
    error_count = 0

    for item in payload.reviews:
        item_log = log.bind(example_id=str(item.example_id), action=item.action)

        # Validate action
        if item.action not in valid_actions:
            results.append({
                "example_id": str(item.example_id),
                "status": "error",
                "detail": f"Invalid action '{item.action}'",
            })
            error_count += 1
            continue

        # Correction requires validated_score
        if item.action == "correct" and item.validated_score is None:
            results.append({
                "example_id": str(item.example_id),
                "status": "error",
                "detail": "validated_score required for 'correct' action",
            })
            error_count += 1
            continue

        # Fetch example
        stmt = select(CalibrationExample).where(
            CalibrationExample.id == item.example_id
        )
        result = await db.execute(stmt)
        example = result.scalar_one_or_none()

        if example is None:
            results.append({
                "example_id": str(item.example_id),
                "status": "error",
                "detail": "Not found",
            })
            error_count += 1
            continue

        # Apply review
        if item.action == "approve":
            example.review_status = "approved"
            example.validated_score = example.gemini_raw_score
        elif item.action == "correct":
            example.review_status = "corrected"
            example.validated_score = item.validated_score
        elif item.action == "reject":
            example.review_status = "rejected"

        example.validated_by = payload.reviewer
        example.validated_at = datetime.now(timezone.utc)
        example.review_notes = item.notes

        results.append({
            "example_id": str(item.example_id),
            "status": "reviewed",
            "new_review_status": example.review_status,
        })
        success_count += 1
        item_log.info("bulk_item_reviewed")

    await db.flush()

    log.info(
        "bulk_review_complete",
        success=success_count,
        errors=error_count,
    )

    return {
        "total": len(payload.reviews),
        "success": success_count,
        "errors": error_count,
        "results": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# GET /stats — Calibration coverage map and statistics
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/stats",
    response_model=CalibrationStatsResponse,
    summary="Calibration statistics and coverage map",
)
async def get_calibration_stats(
    db: AsyncSession = Depends(get_db),
) -> CalibrationStatsResponse:
    """Return calibration coverage statistics.

    Includes total counts by review status, a coverage map showing how
    many validated examples exist per (question_number, sin) cell, and
    the average magnitude of corrections applied.
    """
    logger.info("get_calibration_stats")

    # ── Total counts by status ────────────────────────────────────────────
    count_stmt = select(
        CalibrationExample.review_status,
        func.count().label("cnt"),
    ).group_by(CalibrationExample.review_status)
    count_result = await db.execute(count_stmt)
    status_counts = {row.review_status: row.cnt for row in count_result.all()}

    total = sum(status_counts.values())
    pending = status_counts.get("pending", 0)
    approved = status_counts.get("approved", 0)
    corrected = status_counts.get("corrected", 0)
    rejected = status_counts.get("rejected", 0)

    # ── Coverage map: (question_number, sin) -> count of approved/corrected
    coverage_stmt = select(
        CalibrationExample.question_number,
        CalibrationExample.sin,
        func.count().label("cnt"),
    ).where(
        CalibrationExample.review_status.in_(["approved", "corrected"])
    ).group_by(
        CalibrationExample.question_number,
        CalibrationExample.sin,
    )
    coverage_result = await db.execute(coverage_stmt)

    coverage_map: dict[str, int] = {}
    for row in coverage_result.all():
        key = f"q{row.question_number}_{row.sin}"
        coverage_map[key] = row.cnt

    # ── Average correction magnitude ──────────────────────────────────────
    correction_stmt = select(
        func.avg(
            func.abs(
                CalibrationExample.validated_score - CalibrationExample.gemini_raw_score
            )
        ).label("avg_magnitude")
    ).where(
        CalibrationExample.review_status == "corrected",
        CalibrationExample.validated_score.isnot(None),
    )
    correction_result = await db.execute(correction_stmt)
    avg_correction = correction_result.scalar()
    avg_correction_magnitude = round(float(avg_correction), 4) if avg_correction is not None else None

    logger.info(
        "get_calibration_stats_complete",
        total=total,
        pending=pending,
        approved=approved,
        corrected=corrected,
    )

    return CalibrationStatsResponse(
        total=total,
        pending=pending,
        approved=approved,
        corrected=corrected,
        rejected=rejected,
        coverage_map=coverage_map,
        avg_correction_magnitude=avg_correction_magnitude,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /effectiveness — Drift report and convergence metrics
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/effectiveness",
    summary="Calibration effectiveness — drift and convergence metrics",
)
async def get_calibration_effectiveness(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return calibration effectiveness metrics.

    Includes drift analysis (how much Gemini's raw scores deviate from
    validated scores over time), convergence metrics (whether corrections
    are decreasing), and per-sin accuracy breakdowns.
    """
    logger.info("get_calibration_effectiveness")

    # ── Per-sin correction statistics ─────────────────────────────────────
    sin_stats_stmt = select(
        CalibrationExample.sin,
        func.count().label("total"),
        func.count(CalibrationExample.validated_score).label("validated"),
        func.avg(
            func.abs(
                CalibrationExample.validated_score - CalibrationExample.gemini_raw_score
            )
        ).label("avg_drift"),
        func.max(
            func.abs(
                CalibrationExample.validated_score - CalibrationExample.gemini_raw_score
            )
        ).label("max_drift"),
    ).where(
        CalibrationExample.review_status.in_(["approved", "corrected"]),
    ).group_by(
        CalibrationExample.sin,
    )
    sin_result = await db.execute(sin_stats_stmt)

    per_sin_metrics: dict[str, dict] = {}
    for row in sin_result.all():
        per_sin_metrics[row.sin] = {
            "total_reviewed": row.total,
            "validated_count": row.validated,
            "avg_drift": round(float(row.avg_drift), 4) if row.avg_drift else 0.0,
            "max_drift": round(float(row.max_drift), 4) if row.max_drift else 0.0,
        }

    # ── Approval rate (approved / (approved + corrected + rejected)) ──────
    total_reviewed_stmt = select(func.count()).where(
        CalibrationExample.review_status.in_(["approved", "corrected", "rejected"])
    )
    total_reviewed = (await db.execute(total_reviewed_stmt)).scalar() or 0

    approved_stmt = select(func.count()).where(
        CalibrationExample.review_status == "approved"
    )
    approved_count = (await db.execute(approved_stmt)).scalar() or 0

    approval_rate = (
        round(approved_count / total_reviewed, 4)
        if total_reviewed > 0
        else None
    )

    # ── Overall drift (mean of per-sin avg_drift) ─────────────────────────
    drift_values = [
        m["avg_drift"] for m in per_sin_metrics.values() if m["avg_drift"] > 0
    ]
    overall_avg_drift = (
        round(sum(drift_values) / len(drift_values), 4)
        if drift_values
        else 0.0
    )

    response = {
        "total_reviewed": total_reviewed,
        "approval_rate": approval_rate,
        "overall_avg_drift": overall_avg_drift,
        "per_sin_metrics": per_sin_metrics,
        "convergence_assessment": (
            "good" if overall_avg_drift < 0.5
            else "moderate" if overall_avg_drift < 1.0
            else "needs_attention"
        ),
    }

    logger.info(
        "get_calibration_effectiveness_complete",
        overall_avg_drift=overall_avg_drift,
        approval_rate=approval_rate,
    )
    return response
