"""
Harmonia V3 — Visual Intelligence API

Endpoints for the visual attractiveness pipeline: calibration image sets,
MetaFBP calibration, target scoring, and swipe recording.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.match import Swipe
from app.models.user import User
from app.schemas.match import (
    SwipeCreate,
    SwipeResponse,
    VisualCalibrateRequest,
    VisualScoreRequest,
)
from app.services.visual_service import VisualService

logger = structlog.get_logger("harmonia.api.visual")

router = APIRouter()

# ── Service singleton ─────────────────────────────────────────────────────────

_visual_service: VisualService | None = None


def _get_visual_service() -> VisualService:
    global _visual_service
    if _visual_service is None:
        _visual_service = VisualService()
    return _visual_service


# ── Calibration image set (static reference data) ────────────────────────────

_CALIBRATION_IMAGES: list[dict] = [
    {"image_id": f"cal_{i:03d}", "image_path": f"calibration/image_{i:03d}.jpg"}
    for i in range(1, 31)
]


# ──────────────────────────────────────────────────────────────────────────────
# GET /calibration-set — Return calibration image set
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/calibration-set",
    summary="Get calibration image set",
)
async def get_calibration_set() -> dict:
    """Return the set of calibration images used for MetaFBP adaptation.

    The client displays these images and collects 1-5 ratings from the user
    before calling ``POST /calibrate``.
    """
    logger.info("get_calibration_set")
    return {
        "images": _CALIBRATION_IMAGES,
        "total": len(_CALIBRATION_IMAGES),
        "instructions": (
            "Rate each image on a scale of 1 (not attractive) to 5 "
            "(very attractive). All images must be rated."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /calibrate — Run MetaFBP calibration
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/calibrate",
    status_code=status.HTTP_201_CREATED,
    summary="Calibrate visual preference model",
)
async def calibrate(
    payload: VisualCalibrateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Accept user ratings on calibration images, trigger MetaFBP inner-loop
    adaptation, and store the personalised model weights.

    The ``ratings`` list must contain dicts with ``image_id`` and ``rating``
    (1-5) keys.
    """
    log = logger.bind(
        user_id=str(payload.user_id),
        rating_count=len(payload.ratings),
    )
    log.info("calibrate_start")

    # Verify user exists
    stmt = select(User).where(User.id == payload.user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {payload.user_id} not found.",
        )

    # Build the ratings list with image_path for the service
    calibration_images_map = {img["image_id"]: img["image_path"] for img in _CALIBRATION_IMAGES}

    enriched_ratings: list[dict] = []
    for r in payload.ratings:
        image_id = r.get("image_id", "")
        rating = r.get("rating", 3)
        image_path = calibration_images_map.get(image_id, f"calibration/{image_id}.jpg")
        enriched_ratings.append({
            "image_id": image_id,
            "image_path": image_path,
            "rating": rating,
        })

    visual_svc = _get_visual_service()

    try:
        calibration_result = await visual_svc.calibrate_user(
            user_id=str(payload.user_id),
            ratings=enriched_ratings,
            db_session=db,
        )
    except RuntimeError as exc:
        log.error("calibrate_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    log.info("calibrate_complete")
    return calibration_result


# ──────────────────────────────────────────────────────────────────────────────
# POST /score — Score a target user
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/score",
    summary="Score a target user's attractiveness",
)
async def score_target(
    payload: VisualScoreRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Score a target user through the calibrated visual preference model.

    Returns the S_vis score and component breakdown.  Requires the
    requesting user to have been calibrated first.
    """
    log = logger.bind(
        user_id=str(payload.user_id),
        target_user_id=str(payload.target_user_id),
    )
    log.info("score_target_start")

    # Verify both users exist
    for uid, label in [
        (payload.user_id, "Requesting user"),
        (payload.target_user_id, "Target user"),
    ]:
        stmt = select(User).where(User.id == uid)
        result = await db.execute(stmt)
        if result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{label} {uid} not found.",
            )

    # Fetch the target's primary photo
    target_stmt = select(User).where(User.id == payload.target_user_id)
    target_result = await db.execute(target_stmt)
    target_user = target_result.scalar_one_or_none()
    target_photos = target_user.photos or []

    if not target_photos:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Target user has no photos to score.",
        )

    target_image_path = target_photos[0]

    visual_svc = _get_visual_service()

    try:
        score_result = await visual_svc.score_target(
            user_id=str(payload.user_id),
            target_image_path=target_image_path,
            db_session=db,
        )
    except RuntimeError as exc:
        log.error("score_target_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    log.info("score_target_complete", s_vis=score_result.get("s_vis"))
    return score_result


# ──────────────────────────────────────────────────────────────────────────────
# POST /swipe — Record swipe action
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/swipe",
    response_model=SwipeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a swipe action",
)
async def record_swipe(
    payload: SwipeCreate,
    db: AsyncSession = Depends(get_db),
) -> SwipeResponse:
    """Record a swipe action (left / right / superlike) and check for
    mutual matches.

    If a reciprocal right-swipe or superlike already exists for the
    target -> swiper direction, ``is_mutual_match`` is returned as True.
    """
    log = logger.bind(
        swiper_id=str(payload.swiper_id),
        target_id=str(payload.target_id),
        direction=payload.direction,
    )
    log.info("record_swipe_start")

    # Verify both users exist
    for uid, label in [
        (payload.swiper_id, "Swiper"),
        (payload.target_id, "Target"),
    ]:
        stmt = select(User).where(User.id == uid)
        result = await db.execute(stmt)
        if result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{label} user {uid} not found.",
            )

    # Check if a swipe already exists (upsert)
    existing_stmt = select(Swipe).where(
        Swipe.swiper_id == payload.swiper_id,
        Swipe.target_id == payload.target_id,
    )
    existing_result = await db.execute(existing_stmt)
    existing_swipe = existing_result.scalar_one_or_none()

    if existing_swipe is not None:
        existing_swipe.direction = payload.direction
        log.info("swipe_updated")
    else:
        new_swipe = Swipe(
            swiper_id=payload.swiper_id,
            target_id=payload.target_id,
            direction=payload.direction,
        )
        db.add(new_swipe)
        log.info("swipe_created")

    await db.flush()

    # Check for mutual match (reciprocal right-swipe or superlike)
    is_mutual = False
    match_id = None

    if payload.direction in ("right", "superlike"):
        reciprocal_stmt = select(Swipe).where(
            Swipe.swiper_id == payload.target_id,
            Swipe.target_id == payload.swiper_id,
            Swipe.direction.in_(["right", "superlike"]),
        )
        reciprocal_result = await db.execute(reciprocal_stmt)
        reciprocal = reciprocal_result.scalar_one_or_none()

        if reciprocal is not None:
            is_mutual = True
            log.info("mutual_match_detected")

    log.info("record_swipe_complete", is_mutual_match=is_mutual)

    return SwipeResponse(
        status="recorded",
        is_mutual_match=is_mutual,
        match_id=match_id,
    )
