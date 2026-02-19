"""
Harmonia V3 — Matching API

Endpoints for triggering full Whole-the-Match (WtM) calculations,
retrieving match details, and listing all matches for a user.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.match import Match
from app.models.profile import PersonalityProfile
from app.models.user import User
from app.schemas.match import MatchCalculateResponse, MatchListItem
from app.services.hla_service import HLAService
from app.services.similarity_service import SimilarityService
from app.services.visual_service import VisualService

logger = structlog.get_logger("harmonia.api.matching")

router = APIRouter()

# ── Service singletons ────────────────────────────────────────────────────────

_visual_service: VisualService | None = None
_similarity_service: SimilarityService | None = None
_hla_service: HLAService | None = None


def _get_visual_service() -> VisualService:
    global _visual_service
    if _visual_service is None:
        _visual_service = VisualService()
    return _visual_service


def _get_similarity_service() -> SimilarityService:
    global _similarity_service
    if _similarity_service is None:
        _similarity_service = SimilarityService()
    return _similarity_service


def _get_hla_service() -> HLAService:
    global _hla_service
    if _hla_service is None:
        _hla_service = HLAService()
    return _hla_service


# ──────────────────────────────────────────────────────────────────────────────
# POST /calculate/{user_a_id}/{user_b_id} — Trigger full match calculation
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/calculate/{user_a_id}/{user_b_id}",
    response_model=MatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate full match between two users",
)
async def calculate_match(
    user_a_id: uuid.UUID,
    user_b_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> MatchCalculateResponse:
    """Trigger a full Whole-the-Match calculation for a user pair.

    Orchestrates the three scoring dimensions:
      1. **S_vis** — Visual attractiveness (both directions)
      2. **S_psych** — Personality similarity
      3. **S_bio** — HLA biological compatibility (optional)

    The WtM formula combines all three into a single composite score.
    A customer-facing summary is generated and the match is persisted.
    """
    log = logger.bind(
        user_a_id=str(user_a_id),
        user_b_id=str(user_b_id),
    )
    log.info("calculate_match_start")

    settings = get_settings()

    # ── Verify both users exist ───────────────────────────────────────────
    users: dict[uuid.UUID, User] = {}
    for uid, label in [(user_a_id, "User A"), (user_b_id, "User B")]:
        stmt = select(User).where(User.id == uid)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{label} ({uid}) not found.",
            )
        users[uid] = user

    # ── Check if match already exists ─────────────────────────────────────
    existing_stmt = select(Match).where(
        or_(
            (Match.user_a_id == user_a_id) & (Match.user_b_id == user_b_id),
            (Match.user_a_id == user_b_id) & (Match.user_b_id == user_a_id),
        )
    )
    existing_result = await db.execute(existing_stmt)
    existing_match = existing_result.scalar_one_or_none()

    if existing_match is not None:
        log.info("match_already_exists", match_id=str(existing_match.id))
        return MatchCalculateResponse(
            match_id=existing_match.id,
            user_a_id=existing_match.user_a_id,
            user_b_id=existing_match.user_b_id,
            wtm_score=existing_match.wtm_score,
            s_vis_a_to_b=existing_match.s_vis_a_to_b,
            s_vis_b_to_a=existing_match.s_vis_b_to_a,
            s_psych=existing_match.s_psych,
            s_bio=existing_match.s_bio,
            customer_summary=existing_match.customer_summary or {},
        )

    # ── 1. S_vis — Visual scoring (both directions) ───────────────────────
    visual_svc = _get_visual_service()

    # A -> B
    photos_b = users[user_b_id].photos or []
    if photos_b:
        try:
            vis_a_to_b_result = await visual_svc.score_target(
                user_id=str(user_a_id),
                target_image_path=photos_b[0],
                db_session=db,
            )
            s_vis_a_to_b = vis_a_to_b_result.get("s_vis", 50.0)
        except Exception:
            log.warning("s_vis_a_to_b_failed")
            s_vis_a_to_b = 50.0
    else:
        s_vis_a_to_b = 50.0

    # B -> A
    photos_a = users[user_a_id].photos or []
    if photos_a:
        try:
            vis_b_to_a_result = await visual_svc.score_target(
                user_id=str(user_b_id),
                target_image_path=photos_a[0],
                db_session=db,
            )
            s_vis_b_to_a = vis_b_to_a_result.get("s_vis", 50.0)
        except Exception:
            log.warning("s_vis_b_to_a_failed")
            s_vis_b_to_a = 50.0
    else:
        s_vis_b_to_a = 50.0

    # Average visual score for the WtM composite
    s_vis = (s_vis_a_to_b + s_vis_b_to_a) / 2.0

    log.info(
        "visual_scores",
        s_vis_a_to_b=round(s_vis_a_to_b, 4),
        s_vis_b_to_a=round(s_vis_b_to_a, 4),
    )

    # ── 2. S_psych — Personality similarity ───────────────────────────────
    similarity_svc = _get_similarity_service()

    profile_a_stmt = select(PersonalityProfile).where(
        PersonalityProfile.user_id == user_a_id
    )
    profile_b_stmt = select(PersonalityProfile).where(
        PersonalityProfile.user_id == user_b_id
    )
    profile_a = (await db.execute(profile_a_stmt)).scalar_one_or_none()
    profile_b = (await db.execute(profile_b_stmt)).scalar_one_or_none()

    if profile_a is None or profile_b is None:
        missing = []
        if profile_a is None:
            missing.append(str(user_a_id))
        if profile_b is None:
            missing.append(str(user_b_id))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Personality profiles required for both users.",
                "missing_profiles": missing,
            },
        )

    similarity_result = similarity_svc.calculate_similarity(
        profile_a={"sins": profile_a.sins, "quality_tier": profile_a.quality_tier},
        profile_b={"sins": profile_b.sins, "quality_tier": profile_b.quality_tier},
    )
    s_psych = similarity_result["adjusted_score"] * 100.0  # Convert to 0-100 scale

    log.info("personality_score", s_psych=round(s_psych, 4))

    # ── 3. S_bio — HLA biological compatibility (optional) ────────────────
    hla_svc = _get_hla_service()
    s_bio = None

    try:
        hla_result = await hla_svc.calculate_compatibility(
            user_a_id=str(user_a_id),
            user_b_id=str(user_b_id),
            db_session=db,
        )
        if hla_result.get("error") is None:
            s_bio = hla_result.get("s_bio")
            log.info("hla_score", s_bio=s_bio)
        else:
            log.info("hla_data_not_available")
    except Exception:
        log.warning("hla_calculation_failed")

    # ── WtM composite score ───────────────────────────────────────────────
    if s_bio is not None:
        wtm_score = (
            (s_vis * settings.VISUAL_WEIGHT)
            + (s_psych * settings.PERSONALITY_WEIGHT)
            + (s_bio * settings.HLA_WEIGHT)
        )
    else:
        # Without HLA, redistribute weight between visual and personality
        vis_adj = settings.VISUAL_WEIGHT / (settings.VISUAL_WEIGHT + settings.PERSONALITY_WEIGHT)
        psych_adj = settings.PERSONALITY_WEIGHT / (settings.VISUAL_WEIGHT + settings.PERSONALITY_WEIGHT)
        wtm_score = (s_vis * vis_adj) + (s_psych * psych_adj)

    wtm_score = max(0.0, min(100.0, wtm_score))

    # ── Generate customer summary ─────────────────────────────────────────
    hla_display = similarity_svc.get_hla_display(s_bio)
    match_card = similarity_svc.assemble_match_card(
        similarity_result=similarity_result,
        hla_result=hla_display,
    )

    customer_summary = {
        "wtm_display_score": round(wtm_score),
        "match_card": match_card,
    }

    # ── Persist the match ─────────────────────────────────────────────────
    new_match = Match(
        user_a_id=user_a_id,
        user_b_id=user_b_id,
        s_vis_a_to_b=round(s_vis_a_to_b, 4),
        s_vis_b_to_a=round(s_vis_b_to_a, 4),
        s_psych=round(s_psych, 4),
        s_bio=round(s_bio, 4) if s_bio is not None else None,
        wtm_score=round(wtm_score, 4),
        reasoning_chain={
            "s_vis_a_to_b": round(s_vis_a_to_b, 4),
            "s_vis_b_to_a": round(s_vis_b_to_a, 4),
            "s_psych": round(s_psych, 4),
            "s_bio": round(s_bio, 4) if s_bio is not None else None,
            "weights": {
                "visual": settings.VISUAL_WEIGHT,
                "personality": settings.PERSONALITY_WEIGHT,
                "hla": settings.HLA_WEIGHT,
            },
            "similarity_breakdown": similarity_result.get("breakdown", []),
        },
        customer_summary=customer_summary,
    )
    db.add(new_match)
    await db.flush()

    log.info(
        "calculate_match_complete",
        match_id=str(new_match.id),
        wtm_score=round(wtm_score, 4),
    )

    return MatchCalculateResponse(
        match_id=new_match.id,
        user_a_id=user_a_id,
        user_b_id=user_b_id,
        wtm_score=round(wtm_score, 4),
        s_vis_a_to_b=round(s_vis_a_to_b, 4),
        s_vis_b_to_a=round(s_vis_b_to_a, 4),
        s_psych=round(s_psych, 4),
        s_bio=round(s_bio, 4) if s_bio is not None else None,
        customer_summary=customer_summary,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /{match_id} — Get match details
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{match_id}",
    response_model=MatchCalculateResponse,
    summary="Get match details by ID",
)
async def get_match(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> MatchCalculateResponse:
    """Retrieve a match by its UUID, including all scoring components and
    the customer summary."""
    log = logger.bind(match_id=str(match_id))
    log.info("get_match")

    stmt = select(Match).where(Match.id == match_id)
    result = await db.execute(stmt)
    match = result.scalar_one_or_none()

    if match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match {match_id} not found.",
        )

    return MatchCalculateResponse(
        match_id=match.id,
        user_a_id=match.user_a_id,
        user_b_id=match.user_b_id,
        wtm_score=match.wtm_score,
        s_vis_a_to_b=match.s_vis_a_to_b,
        s_vis_b_to_a=match.s_vis_b_to_a,
        s_psych=match.s_psych,
        s_bio=match.s_bio,
        customer_summary=match.customer_summary or {},
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /list/{user_id} — List all matches for a user
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/list/{user_id}",
    response_model=list[MatchListItem],
    summary="List all matches for a user",
)
async def list_user_matches(
    user_id: uuid.UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[MatchListItem]:
    """Return all matches involving a specific user, sorted by WtM score
    descending.  Each item includes the other user's ID and display name."""
    log = logger.bind(user_id=str(user_id))
    log.info("list_user_matches", limit=limit, offset=offset)

    # Verify user exists
    user_stmt = select(User).where(User.id == user_id)
    user_result = await db.execute(user_stmt)
    if user_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    # Fetch matches where user is either side
    stmt = (
        select(Match)
        .where(or_(Match.user_a_id == user_id, Match.user_b_id == user_id))
        .order_by(Match.wtm_score.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    matches = result.scalars().all()

    items: list[MatchListItem] = []
    for m in matches:
        # Determine the "other" user
        if m.user_a_id == user_id:
            other_user = m.user_b
            other_id = m.user_b_id
        else:
            other_user = m.user_a
            other_id = m.user_a_id

        other_name = other_user.display_name if other_user else "Unknown"

        items.append(MatchListItem(
            match_id=m.id,
            other_user_id=other_id,
            other_user_name=other_name,
            wtm_score=m.wtm_score,
            created_at=m.created_at,
        ))

    log.info("list_user_matches_complete", count=len(items))
    return items
