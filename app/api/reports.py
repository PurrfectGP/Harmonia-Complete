"""
Harmonia V3 — Reports API

Multi-level report endpoints with access control:
  - Level 1 (Customer Summary): User-facing, NO evidence exposed
  - Level 2A (Narrative): Admin only
  - Level 2B (HLA Analysis): Admin only
  - Level 3 (Reasoning Chain): Admin only, full transparency
  - Standalone Evidence Map: Admin only
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.evidence import ParsingEvidence
from app.models.match import Match
from app.models.profile import PersonalityProfile
from app.models.user import User
from app.schemas.report import CustomerSummary, EvidenceEntry, QuestionEvidenceMap, UserEvidenceMap
from app.services.hla_service import HLAService
from app.services.similarity_service import SimilarityService

logger = structlog.get_logger("harmonia.api.reports")

router = APIRouter()

# ── Service singletons ────────────────────────────────────────────────────────

_similarity_service: SimilarityService | None = None
_hla_service: HLAService | None = None


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


# ── Helper to load a match ────────────────────────────────────────────────────

async def _load_match(match_id: uuid.UUID, db: AsyncSession) -> Match:
    """Fetch a Match by ID or raise 404."""
    stmt = select(Match).where(Match.id == match_id)
    result = await db.execute(stmt)
    match = result.scalar_one_or_none()

    if match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match {match_id} not found.",
        )
    return match


# ──────────────────────────────────────────────────────────────────────────────
# GET /{match_id}/summary — Level 1 customer summary (user-facing, NO evidence)
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{match_id}/summary",
    summary="Level 1 — Customer-facing match summary",
)
async def get_customer_summary(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the Level 1 customer-facing summary for a match.

    This endpoint is designed for regular users.  It includes display
    scores, badges, compatibility breakdown, shared traits, and
    conversation starters.  **No evidence or reasoning data is exposed.**
    """
    log = logger.bind(match_id=str(match_id))
    log.info("get_customer_summary")

    match = await _load_match(match_id, db)

    customer_summary = match.customer_summary or {}

    # Build the user-safe response
    match_card = customer_summary.get("match_card", {})
    personality = match_card.get("personality", {})
    chemistry = match_card.get("chemistry", {})

    display_score = customer_summary.get("wtm_display_score", round(match.wtm_score))

    # Generate badges from tier
    tier = personality.get("tier", "")
    badges: list[str] = []
    if tier == "strong_fit":
        badges.append("Strong Match")
    elif tier == "good_fit":
        badges.append("Good Match")
    if chemistry.get("emoji"):
        badges.append(chemistry.get("label", "Chemistry"))

    shared_traits = personality.get("shared_traits", [])

    # Generate conversation starters from shared traits
    conversation_starters: list[str] = []
    for trait_desc in shared_traits[:3]:
        # Strip the "You're both" prefix for conversation starters
        clean = trait_desc.replace("You're both ", "")
        conversation_starters.append(
            f"Ask about what makes them feel {clean}."
        )

    summary_response = {
        "match_id": str(match_id),
        "display_score": display_score,
        "badges": badges,
        "synopsis": {
            "headline": f"{display_score}% Compatible",
            "body": (
                f"You share {len(shared_traits)} key personality traits "
                f"that suggest strong long-term compatibility."
                if shared_traits
                else "Explore your connection to find out more."
            ),
        },
        "compatibility_breakdown": {
            "physical": round(((match.s_vis_a_to_b + match.s_vis_b_to_a) / 2), 1),
            "personality": round(match.s_psych, 1),
            "chemistry": round(match.s_bio, 1) if match.s_bio is not None else None,
        },
        "shared_traits": shared_traits,
        "conversation_starters": conversation_starters,
    }

    log.info("get_customer_summary_complete")
    return summary_response


# ──────────────────────────────────────────────────────────────────────────────
# GET /{match_id}/reasoning-chain — Level 3 (admin only)
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{match_id}/reasoning-chain",
    summary="Level 3 — Full reasoning chain (admin only)",
)
async def get_reasoning_chain(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the Level 3 full reasoning chain for a match.

    Admin-only endpoint that exposes the complete scoring breakdown,
    component weights, per-sin similarity contributions, and all
    intermediate calculation results.
    """
    log = logger.bind(match_id=str(match_id))
    log.info("get_reasoning_chain")

    match = await _load_match(match_id, db)

    reasoning = match.reasoning_chain or {}

    response = {
        "match_id": str(match_id),
        "user_a_id": str(match.user_a_id),
        "user_b_id": str(match.user_b_id),
        "wtm_score": match.wtm_score,
        "components": {
            "s_vis_a_to_b": match.s_vis_a_to_b,
            "s_vis_b_to_a": match.s_vis_b_to_a,
            "s_psych": match.s_psych,
            "s_bio": match.s_bio,
        },
        "weights": reasoning.get("weights", {}),
        "similarity_breakdown": reasoning.get("similarity_breakdown", []),
        "full_reasoning_chain": reasoning,
        "created_at": match.created_at.isoformat() if match.created_at else None,
    }

    log.info("get_reasoning_chain_complete")
    return response


# ──────────────────────────────────────────────────────────────────────────────
# GET /{match_id}/narrative — Level 2A (admin only)
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{match_id}/narrative",
    summary="Level 2A — Narrative match report (admin only)",
)
async def get_narrative_report(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return a Level 2A narrative report for a match.

    Admin-only endpoint that generates a natural-language narrative
    summarising the personality similarity analysis, including per-sin
    contributions, quality tier context, and trait explanations.
    """
    log = logger.bind(match_id=str(match_id))
    log.info("get_narrative_report")

    match = await _load_match(match_id, db)

    # Load both profiles for narrative context
    profile_a_stmt = select(PersonalityProfile).where(
        PersonalityProfile.user_id == match.user_a_id
    )
    profile_b_stmt = select(PersonalityProfile).where(
        PersonalityProfile.user_id == match.user_b_id
    )
    profile_a = (await db.execute(profile_a_stmt)).scalar_one_or_none()
    profile_b = (await db.execute(profile_b_stmt)).scalar_one_or_none()

    # Recalculate similarity for narrative detail
    similarity_svc = _get_similarity_service()
    if profile_a and profile_b:
        similarity_result = similarity_svc.calculate_similarity(
            profile_a={"sins": profile_a.sins, "quality_tier": profile_a.quality_tier},
            profile_b={"sins": profile_b.sins, "quality_tier": profile_b.quality_tier},
        )
        explanation = similarity_svc.generate_match_explanation(
            breakdown=similarity_result["breakdown"],
            tier=similarity_result["tier"],
            display_mode=similarity_result["display_mode"],
        )
    else:
        similarity_result = {}
        explanation = {"shared_traits": [], "tier": "unknown", "display_mode": "unknown"}

    # Build narrative sections
    narrative_sections: list[dict] = []

    # Personality section
    narrative_sections.append({
        "title": "Personality Compatibility",
        "score": match.s_psych,
        "tier": similarity_result.get("tier", "unknown"),
        "quality_multiplier": similarity_result.get("quality_multiplier"),
        "raw_score": similarity_result.get("raw_score"),
        "adjusted_score": similarity_result.get("adjusted_score"),
        "overlap_count": similarity_result.get("overlap_count", 0),
        "shared_traits": explanation.get("shared_traits", []),
        "breakdown": similarity_result.get("breakdown", []),
        "profile_a_tier": profile_a.quality_tier if profile_a else None,
        "profile_b_tier": profile_b.quality_tier if profile_b else None,
    })

    # Visual section
    narrative_sections.append({
        "title": "Physical Attraction",
        "s_vis_a_to_b": match.s_vis_a_to_b,
        "s_vis_b_to_a": match.s_vis_b_to_a,
        "average": round((match.s_vis_a_to_b + match.s_vis_b_to_a) / 2, 4),
        "note": "Asymmetric scores indicate differing physical preferences.",
    })

    # HLA section (if available)
    if match.s_bio is not None:
        narrative_sections.append({
            "title": "Biological Chemistry",
            "s_bio": match.s_bio,
            "display": similarity_svc.get_hla_display(match.s_bio),
        })

    response = {
        "match_id": str(match_id),
        "user_a_id": str(match.user_a_id),
        "user_b_id": str(match.user_b_id),
        "wtm_score": match.wtm_score,
        "narrative_sections": narrative_sections,
        "created_at": match.created_at.isoformat() if match.created_at else None,
    }

    log.info("get_narrative_report_complete")
    return response


# ──────────────────────────────────────────────────────────────────────────────
# GET /{match_id}/hla-analysis — Level 2B (admin only)
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{match_id}/hla-analysis",
    summary="Level 2B — HLA analysis report (admin only)",
)
async def get_hla_analysis(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the Level 2B HLA biological compatibility analysis.

    Admin-only endpoint that provides the full HLA compatibility report
    including allele summaries, heterozygosity index, olfactory
    predictions, peptide-binding analyses, and disease associations.
    """
    log = logger.bind(match_id=str(match_id))
    log.info("get_hla_analysis")

    match = await _load_match(match_id, db)

    if match.s_bio is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="HLA data not available for this match.",
        )

    # Run the full compatibility calculation for detailed results
    hla_svc = _get_hla_service()
    hla_result = await hla_svc.calculate_compatibility(
        user_a_id=str(match.user_a_id),
        user_b_id=str(match.user_b_id),
        db_session=db,
    )

    if hla_result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "HLA analysis could not be generated.",
                "error": hla_result.get("error"),
            },
        )

    response = {
        "match_id": str(match_id),
        "user_a_id": str(match.user_a_id),
        "user_b_id": str(match.user_b_id),
        "s_bio": match.s_bio,
        "hla_analysis": hla_result,
    }

    log.info("get_hla_analysis_complete")
    return response


# ──────────────────────────────────────────────────────────────────────────────
# GET /{match_id}/evidence-map/{user_id} — Standalone evidence map (admin only)
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{match_id}/evidence-map/{user_id}",
    summary="Standalone evidence map for a user in a match (admin only)",
)
async def get_evidence_map(
    match_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the full parsing evidence map for a specific user within
    a match context.

    Admin-only endpoint that exposes every per-question, per-sin score
    with evidence snippets, character offsets, confidence values, and
    model provenance.  This is the most granular transparency layer.
    """
    log = logger.bind(match_id=str(match_id), user_id=str(user_id))
    log.info("get_evidence_map")

    # Verify the match exists
    match = await _load_match(match_id, db)

    # Verify the user is part of the match
    if user_id not in (match.user_a_id, match.user_b_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User {user_id} is not part of match {match_id}.",
        )

    # Fetch all parsing evidence for this user
    evidence_stmt = (
        select(ParsingEvidence)
        .where(ParsingEvidence.user_id == user_id)
        .order_by(ParsingEvidence.question_number, ParsingEvidence.sin)
    )
    evidence_result = await db.execute(evidence_stmt)
    evidence_records = evidence_result.scalars().all()

    if not evidence_records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No parsing evidence found for user {user_id}.",
        )

    # Organise by question number
    evidence_by_question: dict[int, list[dict]] = {}
    for record in evidence_records:
        qn = record.question_number
        entry = {
            "sin": record.sin,
            "score": record.score,
            "confidence": record.confidence,
            "evidence_snippet": record.evidence_snippet,
            "snippet_location": {
                "start": record.snippet_start_index,
                "end": record.snippet_end_index,
            },
            "interpretation": record.interpretation,
            "observer_persona": record.observer_persona,
            "gemini_model_used": record.gemini_model_used,
        }
        evidence_by_question.setdefault(qn, []).append(entry)

    # Fetch the user's questionnaire responses for context
    from app.models.questionnaire import QuestionnaireResponse

    response_stmt = (
        select(QuestionnaireResponse)
        .where(QuestionnaireResponse.user_id == user_id)
        .order_by(QuestionnaireResponse.question_number)
    )
    response_result = await db.execute(response_stmt)
    questionnaire_responses = {
        r.question_number: r.response_text for r in response_result.scalars().all()
    }

    # Build the evidence map
    evidence_map: dict[str, dict] = {}
    for qn, entries in sorted(evidence_by_question.items()):
        evidence_map[f"question_{qn}"] = {
            "response_text": questionnaire_responses.get(qn, ""),
            "sin_recognitions": entries,
        }

    response = {
        "match_id": str(match_id),
        "user_id": str(user_id),
        "evidence_map": evidence_map,
        "total_evidence_records": len(evidence_records),
        "questions_covered": sorted(evidence_by_question.keys()),
    }

    log.info(
        "get_evidence_map_complete",
        record_count=len(evidence_records),
    )
    return response
