"""
Harmonia V3 — Questionnaire API

Endpoints for submitting questionnaire responses and retrieving the
canonical Felix question set.  Submissions trigger Gemini-powered
personality parsing and, when a full batch is submitted, the profile
aggregation pipeline.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.questionnaire import Question, QuestionnaireResponse
from app.models.user import User
from app.schemas.questionnaire import (
    BatchResponseSubmit,
    BatchSubmitResponse,
    QuestionSubmitResponse,
    SingleResponseSubmit,
)
from app.services.gemini_service import GeminiService
from app.services.profile_service import ProfileService

logger = structlog.get_logger("harmonia.api.questionnaire")

router = APIRouter()

# ── Service singletons (lazy, constructed on first use) ───────────────────────

_gemini_service: GeminiService | None = None
_profile_service: ProfileService | None = None


def _get_gemini_service() -> GeminiService:
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


def _get_profile_service() -> ProfileService:
    global _profile_service
    if _profile_service is None:
        _profile_service = ProfileService()
    return _profile_service


# ──────────────────────────────────────────────────────────────────────────────
# GET /questions — Return the six Felix questions
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/questions",
    summary="Get all questionnaire questions",
)
async def get_questions(
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return the list of all 6 Felix questions.

    Each entry contains ``question_number``, ``question_text``, and
    ``category``.
    """
    logger.info("get_questions")

    stmt = select(Question).order_by(Question.question_number)
    result = await db.execute(stmt)
    questions = result.scalars().all()

    return [
        {
            "question_number": q.question_number,
            "question_text": q.question_text,
            "category": q.category,
        }
        for q in questions
    ]


# ──────────────────────────────────────────────────────────────────────────────
# POST /submit — Submit a single question response
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/submit",
    response_model=QuestionSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a single questionnaire response",
)
async def submit_single_response(
    payload: SingleResponseSubmit,
    db: AsyncSession = Depends(get_db),
) -> QuestionSubmitResponse:
    """Submit a single question response.

    The response text is validated for word count (25-150 words), stored
    in the database, and a background Gemini parsing task is triggered to
    extract personality signals.
    """
    log = logger.bind(
        user_id=str(payload.user_id),
        question_number=payload.question_number,
    )
    log.info("submit_single_start")

    # Verify user exists
    user_stmt = select(User).where(User.id == payload.user_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {payload.user_id} not found.",
        )

    # Fetch the question text
    q_stmt = select(Question).where(
        Question.question_number == payload.question_number
    )
    q_result = await db.execute(q_stmt)
    question = q_result.scalar_one_or_none()

    question_text = question.question_text if question else f"Question {payload.question_number}"
    word_count = len(payload.response_text.split())

    # Upsert the response (replace if already exists for this user+question)
    existing_stmt = select(QuestionnaireResponse).where(
        QuestionnaireResponse.user_id == payload.user_id,
        QuestionnaireResponse.question_number == payload.question_number,
    )
    existing_result = await db.execute(existing_stmt)
    existing = existing_result.scalar_one_or_none()

    if existing is not None:
        existing.response_text = payload.response_text
        existing.question_text = question_text
        existing.word_count = word_count
        log.info("submit_single_updated_existing")
    else:
        new_response = QuestionnaireResponse(
            user_id=payload.user_id,
            question_number=payload.question_number,
            question_text=question_text,
            response_text=payload.response_text,
            word_count=word_count,
        )
        db.add(new_response)
        log.info("submit_single_created")

    await db.flush()

    # Trigger Gemini parsing (best-effort; errors are logged but do not
    # fail the submission)
    try:
        gemini = _get_gemini_service()
        parse_result = await gemini.parse_single_response(
            question=question_text,
            answer=payload.response_text,
            question_number=payload.question_number,
            user_id=str(payload.user_id),
        )

        # Store parsing evidence
        await gemini.store_parsing_evidence(
            db_session=db,
            user_id=str(payload.user_id),
            question_number=payload.question_number,
            sins=parse_result.get("sins", {}),
        )
        log.info("submit_single_parsing_complete")
    except Exception:
        log.exception("submit_single_parsing_failed")

    return QuestionSubmitResponse(
        status="submitted",
        question_number=payload.question_number,
        word_count=word_count,
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /submit-all — Submit all 6 responses in a batch
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/submit-all",
    response_model=BatchSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit all 6 questionnaire responses",
)
async def submit_all_responses(
    payload: BatchResponseSubmit,
    db: AsyncSession = Depends(get_db),
) -> BatchSubmitResponse:
    """Submit all 6 questionnaire responses in a single batch.

    Triggers the full Gemini parsing pipeline and the profile aggregation
    pipeline, returning the resulting profile quality tier and ID.
    """
    log = logger.bind(
        user_id=str(payload.user_id),
        response_count=len(payload.responses),
    )
    log.info("submit_all_start")

    # Verify user exists
    user_stmt = select(User).where(User.id == payload.user_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {payload.user_id} not found.",
        )

    # Fetch all questions for text lookup
    q_stmt = select(Question).order_by(Question.question_number)
    q_result = await db.execute(q_stmt)
    questions = {q.question_number: q for q in q_result.scalars().all()}

    # Persist each response
    gemini_input: list[dict] = []

    for resp in payload.responses:
        qn = resp.get("question_number") or resp.get("questionNumber")
        text = resp.get("response_text") or resp.get("responseText", "")
        question_obj = questions.get(qn)
        question_text = question_obj.question_text if question_obj else f"Question {qn}"
        word_count = len(text.split())

        # Upsert
        existing_stmt = select(QuestionnaireResponse).where(
            QuestionnaireResponse.user_id == payload.user_id,
            QuestionnaireResponse.question_number == qn,
        )
        existing_result = await db.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        if existing is not None:
            existing.response_text = text
            existing.question_text = question_text
            existing.word_count = word_count
        else:
            db.add(QuestionnaireResponse(
                user_id=payload.user_id,
                question_number=qn,
                question_text=question_text,
                response_text=text,
                word_count=word_count,
            ))

        gemini_input.append({
            "question_number": qn,
            "question_text": question_text,
            "response_text": text,
        })

    await db.flush()
    log.info("submit_all_responses_persisted")

    # Full Gemini parsing + profile creation
    profile_id = None
    quality_tier = None
    quality_score = None

    try:
        gemini = _get_gemini_service()
        parse_results = await gemini.parse_all_responses(
            responses=gemini_input,
            user_id=str(payload.user_id),
        )
        log.info("submit_all_parsing_complete")

        # Store all parsing evidence
        for qr in parse_results.get("per_question", []):
            await gemini.store_parsing_evidence(
                db_session=db,
                user_id=str(payload.user_id),
                question_number=qr["question_number"],
                sins=qr.get("sins", {}),
            )

        # Build personality profile
        profile_svc = _get_profile_service()
        profile_data = await profile_svc.build_profile(
            user_id=str(payload.user_id),
            parsed_responses=parse_results.get("per_question", []),
        )

        # Save profile to DB
        save_result = await profile_svc.save_profile(
            user_id=str(payload.user_id),
            profile_data=profile_data,
            source="real_user",
        )

        profile_id = uuid.UUID(save_result["profile_id"])
        quality_tier = save_result["quality_tier"]
        quality_score = profile_data.get("quality_score")

        log.info(
            "submit_all_profile_created",
            profile_id=str(profile_id),
            quality_tier=quality_tier,
        )
    except Exception:
        log.exception("submit_all_pipeline_failed")

    return BatchSubmitResponse(
        status="completed" if profile_id else "partial",
        profile_id=profile_id,
        quality_tier=quality_tier,
        quality_score=quality_score,
    )
