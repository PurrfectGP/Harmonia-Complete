"""
Harmonia V3 — HLA Genetics API

Endpoints for uploading HLA genotype data and computing biological
compatibility scores between user pairs.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.match import HLAUploadResponse
from app.services.hla_service import HLAService

logger = structlog.get_logger("harmonia.api.hla")

router = APIRouter()

# ── Service singleton ─────────────────────────────────────────────────────────

_hla_service: HLAService | None = None


def _get_hla_service() -> HLAService:
    global _hla_service
    if _hla_service is None:
        _hla_service = HLAService()
    return _hla_service


# ──────────────────────────────────────────────────────────────────────────────
# POST /upload — Upload HLA genetic data
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=HLAUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload HLA genetic data",
)
async def upload_hla_data(
    user_id: uuid.UUID = Body(..., description="User UUID"),
    alleles: dict = Body(
        ...,
        description="Dict of locus -> allele list, e.g. {'A': ['A*01:01', 'A*02:01']}",
    ),
    source: str = Body("direct_upload", description="Data source provenance"),
    db: AsyncSession = Depends(get_db),
) -> HLAUploadResponse:
    """Validate, encrypt, and store HLA allele data for a user.

    Expects an ``alleles`` dict mapping locus names to lists of allele
    identifiers. The data is validated for format correctness, encrypted
    with Fernet, and persisted to the ``hla_data`` table.
    """
    log = logger.bind(user_id=str(user_id), source=source)
    log.info("upload_hla_start")

    # Verify user exists
    user_stmt = select(User).where(User.id == user_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    hla_svc = _get_hla_service()
    result = await hla_svc.upload_hla_data(
        user_id=str(user_id),
        alleles=alleles,
        source=source,
        db_session=db,
    )

    if not result.get("success", False):
        log.warning("upload_hla_validation_failed", errors=result.get("errors"))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "HLA data validation failed.",
                "errors": result.get("errors", []),
            },
        )

    log.info(
        "upload_hla_complete",
        allele_count=result.get("allele_count"),
        imputation_confidence=result.get("imputation_confidence"),
    )

    return HLAUploadResponse(
        status="uploaded",
        snp_count=result.get("allele_count", 0),
        imputation_confidence=result.get("imputation_confidence", 0.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /compatibility/{user_a_id}/{user_b_id} — HLA compatibility score
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/compatibility/{user_a_id}/{user_b_id}",
    summary="Get HLA compatibility score",
)
async def get_hla_compatibility(
    user_a_id: uuid.UUID,
    user_b_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Compute the biological compatibility score between two users.

    The response includes the S_bio score, heterozygosity index,
    olfactory attraction prediction, and display tier.  Per the spec,
    if S_bio is below 25 the chemistry signal is hidden from the
    user-facing display (``display.show`` will be ``false``).
    """
    log = logger.bind(
        user_a_id=str(user_a_id),
        user_b_id=str(user_b_id),
    )
    log.info("hla_compatibility_start")

    # Verify both users exist
    for uid, label in [
        (user_a_id, "User A"),
        (user_b_id, "User B"),
    ]:
        stmt = select(User).where(User.id == uid)
        result = await db.execute(stmt)
        if result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{label} ({uid}) not found.",
            )

    hla_svc = _get_hla_service()
    compatibility = await hla_svc.calculate_compatibility(
        user_a_id=str(user_a_id),
        user_b_id=str(user_b_id),
        db_session=db,
    )

    # Handle missing HLA data
    if compatibility.get("error") == "hla_data_missing":
        missing = compatibility.get("missing_users", [])
        log.warning("hla_data_missing", missing_users=missing)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "HLA data missing for one or both users.",
                "missing_users": missing,
            },
        )

    # Respect display logic: hide full details if S_bio < 25
    s_bio = compatibility.get("s_bio", 0.0)
    display = compatibility.get("display", {})

    if not display.get("show", True):
        log.info("hla_compatibility_hidden", s_bio=s_bio)
        return {
            "user_a_id": str(user_a_id),
            "user_b_id": str(user_b_id),
            "display": {"show": False, "tier": "hidden"},
            "message": "Chemistry signal is not strong enough to display.",
        }

    log.info("hla_compatibility_complete", s_bio=s_bio)
    return compatibility
