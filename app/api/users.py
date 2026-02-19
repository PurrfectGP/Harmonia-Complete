"""
Harmonia V3 — Users API

Endpoints for user CRUD, photo uploads, and discovery feed.
"""

from __future__ import annotations

import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.utils.storage import upload_file

logger = structlog.get_logger("harmonia.api.users")

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# POST / — Create a new user
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
)
async def create_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Register a new user account.

    Validates that the email is not already in use, creates the user record,
    and returns the full user response.
    """
    log = logger.bind(email=payload.email)
    log.info("create_user_start")

    # Check for duplicate email
    stmt = select(User).where(User.email == payload.email)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing is not None:
        log.warning("create_user_duplicate_email")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    new_user = User(
        email=payload.email,
        display_name=payload.display_name,
        age=payload.age,
        gender=payload.gender,
        location=payload.location,
        photos=[],
    )
    db.add(new_user)
    await db.flush()

    log.info("create_user_complete", user_id=str(new_user.id))
    return new_user


# ──────────────────────────────────────────────────────────────────────────────
# GET / — List users with pagination
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=list[UserResponse],
    summary="List users with pagination",
)
async def list_users(
    limit: int = Query(20, ge=1, le=100, description="Max users to return"),
    offset: int = Query(0, ge=0, description="Number of users to skip"),
    db: AsyncSession = Depends(get_db),
) -> list[User]:
    """Return a paginated list of active users."""
    logger.info("list_users", limit=limit, offset=offset)

    stmt = (
        select(User)
        .where(User.is_active.is_(True))
        .order_by(User.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    users = result.scalars().all()

    return list(users)


# ──────────────────────────────────────────────────────────────────────────────
# GET /{user_id} — Get user by ID
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
)
async def get_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Retrieve a single user by their UUID."""
    log = logger.bind(user_id=str(user_id))
    log.info("get_user")

    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        log.warning("get_user_not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    return user


# ──────────────────────────────────────────────────────────────────────────────
# PUT /{user_id} — Update user
# ──────────────────────────────────────────────────────────────────────────────

@router.put(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user details",
)
async def update_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Update mutable fields on a user record.

    Only fields present in the request body (non-None) are applied.
    """
    log = logger.bind(user_id=str(user_id))
    log.info("update_user_start")

    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        log.warning("update_user_not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)

    await db.flush()
    log.info("update_user_complete", updated_fields=list(update_data.keys()))
    return user


# ──────────────────────────────────────────────────────────────────────────────
# POST /{user_id}/photos — Upload photos
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/{user_id}/photos",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload user photos",
)
async def upload_photos(
    user_id: uuid.UUID,
    files: list[UploadFile] = File(..., description="One or more image files"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Upload one or more photos for a user.

    Files are stored in GCS under ``users/{user_id}/photos/`` and the
    resulting URIs are appended to the user's ``photos`` array.
    """
    log = logger.bind(user_id=str(user_id), file_count=len(files))
    log.info("upload_photos_start")

    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        log.warning("upload_photos_user_not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    current_photos: list[str] = user.photos or []

    for upload in files:
        file_bytes = await upload.read()
        content_type = upload.content_type or "image/jpeg"
        ext = upload.filename.rsplit(".", 1)[-1] if upload.filename and "." in upload.filename else "jpg"
        gcs_path = f"users/{user_id}/photos/{uuid.uuid4().hex}.{ext}"

        gcs_uri = upload_file(gcs_path, file_bytes, content_type=content_type)
        current_photos.append(gcs_uri)

        log.info("photo_uploaded", gcs_path=gcs_path)

    user.photos = current_photos
    await db.flush()

    log.info("upload_photos_complete", total_photos=len(current_photos))
    return user


# ──────────────────────────────────────────────────────────────────────────────
# GET /{user_id}/discover — Discovery feed candidates
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{user_id}/discover",
    response_model=list[UserResponse],
    summary="Get discovery feed candidates",
)
async def discover_feed(
    user_id: uuid.UUID,
    limit: int = Query(20, ge=1, le=100, description="Max candidates to return"),
    offset: int = Query(0, ge=0, description="Number of candidates to skip"),
    db: AsyncSession = Depends(get_db),
) -> list[User]:
    """Return a list of candidate users for the discovery feed.

    Excludes the requesting user and returns only active users. In a
    production system this would also filter by already-swiped targets,
    location preferences, and other criteria.
    """
    log = logger.bind(user_id=str(user_id))
    log.info("discover_feed", limit=limit, offset=offset)

    # Verify the requesting user exists
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    requesting_user = result.scalar_one_or_none()

    if requesting_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    # Fetch candidates — exclude self, only active users
    stmt = (
        select(User)
        .where(User.is_active.is_(True))
        .where(User.id != user_id)
        .order_by(User.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    candidates = result.scalars().all()

    log.info("discover_feed_complete", candidate_count=len(candidates))
    return list(candidates)
