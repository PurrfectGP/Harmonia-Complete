"""
Harmonia V3 — Match and Swipe models.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("user_a_id", "user_b_id", name="uq_match_pair"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_a_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    user_b_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    s_vis_a_to_b: Mapped[float] = mapped_column(Float, nullable=False)
    s_vis_b_to_a: Mapped[float] = mapped_column(Float, nullable=False)
    s_psych: Mapped[float] = mapped_column(Float, nullable=False)
    s_bio: Mapped[float | None] = mapped_column(Float, nullable=True)
    wtm_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning_chain: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="Level 3 reasoning chain"
    )
    customer_summary: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="Level 1 customer-facing summary"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────
    user_a: Mapped["User"] = relationship(
        "User", foreign_keys=[user_a_id], lazy="selectin"
    )
    user_b: Mapped["User"] = relationship(
        "User", foreign_keys=[user_b_id], lazy="selectin"
    )

    def __repr__(self) -> str:
        return (
            f"<Match {self.user_a_id} <-> {self.user_b_id} "
            f"wtm={self.wtm_score:.3f}>"
        )


class Swipe(Base):
    __tablename__ = "swipes"
    __table_args__ = (
        UniqueConstraint("swiper_id", "target_id", name="uq_swipe_pair"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    swiper_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    direction: Mapped[str] = mapped_column(
        String, nullable=False, comment="left / right / superlike"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────
    swiper: Mapped["User"] = relationship(
        "User", foreign_keys=[swiper_id], lazy="selectin"
    )
    target: Mapped["User"] = relationship(
        "User", foreign_keys=[target_id], lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Swipe {self.swiper_id} -> {self.target_id} dir={self.direction!r}>"
