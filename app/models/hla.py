"""
Harmonia V3 — HLA (Human Leukocyte Antigen) data model.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class HLAData(Base):
    __tablename__ = "hla_data"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    encrypted_data: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False, comment="Fernet-encrypted HLA payload"
    )
    source: Mapped[str] = mapped_column(
        String, nullable=False, comment="e.g. 23andMe_v5"
    )
    imputation_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    ancestry_model: Mapped[str | None] = mapped_column(String, nullable=True)
    snp_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="hla_data")

    def __repr__(self) -> str:
        return f"<HLAData user={self.user_id} source={self.source!r} snps={self.snp_count}>"
