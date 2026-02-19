"""Initial schema — all 11 Harmonia V3 tables.

Revision ID: 001_initial
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. users ────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String, unique=True, index=True, nullable=False),
        sa.Column("display_name", sa.String, nullable=False),
        sa.Column("age", sa.Integer, nullable=False),
        sa.Column("gender", sa.String, nullable=False),
        sa.Column("location", sa.String, nullable=True),
        sa.Column(
            "photos",
            postgresql.JSONB,
            nullable=True,
            comment="Array of photo URLs",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean,
            server_default="true",
            nullable=False,
        ),
    )

    # ── 2. questions (reference table) ──────────────────────────────
    op.create_table(
        "questions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("question_number", sa.Integer, unique=True, nullable=False),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column("category", sa.String, nullable=False),
    )

    # ── 3. questionnaire_responses ──────────────────────────────────
    op.create_table(
        "questionnaire_responses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "question_number",
            sa.Integer,
            nullable=False,
            comment="1-6",
        ),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column("response_text", sa.Text, nullable=False),
        sa.Column("word_count", sa.Integer, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("user_id", "question_number", name="uq_user_question"),
    )

    # ── 4. personality_profiles ─────────────────────────────────────
    op.create_table(
        "personality_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column(
            "sins",
            postgresql.JSONB,
            nullable=False,
            comment="7 aggregated sin scores",
        ),
        sa.Column("quality_score", sa.Float, nullable=False),
        sa.Column(
            "quality_tier",
            sa.String,
            nullable=False,
            comment="high / moderate / low / rejected",
        ),
        sa.Column("response_styles", postgresql.JSONB, nullable=True),
        sa.Column(
            "flags",
            postgresql.JSONB,
            nullable=True,
            comment="Array of flag strings",
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=True,
            comment="Extra metadata (column name: metadata)",
        ),
        sa.Column(
            "source",
            sa.String,
            nullable=False,
            comment="real_user / claude_agent",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── 5. visual_preferences ───────────────────────────────────────
    op.create_table(
        "visual_preferences",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column("support_set_stats", postgresql.JSONB, nullable=True),
        sa.Column("mandatory_traits", postgresql.JSONB, nullable=True),
        sa.Column("preferred_traits", postgresql.JSONB, nullable=True),
        sa.Column("aversion_traits", postgresql.JSONB, nullable=True),
        sa.Column(
            "adapted_weights_key",
            sa.String,
            nullable=True,
            comment="Redis key for adapted weights",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── 6. visual_ratings ───────────────────────────────────────────
    op.create_table(
        "visual_ratings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("image_id", sa.String, nullable=False),
        sa.Column("image_path", sa.String, nullable=False),
        sa.Column("rating", sa.Integer, nullable=False, comment="1-5"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_visual_ratings_user_rating",
        "visual_ratings",
        ["user_id", "rating"],
    )

    # ── 7. hla_data ─────────────────────────────────────────────────
    op.create_table(
        "hla_data",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column(
            "encrypted_data",
            sa.LargeBinary,
            nullable=False,
            comment="Fernet-encrypted HLA payload",
        ),
        sa.Column(
            "source",
            sa.String,
            nullable=False,
            comment="e.g. 23andMe_v5",
        ),
        sa.Column("imputation_confidence", sa.Float, nullable=False),
        sa.Column("ancestry_model", sa.String, nullable=True),
        sa.Column("snp_count", sa.Integer, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ── 8. matches ──────────────────────────────────────────────────
    op.create_table(
        "matches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_a_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_b_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("s_vis_a_to_b", sa.Float, nullable=False),
        sa.Column("s_vis_b_to_a", sa.Float, nullable=False),
        sa.Column("s_psych", sa.Float, nullable=False),
        sa.Column("s_bio", sa.Float, nullable=True),
        sa.Column("wtm_score", sa.Float, nullable=False),
        sa.Column(
            "reasoning_chain",
            postgresql.JSONB,
            nullable=True,
            comment="Level 3 reasoning chain",
        ),
        sa.Column(
            "customer_summary",
            postgresql.JSONB,
            nullable=True,
            comment="Level 1 customer-facing summary",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("user_a_id", "user_b_id", name="uq_match_pair"),
    )

    # ── 9. swipes ───────────────────────────────────────────────────
    op.create_table(
        "swipes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "swiper_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "direction",
            sa.String,
            nullable=False,
            comment="left / right / superlike",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("swiper_id", "target_id", name="uq_swipe_pair"),
    )

    # ── 10. parsing_evidence ────────────────────────────────────────
    op.create_table(
        "parsing_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "question_number",
            sa.Integer,
            nullable=False,
            comment="1-6",
        ),
        sa.Column("sin", sa.String, nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("evidence_snippet", sa.Text, nullable=False),
        sa.Column("snippet_start_index", sa.Integer, nullable=False),
        sa.Column("snippet_end_index", sa.Integer, nullable=False),
        sa.Column("interpretation", sa.Text, nullable=True),
        sa.Column("observer_persona", sa.String, nullable=True),
        sa.Column("gemini_model_used", sa.String, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_parsing_evidence_user_question_sin",
        "parsing_evidence",
        ["user_id", "question_number", "sin"],
    )

    # ── 11. calibration_examples ────────────────────────────────────
    op.create_table(
        "calibration_examples",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "question_number",
            sa.Integer,
            nullable=False,
            comment="1-6",
        ),
        sa.Column("response_text", sa.Text, nullable=False),
        sa.Column("sin", sa.String, nullable=False),
        sa.Column("gemini_raw_score", sa.Float, nullable=False),
        sa.Column("gemini_raw_confidence", sa.Float, nullable=False),
        sa.Column("gemini_raw_evidence", sa.Text, nullable=False),
        sa.Column("validated_score", sa.Float, nullable=True),
        sa.Column("validated_by", sa.String, nullable=True),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "review_status",
            sa.String,
            nullable=False,
            server_default="pending",
            comment="pending / approved / corrected / rejected",
        ),
        sa.Column("review_notes", sa.Text, nullable=True),
        sa.Column(
            "source_profile_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("personality_profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_calibration_question_sin_status",
        "calibration_examples",
        ["question_number", "sin", "review_status"],
    )


def downgrade() -> None:
    # Drop in reverse order (children / dependents first).
    op.drop_index(
        "ix_calibration_question_sin_status", table_name="calibration_examples"
    )
    op.drop_table("calibration_examples")

    op.drop_index(
        "ix_parsing_evidence_user_question_sin", table_name="parsing_evidence"
    )
    op.drop_table("parsing_evidence")

    op.drop_table("swipes")
    op.drop_table("matches")
    op.drop_table("hla_data")

    op.drop_index("ix_visual_ratings_user_rating", table_name="visual_ratings")
    op.drop_table("visual_ratings")

    op.drop_table("visual_preferences")
    op.drop_table("personality_profiles")
    op.drop_table("questionnaire_responses")
    op.drop_table("questions")
    op.drop_table("users")
