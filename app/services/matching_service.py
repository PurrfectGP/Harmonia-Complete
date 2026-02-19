"""
Harmonia V3 — Three-Stage Matching Engine & WtM Calculation

Orchestrates the cascaded matching pipeline:
  Stage 1: Visual Gate — Mutual swipe check + asymmetric S_vis scoring
  Stage 2: Personality Reveal — Perceived similarity (soft gate)
  Stage 3: Genetics Info — HLA compatibility (informational only)

Then calculates the reciprocal Willingness-to-Meet (WtM) score:
  combined_A_to_B = (w_vis × s_vis_a_to_b) + (w_psych × s_psych) + (w_bio × s_bio)
  combined_B_to_A = (w_vis × s_vis_b_to_a) + (w_psych × s_psych) + (w_bio × s_bio)
  reciprocal_wtm  = √(combined_A_to_B × combined_B_to_A)

Default weights: visual=0.4, personality=0.3, bio=0.3.
When S_bio is unavailable, weights redistribute proportionally:
  visual → 0.571, personality → 0.429, bio → 0.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.match import Match, Swipe
from app.models.profile import PersonalityProfile

logger = structlog.get_logger("harmonia.matching_service")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_FRICTION_THRESHOLD = 0.3  # Sin delta threshold for friction flags

# Sin labels used for friction detection
_SIN_NAMES: list[str] = [
    "greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth",
]

# Human-readable friction labels for specific sin conflicts
_FRICTION_LABELS: dict[str, str] = {
    "wrath": "bluntness_delta",
    "sloth": "sloth_delta",
    "pride": "pride_delta",
    "lust": "impulsivity_delta",
    "greed": "generosity_delta",
    "gluttony": "indulgence_delta",
    "envy": "ambition_delta",
}


class MatchingService:
    """Three-stage cascaded matching engine with WtM scoring.

    Dependencies are injected at construction so that the service can be
    tested with mocks and swapped in FastAPI's dependency-injection graph.
    """

    def __init__(
        self,
        visual_service: Any | None = None,
        similarity_service: Any | None = None,
        hla_service: Any | None = None,
    ) -> None:
        """Initialise the matching service with injected dependencies.

        Parameters
        ----------
        visual_service:
            Instance of VisualService for S_vis scoring.
        similarity_service:
            Instance of SimilarityService for S_psych calculation.
        hla_service:
            Instance of HLAService for S_bio calculation.
        """
        self.visual_service = visual_service
        self.similarity_service = similarity_service
        self.hla_service = hla_service

        # Load default weights from config
        settings = get_settings()
        self.w_vis: float = settings.VISUAL_WEIGHT       # 0.4
        self.w_psych: float = settings.PERSONALITY_WEIGHT  # 0.3
        self.w_bio: float = settings.HLA_WEIGHT           # 0.3

        logger.info(
            "matching_service_initialised",
            w_vis=self.w_vis,
            w_psych=self.w_psych,
            w_bio=self.w_bio,
            has_visual=visual_service is not None,
            has_similarity=similarity_service is not None,
            has_hla=hla_service is not None,
        )

    # ── Public API ────────────────────────────────────────────────────────

    async def calculate_match(
        self,
        user_a_id: str,
        user_b_id: str,
        db_session: AsyncSession,
    ) -> dict:
        """Execute the full three-stage matching pipeline.

        Pipeline:
          Stage 1 — Visual Gate: check mutual swipe, compute S_vis.
          Stage 2 — Personality Reveal: compute S_psych via SimilarityService.
          Stage 3 — Genetics Info: compute S_bio via HLAService (informational).
          Final  — Calculate WtM, apply friction, store match.

        Parameters
        ----------
        user_a_id:
            UUID string of the first user.
        user_b_id:
            UUID string of the second user.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            Match result including all scores, WtM, friction flags,
            reasoning_chain, and customer_summary.  If the visual gate
            fails, returns early with ``status: "no_mutual_swipe"``.
        """
        log = logger.bind(user_a=user_a_id, user_b=user_b_id)
        log.info("match_calculation_start")

        # ── Stage 1: Visual Gate ──────────────────────────────────────
        log.info("stage_1_visual_gate_start")

        is_mutual = await self._check_mutual_swipe(user_a_id, user_b_id, db_session)
        if not is_mutual:
            log.info("stage_1_visual_gate_failed", reason="no_mutual_swipe")
            return {
                "status": "no_mutual_swipe",
                "user_a_id": user_a_id,
                "user_b_id": user_b_id,
                "message": "Both users must swipe right or superlike to proceed.",
            }

        # Calculate asymmetric visual scores
        s_vis_a_to_b = 50.0  # default if visual service unavailable
        s_vis_b_to_a = 50.0

        if self.visual_service is not None:
            try:
                # A's adapted model scores B's photos
                score_a_to_b = await self.visual_service.score_target(
                    user_id=user_a_id,
                    target_image_path=await self._get_user_primary_photo(
                        user_b_id, db_session
                    ),
                    db_session=db_session,
                )
                s_vis_a_to_b = score_a_to_b.get("s_vis", 50.0)
            except Exception:
                log.warning("visual_scoring_failed", direction="a_to_b")

            try:
                # B's adapted model scores A's photos
                score_b_to_a = await self.visual_service.score_target(
                    user_id=user_b_id,
                    target_image_path=await self._get_user_primary_photo(
                        user_a_id, db_session
                    ),
                    db_session=db_session,
                )
                s_vis_b_to_a = score_b_to_a.get("s_vis", 50.0)
            except Exception:
                log.warning("visual_scoring_failed", direction="b_to_a")

        log.info(
            "stage_1_visual_gate_complete",
            s_vis_a_to_b=round(s_vis_a_to_b, 4),
            s_vis_b_to_a=round(s_vis_b_to_a, 4),
        )

        # ── Stage 2: Personality Reveal ───────────────────────────────
        log.info("stage_2_personality_reveal_start")

        s_psych = 0.0
        similarity_result: dict = {}
        profile_a_data: dict = {}
        profile_b_data: dict = {}
        friction_flags: dict = {}

        # Fetch personality profiles
        stmt_a = select(PersonalityProfile).where(
            PersonalityProfile.user_id == user_a_id
        )
        stmt_b = select(PersonalityProfile).where(
            PersonalityProfile.user_id == user_b_id
        )
        result_a = await db_session.execute(stmt_a)
        result_b = await db_session.execute(stmt_b)
        profile_a = result_a.scalar_one_or_none()
        profile_b = result_b.scalar_one_or_none()

        if profile_a is not None and profile_b is not None:
            profile_a_data = {
                "sins": profile_a.sins,
                "quality_tier": profile_a.quality_tier,
                "quality_score": profile_a.quality_score,
            }
            profile_b_data = {
                "sins": profile_b.sins,
                "quality_tier": profile_b.quality_tier,
                "quality_score": profile_b.quality_score,
            }

            if self.similarity_service is not None:
                similarity_result = self.similarity_service.calculate_similarity(
                    profile_a_data, profile_b_data
                )
                s_psych = similarity_result.get("adjusted_score", 0.0)

                # Scale s_psych from [0, 1] to [0, 100] for WtM formula
                s_psych = s_psych * 100.0

            # Calculate friction flags
            friction_flags = self._calculate_friction_flags(
                profile_a_data, profile_b_data
            )

            # Apply friction penalty to s_psych
            p_friction = friction_flags.get("p_friction", 1.0)
            s_psych = s_psych * p_friction

            log.info(
                "stage_2_personality_reveal_complete",
                s_psych=round(s_psych, 4),
                p_friction=round(p_friction, 4),
                friction_count=friction_flags.get("flag_count", 0),
            )
        else:
            missing = []
            if profile_a is None:
                missing.append(user_a_id)
            if profile_b is None:
                missing.append(user_b_id)
            log.warning(
                "stage_2_personality_profiles_missing",
                missing_users=missing,
            )

        # ── Stage 3: Genetics Info (informational) ────────────────────
        log.info("stage_3_genetics_info_start")

        s_bio: float | None = None
        hla_result: dict = {}

        if self.hla_service is not None:
            try:
                hla_result = await self.hla_service.calculate_compatibility(
                    user_a_id, user_b_id, db_session
                )
                if hla_result.get("error") is None:
                    s_bio = hla_result.get("s_bio")
                    log.info("stage_3_genetics_info_complete", s_bio=s_bio)
                else:
                    log.info(
                        "stage_3_genetics_info_unavailable",
                        reason=hla_result.get("error"),
                    )
            except Exception:
                log.warning("stage_3_genetics_info_failed")
        else:
            log.info("stage_3_genetics_info_skipped", reason="no_hla_service")

        # ── Final: Calculate WtM ──────────────────────────────────────
        wtm_result = self._calculate_wtm(
            s_vis_a_to_b=s_vis_a_to_b,
            s_vis_b_to_a=s_vis_b_to_a,
            s_psych=s_psych,
            s_bio=s_bio,
        )

        log.info(
            "wtm_calculation_complete",
            reciprocal_wtm=round(wtm_result["reciprocal_wtm"], 4),
            combined_a_to_b=round(wtm_result["combined_a_to_b"], 4),
            combined_b_to_a=round(wtm_result["combined_b_to_a"], 4),
        )

        # ── Build reasoning chain (Level 3) ───────────────────────────
        reasoning_chain = {
            "phase_1_visual": {
                "s_vis_a_to_b": round(s_vis_a_to_b, 4),
                "s_vis_b_to_a": round(s_vis_b_to_a, 4),
                "mutual_swipe": True,
            },
            "phase_2_psychometric": {
                "s_psych_raw": round(
                    similarity_result.get("adjusted_score", 0.0) * 100.0, 4
                ) if similarity_result else 0.0,
                "s_psych_after_friction": round(s_psych, 4),
                "friction_flags": friction_flags,
                "similarity_breakdown": similarity_result.get("breakdown", []),
                "similarity_tier": similarity_result.get("tier", "unknown"),
            },
            "phase_3_biological": {
                "s_bio": round(s_bio, 4) if s_bio is not None else None,
                "hla_available": s_bio is not None,
                "hla_details": {
                    k: v for k, v in hla_result.items()
                    if k not in ("user_a_id", "user_b_id")
                } if hla_result else {},
            },
            "final_calculation": {
                "weights_used": wtm_result["weights_used"],
                "combined_a_to_b": round(wtm_result["combined_a_to_b"], 4),
                "combined_b_to_a": round(wtm_result["combined_b_to_a"], 4),
                "reciprocal_wtm": round(wtm_result["reciprocal_wtm"], 4),
                "bio_available": s_bio is not None,
            },
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

        # ── Build customer summary (Level 1) ──────────────────────────
        # Generate match explanation if similarity service is available
        match_explanation: dict = {}
        if self.similarity_service is not None and similarity_result:
            match_explanation = self.similarity_service.generate_match_explanation(
                breakdown=similarity_result.get("breakdown", []),
                tier=similarity_result.get("tier", "low_fit"),
                display_mode=similarity_result.get("display_mode", "minimal"),
            )

        # HLA display for customer card
        hla_display: dict = {"show": False}
        if self.similarity_service is not None and s_bio is not None:
            hla_display = self.similarity_service.get_hla_display(s_bio)

        display_score = round(wtm_result["reciprocal_wtm"], 1)

        customer_summary = {
            "display_score": display_score,
            "compatibility_breakdown": {
                "physical": {
                    "score": round((s_vis_a_to_b + s_vis_b_to_a) / 2.0, 1),
                    "label": "Physical Attraction",
                },
                "personality": {
                    "score": round(s_psych, 1),
                    "label": "Personality Match",
                },
                "chemistry": {
                    "score": round(s_bio, 1) if s_bio is not None else None,
                    "label": "Chemistry Signal",
                    "available": s_bio is not None,
                },
            },
            "shared_traits": match_explanation.get("shared_traits", []),
            "tier": match_explanation.get("tier", similarity_result.get("tier", "unknown")),
            "display_mode": match_explanation.get(
                "display_mode",
                similarity_result.get("display_mode", "minimal"),
            ),
            "hla_display": hla_display,
        }

        # ── Store match ───────────────────────────────────────────────
        match_data = {
            "user_a_id": user_a_id,
            "user_b_id": user_b_id,
            "s_vis_a_to_b": round(s_vis_a_to_b, 4),
            "s_vis_b_to_a": round(s_vis_b_to_a, 4),
            "s_psych": round(s_psych, 4),
            "s_bio": round(s_bio, 4) if s_bio is not None else None,
            "wtm_score": round(wtm_result["reciprocal_wtm"], 4),
            "reasoning_chain": reasoning_chain,
            "customer_summary": customer_summary,
        }

        match_id = await self._store_match(match_data, db_session)

        log.info(
            "match_calculation_complete",
            match_id=match_id,
            wtm_score=round(wtm_result["reciprocal_wtm"], 4),
        )

        return {
            "status": "match_created",
            "match_id": match_id,
            **match_data,
        }

    async def get_match(
        self,
        match_id: str,
        db_session: AsyncSession,
    ) -> dict:
        """Retrieve a match by its ID.

        Parameters
        ----------
        match_id:
            UUID string of the match record.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            Match data including all scores, reasoning chain,
            and customer summary.  Returns error dict if not found.
        """
        log = logger.bind(match_id=match_id)
        log.info("get_match_start")

        stmt = select(Match).where(Match.id == match_id)
        result = await db_session.execute(stmt)
        match = result.scalar_one_or_none()

        if match is None:
            log.warning("match_not_found")
            return {"error": "match_not_found", "match_id": match_id}

        log.info("match_retrieved", wtm_score=match.wtm_score)

        return {
            "match_id": str(match.id),
            "user_a_id": str(match.user_a_id),
            "user_b_id": str(match.user_b_id),
            "s_vis_a_to_b": match.s_vis_a_to_b,
            "s_vis_b_to_a": match.s_vis_b_to_a,
            "s_psych": match.s_psych,
            "s_bio": match.s_bio,
            "wtm_score": match.wtm_score,
            "reasoning_chain": match.reasoning_chain,
            "customer_summary": match.customer_summary,
            "created_at": match.created_at.isoformat() if match.created_at else None,
        }

    async def get_user_matches(
        self,
        user_id: str,
        db_session: AsyncSession,
    ) -> list[dict]:
        """List all matches for a given user.

        Parameters
        ----------
        user_id:
            UUID string of the user.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        list[dict]
            List of match records where the user appears as either
            user_a or user_b, ordered by creation date descending.
        """
        log = logger.bind(user_id=user_id)
        log.info("get_user_matches_start")

        stmt = (
            select(Match)
            .where(
                or_(
                    Match.user_a_id == user_id,
                    Match.user_b_id == user_id,
                )
            )
            .order_by(Match.created_at.desc())
        )
        result = await db_session.execute(stmt)
        matches = result.scalars().all()

        log.info("user_matches_retrieved", count=len(matches))

        return [
            {
                "match_id": str(m.id),
                "user_a_id": str(m.user_a_id),
                "user_b_id": str(m.user_b_id),
                "other_user_id": str(m.user_b_id)
                if str(m.user_a_id) == str(user_id)
                else str(m.user_a_id),
                "s_vis_a_to_b": m.s_vis_a_to_b,
                "s_vis_b_to_a": m.s_vis_b_to_a,
                "s_psych": m.s_psych,
                "s_bio": m.s_bio,
                "wtm_score": m.wtm_score,
                "customer_summary": m.customer_summary,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in matches
        ]

    # ── WtM Calculation ──────────────────────────────────────────────────

    def _calculate_wtm(
        self,
        s_vis_a_to_b: float,
        s_vis_b_to_a: float,
        s_psych: float,
        s_bio: float | None,
    ) -> dict:
        """Calculate the reciprocal Willingness-to-Meet score.

        Formula (with bio data):
          combined_A_to_B = (0.4 x s_vis_a_to_b) + (0.3 x s_psych) + (0.3 x s_bio)
          combined_B_to_A = (0.4 x s_vis_b_to_a) + (0.3 x s_psych) + (0.3 x s_bio)
          reciprocal_wtm  = sqrt(combined_A_to_B x combined_B_to_A)

        When s_bio is unavailable, weights are redistributed proportionally:
          w_vis  = 0.4 / (0.4 + 0.3) = 0.571
          w_psych = 0.3 / (0.4 + 0.3) = 0.429
          w_bio  = 0.0

        Parameters
        ----------
        s_vis_a_to_b:
            Visual attractiveness score from A's perspective of B (0-100).
        s_vis_b_to_a:
            Visual attractiveness score from B's perspective of A (0-100).
        s_psych:
            Personality similarity score (0-100), after friction penalty.
        s_bio:
            HLA biological compatibility score (0-100), or None if unavailable.

        Returns
        -------
        dict
            Contains combined_a_to_b, combined_b_to_a, reciprocal_wtm,
            and the weights_used for transparency.
        """
        if s_bio is not None:
            # Full three-signal calculation
            w_vis = self.w_vis
            w_psych = self.w_psych
            w_bio = self.w_bio

            combined_a_to_b = (
                (w_vis * s_vis_a_to_b)
                + (w_psych * s_psych)
                + (w_bio * s_bio)
            )
            combined_b_to_a = (
                (w_vis * s_vis_b_to_a)
                + (w_psych * s_psych)
                + (w_bio * s_bio)
            )
        else:
            # Redistribute weights proportionally (exclude bio)
            total_non_bio = self.w_vis + self.w_psych
            w_vis = self.w_vis / total_non_bio if total_non_bio > 0 else 0.5
            w_psych = self.w_psych / total_non_bio if total_non_bio > 0 else 0.5
            w_bio = 0.0

            combined_a_to_b = (w_vis * s_vis_a_to_b) + (w_psych * s_psych)
            combined_b_to_a = (w_vis * s_vis_b_to_a) + (w_psych * s_psych)

        # Geometric mean for reciprocal score
        product = combined_a_to_b * combined_b_to_a
        if product < 0:
            # Guard against negative products (shouldn't happen with valid scores)
            reciprocal_wtm = 0.0
        else:
            reciprocal_wtm = math.sqrt(product)

        # Clamp to [0, 100]
        reciprocal_wtm = max(0.0, min(100.0, reciprocal_wtm))

        logger.debug(
            "wtm_calculated",
            combined_a_to_b=round(combined_a_to_b, 4),
            combined_b_to_a=round(combined_b_to_a, 4),
            reciprocal_wtm=round(reciprocal_wtm, 4),
            w_vis=round(w_vis, 4),
            w_psych=round(w_psych, 4),
            w_bio=round(w_bio, 4),
        )

        return {
            "combined_a_to_b": combined_a_to_b,
            "combined_b_to_a": combined_b_to_a,
            "reciprocal_wtm": reciprocal_wtm,
            "weights_used": {
                "w_vis": round(w_vis, 4),
                "w_psych": round(w_psych, 4),
                "w_bio": round(w_bio, 4),
            },
        }

    # ── Friction Flags ───────────────────────────────────────────────────

    def _calculate_friction_flags(
        self,
        profile_a: dict,
        profile_b: dict,
    ) -> dict:
        """Detect sin deltas exceeding the friction threshold and calculate
        the friction penalty applied to S_psych.

        When two matched users have a large delta (> 0.3) on any sin axis,
        that trait is flagged as a potential friction point.  The penalty
        P_friction is calculated as:
          P_friction = max(0.5, 1.0 - (0.1 x n_flags))

        Where n_flags is the number of traits exceeding the threshold.
        This penalty multiplies S_psych before WtM calculation.

        Parameters
        ----------
        profile_a:
            Dict with ``sins`` key mapping sin names to score dicts.
        profile_b:
            Dict with ``sins`` key mapping sin names to score dicts.

        Returns
        -------
        dict
            Contains ``flags`` (list of flagged traits), ``deltas`` (dict
            of per-sin deltas), ``flag_count`` (int), and ``p_friction``
            (the multiplicative penalty).
        """
        sins_a = profile_a.get("sins", {})
        sins_b = profile_b.get("sins", {})

        flags: list[dict] = []
        deltas: dict[str, float] = {}

        for sin in _SIN_NAMES:
            entry_a = sins_a.get(sin, {})
            entry_b = sins_b.get(sin, {})
            score_a = float(entry_a.get("score", 0.0))
            score_b = float(entry_b.get("score", 0.0))

            # Normalise scores to [0, 1] range for delta calculation
            # Sin scores are on a [-5, +5] scale, normalise to [0, 1]
            norm_a = (score_a + 5.0) / 10.0
            norm_b = (score_b + 5.0) / 10.0
            delta = abs(norm_a - norm_b)

            deltas[sin] = round(delta, 4)

            if delta > _FRICTION_THRESHOLD:
                flag_label = _FRICTION_LABELS.get(sin, f"{sin}_delta")
                flags.append({
                    "sin": sin,
                    "flag": flag_label,
                    "delta": round(delta, 4),
                    "score_a": score_a,
                    "score_b": score_b,
                    "severity": "high" if delta > 0.5 else "moderate",
                })

        # Calculate P_friction penalty
        n_flags = len(flags)
        p_friction = max(0.5, 1.0 - (0.1 * n_flags))

        logger.debug(
            "friction_flags_calculated",
            flag_count=n_flags,
            p_friction=round(p_friction, 4),
            flagged_sins=[f["sin"] for f in flags],
        )

        return {
            "flags": flags,
            "deltas": deltas,
            "flag_count": n_flags,
            "p_friction": round(p_friction, 4),
        }

    # ── Stage helpers ────────────────────────────────────────────────────

    async def _check_mutual_swipe(
        self,
        user_a_id: str,
        user_b_id: str,
        db_session: AsyncSession,
    ) -> bool:
        """Check whether both users have swiped right or superliked each other.

        Queries the swipes table for a right or superlike swipe from A to B
        AND from B to A.  Both must exist for a mutual match.

        Parameters
        ----------
        user_a_id:
            UUID string of user A.
        user_b_id:
            UUID string of user B.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        bool
            True if both users have swiped right or superliked each other.
        """
        # Check A -> B swipe
        stmt_a_to_b = select(Swipe).where(
            Swipe.swiper_id == user_a_id,
            Swipe.target_id == user_b_id,
            Swipe.direction.in_(["right", "superlike"]),
        )
        result_a_to_b = await db_session.execute(stmt_a_to_b)
        swipe_a_to_b = result_a_to_b.scalar_one_or_none()

        if swipe_a_to_b is None:
            logger.debug(
                "mutual_swipe_check_failed",
                reason="a_to_b_missing",
                user_a=user_a_id,
                user_b=user_b_id,
            )
            return False

        # Check B -> A swipe
        stmt_b_to_a = select(Swipe).where(
            Swipe.swiper_id == user_b_id,
            Swipe.target_id == user_a_id,
            Swipe.direction.in_(["right", "superlike"]),
        )
        result_b_to_a = await db_session.execute(stmt_b_to_a)
        swipe_b_to_a = result_b_to_a.scalar_one_or_none()

        if swipe_b_to_a is None:
            logger.debug(
                "mutual_swipe_check_failed",
                reason="b_to_a_missing",
                user_a=user_a_id,
                user_b=user_b_id,
            )
            return False

        logger.debug(
            "mutual_swipe_confirmed",
            user_a=user_a_id,
            user_b=user_b_id,
            a_direction=swipe_a_to_b.direction,
            b_direction=swipe_b_to_a.direction,
        )
        return True

    async def _store_match(
        self,
        match_data: dict,
        db_session: AsyncSession,
    ) -> str:
        """Persist a match record to the matches table.

        Parameters
        ----------
        match_data:
            Dict containing all match fields: user_a_id, user_b_id,
            s_vis_a_to_b, s_vis_b_to_a, s_psych, s_bio, wtm_score,
            reasoning_chain, and customer_summary.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        str
            The UUID string of the newly created match record.
        """
        match_record = Match(
            user_a_id=match_data["user_a_id"],
            user_b_id=match_data["user_b_id"],
            s_vis_a_to_b=match_data["s_vis_a_to_b"],
            s_vis_b_to_a=match_data["s_vis_b_to_a"],
            s_psych=match_data["s_psych"],
            s_bio=match_data.get("s_bio"),
            wtm_score=match_data["wtm_score"],
            reasoning_chain=match_data.get("reasoning_chain"),
            customer_summary=match_data.get("customer_summary"),
        )

        db_session.add(match_record)
        await db_session.flush()

        match_id = str(match_record.id)

        logger.info(
            "match_stored",
            match_id=match_id,
            wtm_score=match_data["wtm_score"],
        )

        return match_id

    # ── Private helpers ──────────────────────────────────────────────────

    async def _get_user_primary_photo(
        self,
        user_id: str,
        db_session: AsyncSession,
    ) -> str:
        """Retrieve the primary photo path for a user.

        Falls back to an empty string if the user has no photos.

        Parameters
        ----------
        user_id:
            UUID string of the user.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        str
            Path or URL to the user's primary photo.
        """
        from app.models.user import User

        stmt = select(User).where(User.id == user_id)
        result = await db_session.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None or user.photos is None:
            logger.debug("user_primary_photo_not_found", user_id=user_id)
            return ""

        # Photos is a JSONB array of URLs — return the first one
        photos = user.photos
        if isinstance(photos, list) and len(photos) > 0:
            return str(photos[0])
        elif isinstance(photos, dict) and photos.get("primary"):
            return str(photos["primary"])

        return ""
