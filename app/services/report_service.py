"""
Harmonia V3 — Multi-Level Report Generation Service

Implements the four-tier reporting hierarchy with full evidence traceability:

  Level 3  reasoning_chain.json     Pure math, no AI — complete audit trail
  Level 2A gemini_narrative.md      Gemini Protocol A — cynical psych narrative
  Level 2B hla_gemini_analysis.md   Gemini Protocol B — MHC geneticist report
  Level 1  customer_summary.json    Sanitised, user-facing — NO evidence exposure

Evidence flow:
  parsing_evidence table
      -> EvidenceMapBuilder compiles structured maps per user
          -> Level 3 embeds full evidence maps for both users
          -> Level 2A injects maps into Gemini prompt for citation
          -> Level 1 strips ALL evidence, sin labels, raw scores
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any

import google.generativeai as genai
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import (
    HLAData,
    Match,
    ParsingEvidence,
    PersonalityProfile,
    QuestionnaireResponse,
    User,
)
from app.utils.encryption import decrypt_hla_data

logger = structlog.get_logger("harmonia.report_service")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SIN_NAMES: list[str] = [
    "greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth",
]

# Question labels used as keys in the evidence map.  Indexed 1-6.
_QUESTION_LABELS: dict[int, str] = {
    1: "q1_group_dinner",
    2: "q2_unexpected_expense",
    3: "q3_social_conflict",
    4: "q4_career_opportunity",
    5: "q5_leisure_choice",
    6: "q6_moral_dilemma",
}

# Sin weights mirrored from SimilarityService / config for report calculations
_SIN_WEIGHTS: dict[str, float] = {
    "wrath": 1.5,
    "sloth": 1.3,
    "pride": 1.2,
    "lust": 1.0,
    "greed": 0.9,
    "gluttony": 0.8,
    "envy": 0.7,
}

_TOTAL_SIN_WEIGHT: float = sum(_SIN_WEIGHTS.values())  # 7.4

# Friction threshold — sin deltas above this are flagged
_FRICTION_DELTA_THRESHOLD: float = 0.3

# Badge thresholds
_BADGE_PHYSICAL_THRESHOLD: float = 70.0
_BADGE_PERSONALITY_THRESHOLD: float = 60.0
_BADGE_CHEMISTRY_THRESHOLD: float = 75.0
_BADGE_SPARK_WTM_THRESHOLD: float = 75.0

# ──────────────────────────────────────────────────────────────────────────────
# Similarity-tier display mapping (mirrors SimilarityService thresholds)
# ──────────────────────────────────────────────────────────────────────────────

_SIMILARITY_TIER_LABELS: dict[str, str] = {
    "strong_fit": "Strong personality alignment",
    "good_fit": "Good personality alignment",
    "moderate_fit": "Moderate personality alignment",
    "low_fit": "Some personality alignment",
}

_TRAIT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "greed": {
        "virtue": "generous and easygoing about money",
        "vice": "practical and thoughtful about resources",
    },
    "pride": {
        "virtue": "humble and down-to-earth",
        "vice": "confident and self-assured",
    },
    "lust": {
        "virtue": "thoughtful and deliberate",
        "vice": "spontaneous and adventurous",
    },
    "wrath": {
        "virtue": "easygoing and harmony-seeking",
        "vice": "direct and unafraid of confrontation",
    },
    "gluttony": {
        "virtue": "balanced and moderate",
        "vice": "fun-loving and indulgent",
    },
    "envy": {
        "virtue": "content and secure",
        "vice": "ambitious and driven",
    },
    "sloth": {
        "virtue": "proactive and energetic",
        "vice": "relaxed and laid-back",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# EvidenceMapBuilder — compiles structured evidence maps from parsing_evidence
# ──────────────────────────────────────────────────────────────────────────────


class EvidenceMapBuilder:
    """Queries the ``parsing_evidence`` table and compiles a structured
    evidence map for a single user.

    The evidence map groups every sin recognition by question, including
    the exact response text, the evidence snippet, character offsets,
    and the Gemini-generated interpretation.
    """

    async def build_evidence_map(
        self,
        user_id: str,
        db_session: AsyncSession,
    ) -> dict:
        """Build a complete evidence map for a user across all questions.

        Parameters
        ----------
        user_id:
            UUID string of the user.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            Structured evidence map::

                {
                    "user_id": "abc123",
                    "evidence_map": {
                        "q1_group_dinner": {
                            "response_text": "...",
                            "sin_recognitions": [
                                {
                                    "sin": "wrath",
                                    "score": -3,
                                    "confidence": 0.88,
                                    "evidence_snippet": "...",
                                    "snippet_location": {"start": 67, "end": 142},
                                    "interpretation": "..."
                                }
                            ]
                        },
                        ...
                    }
                }
        """
        log = logger.bind(user_id=user_id)
        log.info("evidence_map_build_start")

        # Fetch all parsing evidence rows for this user
        stmt = (
            select(ParsingEvidence)
            .where(ParsingEvidence.user_id == user_id)
            .order_by(ParsingEvidence.question_number, ParsingEvidence.sin)
        )
        result = await db_session.execute(stmt)
        evidence_rows = result.scalars().all()

        # Fetch questionnaire responses to get the original response text
        resp_stmt = (
            select(QuestionnaireResponse)
            .where(QuestionnaireResponse.user_id == user_id)
            .order_by(QuestionnaireResponse.question_number)
        )
        resp_result = await db_session.execute(resp_stmt)
        responses = {
            r.question_number: r for r in resp_result.scalars().all()
        }

        # Group evidence by question number
        evidence_by_question: dict[int, list[ParsingEvidence]] = {}
        for row in evidence_rows:
            evidence_by_question.setdefault(row.question_number, []).append(row)

        # Build the structured map
        evidence_map: dict[str, dict] = {}

        for q_num in sorted(
            set(list(evidence_by_question.keys()) + list(responses.keys()))
        ):
            label = _QUESTION_LABELS.get(q_num, f"q{q_num}_unknown")
            response_obj = responses.get(q_num)
            response_text = response_obj.response_text if response_obj else ""

            sin_recognitions: list[dict] = []
            for ev in evidence_by_question.get(q_num, []):
                sin_recognitions.append({
                    "sin": ev.sin,
                    "score": ev.score,
                    "confidence": ev.confidence,
                    "evidence_snippet": ev.evidence_snippet,
                    "snippet_location": {
                        "start": ev.snippet_start_index,
                        "end": ev.snippet_end_index,
                    },
                    "interpretation": ev.interpretation or "",
                })

            evidence_map[label] = {
                "response_text": response_text,
                "sin_recognitions": sin_recognitions,
            }

        log.info(
            "evidence_map_build_complete",
            question_count=len(evidence_map),
            total_recognitions=sum(
                len(q["sin_recognitions"]) for q in evidence_map.values()
            ),
        )

        return {
            "user_id": str(user_id),
            "evidence_map": evidence_map,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ReportService — multi-level report generation
# ──────────────────────────────────────────────────────────────────────────────


class ReportService:
    """Generates all four report levels for a given match.

    Report levels:
      - Level 3: reasoning_chain.json — pure math, full evidence maps
      - Level 2A: gemini_narrative.md — Protocol A psych narrative
      - Level 2B: hla_gemini_analysis.md — Protocol B genetics report
      - Level 1: customer_summary.json — sanitised, user-facing
    """

    def __init__(self) -> None:
        self._evidence_builder = EvidenceMapBuilder()
        self._gemini_configured = False

    # ── Gemini lazy initialisation ────────────────────────────────────────

    def _ensure_gemini(self) -> None:
        """Configure the Gemini SDK if not already done."""
        if not self._gemini_configured:
            settings = get_settings()
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._gemini_configured = True

    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini with the model fallback chain and return the text
        response.

        Uses the primary model first, falling back to the secondary and
        then the stable model if earlier models fail.

        Parameters
        ----------
        prompt:
            The complete prompt string.

        Returns
        -------
        str
            The raw text content from the Gemini response.
        """
        self._ensure_gemini()
        settings = get_settings()
        model_chain = [
            settings.GEMINI_MODEL_PRIMARY,
            settings.GEMINI_MODEL_FALLBACK,
            settings.GEMINI_MODEL_STABLE,
        ]

        last_exception: Exception | None = None
        for model_name in model_chain:
            try:
                model = genai.GenerativeModel(model_name)
                response = await model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=8192,
                    ),
                )
                if response.candidates and response.text:
                    logger.debug(
                        "gemini_report_call_success",
                        model=model_name,
                    )
                    return response.text
                raise ValueError(
                    f"Gemini returned empty response from model {model_name}"
                )
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "gemini_report_call_fallback",
                    model=model_name,
                    error=str(exc),
                )
                continue

        raise RuntimeError(
            f"All Gemini models exhausted for report generation. "
            f"Last error: {last_exception}"
        )

    # ── Data fetching helpers ─────────────────────────────────────────────

    async def _fetch_match(
        self, match_id: str, db_session: AsyncSession
    ) -> Match:
        """Fetch a Match record or raise ValueError."""
        stmt = select(Match).where(Match.id == match_id)
        result = await db_session.execute(stmt)
        match = result.scalar_one_or_none()
        if match is None:
            raise ValueError(f"Match not found: {match_id}")
        return match

    async def _fetch_profile(
        self, user_id: str, db_session: AsyncSession
    ) -> PersonalityProfile | None:
        """Fetch a PersonalityProfile for a user, or None."""
        stmt = select(PersonalityProfile).where(
            PersonalityProfile.user_id == user_id
        )
        result = await db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _fetch_user(
        self, user_id: str, db_session: AsyncSession
    ) -> User | None:
        """Fetch a User record, or None."""
        stmt = select(User).where(User.id == user_id)
        result = await db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _fetch_hla_data(
        self, user_id: str, db_session: AsyncSession
    ) -> dict | None:
        """Fetch and decrypt HLA data for a user, or None."""
        stmt = select(HLAData).where(HLAData.user_id == user_id)
        result = await db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        try:
            return decrypt_hla_data(record.encrypted_data)
        except Exception:
            logger.exception("hla_decryption_failed_in_report", user_id=user_id)
            return None

    @staticmethod
    def _extract_alleles(hla_data: dict) -> list[str]:
        """Extract a flat allele list from a decrypted HLA payload."""
        if "all_alleles" in hla_data and isinstance(hla_data["all_alleles"], list):
            return hla_data["all_alleles"]
        alleles: list[str] = []
        locus_data = hla_data.get("alleles", {})
        if isinstance(locus_data, dict):
            for locus_alleles in locus_data.values():
                if isinstance(locus_alleles, list):
                    alleles.extend(locus_alleles)
        return alleles

    # ══════════════════════════════════════════════════════════════════════
    # Level 3: reasoning_chain.json — Pure math, full evidence maps
    # ══════════════════════════════════════════════════════════════════════

    async def generate_level3_report(
        self,
        match_id: str,
        db_session: AsyncSession,
    ) -> dict:
        """Generate the Level 3 reasoning chain — pure mathematical output
        with complete evidence maps for both users.

        No AI interpretation is applied at this level.  The report contains
        the full audit trail: every sin recognition can be traced back to
        the exact words each user wrote.

        Parameters
        ----------
        match_id:
            UUID of the match record.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            The complete Level 3 reasoning chain with keys:
            ``phase_1_visual``, ``phase_2_psychometric``,
            ``phase_3_biological``, ``final_calculation``,
            ``generated_at``.
        """
        log = logger.bind(match_id=match_id, report_level="L3")
        log.info("level3_generation_start")

        match = await self._fetch_match(match_id, db_session)
        user_a_id = str(match.user_a_id)
        user_b_id = str(match.user_b_id)

        # ── Fetch profiles ──────────────────────────────────────────────
        profile_a = await self._fetch_profile(user_a_id, db_session)
        profile_b = await self._fetch_profile(user_b_id, db_session)

        sins_a = profile_a.sins if profile_a else {}
        sins_b = profile_b.sins if profile_b else {}

        # ── Build evidence maps ─────────────────────────────────────────
        evidence_map_a = await self._evidence_builder.build_evidence_map(
            user_a_id, db_session
        )
        evidence_map_b = await self._evidence_builder.build_evidence_map(
            user_b_id, db_session
        )

        # ── Phase 1: Visual ─────────────────────────────────────────────
        phase_1_visual = {
            "s_vis_a_to_b": match.s_vis_a_to_b,
            "s_vis_b_to_a": match.s_vis_b_to_a,
            "visual_weight": get_settings().VISUAL_WEIGHT,
        }

        # ── Phase 2: Psychometric ───────────────────────────────────────
        # Aggregated sin vectors
        aggregated_a: dict[str, dict] = {}
        aggregated_b: dict[str, dict] = {}
        for sin in SIN_NAMES:
            aggregated_a[sin] = sins_a.get(sin, {"score": 0.0, "confidence": 0.0})
            aggregated_b[sin] = sins_b.get(sin, {"score": 0.0, "confidence": 0.0})

        # Similarity breakdown (per-sin contributions)
        similarity_breakdown: list[dict] = []
        total_contribution: float = 0.0
        overlap_count: int = 0

        for sin in SIN_NAMES:
            score_a = float(aggregated_a[sin].get("score", 0.0))
            score_b = float(aggregated_b[sin].get("score", 0.0))
            conf_a = float(aggregated_a[sin].get("confidence", 0.0))
            conf_b = float(aggregated_b[sin].get("confidence", 0.0))

            # Direction check
            shared_direction = self._check_shared_direction(score_a, score_b)

            if shared_direction is None:
                similarity_breakdown.append({
                    "sin": sin,
                    "score_a": score_a,
                    "score_b": score_b,
                    "shared_direction": None,
                    "trait_similarity": 0.0,
                    "confidence_weighted": 0.0,
                    "weighted_contribution": 0.0,
                })
                continue

            overlap_count += 1
            trait_similarity = 1.0 - (abs(score_a - score_b) / 10.0)
            avg_confidence = (conf_a + conf_b) / 2.0
            confidence_weighted = trait_similarity * avg_confidence
            weighted_contribution = confidence_weighted * _SIN_WEIGHTS[sin]
            total_contribution += weighted_contribution

            similarity_breakdown.append({
                "sin": sin,
                "score_a": round(score_a, 4),
                "score_b": round(score_b, 4),
                "shared_direction": shared_direction,
                "trait_similarity": round(trait_similarity, 4),
                "avg_confidence": round(avg_confidence, 4),
                "confidence_weighted": round(confidence_weighted, 4),
                "sin_weight": _SIN_WEIGHTS[sin],
                "weighted_contribution": round(weighted_contribution, 4),
            })

        raw_similarity = total_contribution / _TOTAL_SIN_WEIGHT

        # Quality multiplier
        tier_a = profile_a.quality_tier if profile_a else "low"
        tier_b = profile_b.quality_tier if profile_b else "low"
        quality_multiplier = self._get_quality_multiplier(tier_a, tier_b)
        adjusted_similarity = raw_similarity * quality_multiplier

        # Friction flags
        friction_flags = self._compute_friction_flags(sins_a, sins_b)

        phase_2_psychometric = {
            "evidence_map_user_a": evidence_map_a,
            "evidence_map_user_b": evidence_map_b,
            "aggregated_sin_vector_a": aggregated_a,
            "aggregated_sin_vector_b": aggregated_b,
            "quality_tier_a": tier_a,
            "quality_tier_b": tier_b,
            "quality_multiplier": quality_multiplier,
            "similarity_breakdown": similarity_breakdown,
            "raw_similarity": round(raw_similarity, 4),
            "adjusted_similarity": round(adjusted_similarity, 4),
            "overlap_count": overlap_count,
            "s_psych": match.s_psych,
            "friction_flags": friction_flags,
            "personality_weight": get_settings().PERSONALITY_WEIGHT,
        }

        # ── Phase 3: Biological ─────────────────────────────────────────
        hla_a = await self._fetch_hla_data(user_a_id, db_session)
        hla_b = await self._fetch_hla_data(user_b_id, db_session)

        phase_3_biological: dict[str, Any] = {
            "s_bio": match.s_bio,
            "hla_weight": get_settings().HLA_WEIGHT,
            "hla_available": hla_a is not None and hla_b is not None,
        }

        if hla_a is not None and hla_b is not None:
            alleles_a = self._extract_alleles(hla_a)
            alleles_b = self._extract_alleles(hla_b)
            combined = alleles_a + alleles_b
            n_unique = len(set(combined))
            n_total = max(len(combined), 12)

            phase_3_biological.update({
                "alleles_user_a": alleles_a,
                "alleles_user_b": alleles_b,
                "n_unique": n_unique,
                "n_total": n_total,
                "s_bio_calculation": f"({n_unique} / {n_total}) * 100 = {round((n_unique / n_total) * 100, 4) if n_total else 0}",
            })

        # ── Final calculation: WtM ──────────────────────────────────────
        settings = get_settings()
        vis_weight = settings.VISUAL_WEIGHT
        psych_weight = settings.PERSONALITY_WEIGHT
        hla_weight = settings.HLA_WEIGHT

        s_vis_avg = (match.s_vis_a_to_b + match.s_vis_b_to_a) / 2.0

        # Handle missing bio by redistributing weights
        if match.s_bio is not None:
            effective_vis_weight = vis_weight
            effective_psych_weight = psych_weight
            effective_hla_weight = hla_weight
            s_bio_val = match.s_bio
        else:
            # Redistribute: visual gets 57%, personality gets 43%, bio 0%
            total_non_bio = vis_weight + psych_weight
            effective_vis_weight = vis_weight / total_non_bio if total_non_bio else 0.5
            effective_psych_weight = psych_weight / total_non_bio if total_non_bio else 0.5
            effective_hla_weight = 0.0
            s_bio_val = 0.0

        # Per-direction WtM
        combined_a_to_b = (
            (effective_vis_weight * match.s_vis_a_to_b)
            + (effective_psych_weight * match.s_psych)
            + (effective_hla_weight * s_bio_val)
        )
        combined_b_to_a = (
            (effective_vis_weight * match.s_vis_b_to_a)
            + (effective_psych_weight * match.s_psych)
            + (effective_hla_weight * s_bio_val)
        )
        reciprocal_wtm = math.sqrt(
            max(0.0, combined_a_to_b) * max(0.0, combined_b_to_a)
        )

        final_calculation = {
            "s_vis_a_to_b": match.s_vis_a_to_b,
            "s_vis_b_to_a": match.s_vis_b_to_a,
            "s_vis_avg": round(s_vis_avg, 4),
            "s_psych": match.s_psych,
            "s_bio": match.s_bio,
            "weights": {
                "visual": vis_weight,
                "personality": psych_weight,
                "hla": hla_weight,
            },
            "effective_weights": {
                "visual": round(effective_vis_weight, 4),
                "personality": round(effective_psych_weight, 4),
                "hla": round(effective_hla_weight, 4),
            },
            "combined_a_to_b": round(combined_a_to_b, 4),
            "combined_b_to_a": round(combined_b_to_a, 4),
            "reciprocal_wtm": round(reciprocal_wtm, 4),
            "wtm_score": match.wtm_score,
            "formula": (
                f"combined_A_to_B = ({effective_vis_weight:.2f} x {match.s_vis_a_to_b}) "
                f"+ ({effective_psych_weight:.2f} x {match.s_psych}) "
                f"+ ({effective_hla_weight:.2f} x {s_bio_val}) "
                f"= {combined_a_to_b:.4f}"
            ),
            "formula_b_to_a": (
                f"combined_B_to_A = ({effective_vis_weight:.2f} x {match.s_vis_b_to_a}) "
                f"+ ({effective_psych_weight:.2f} x {match.s_psych}) "
                f"+ ({effective_hla_weight:.2f} x {s_bio_val}) "
                f"= {combined_b_to_a:.4f}"
            ),
            "formula_reciprocal": (
                f"reciprocal_wtm = sqrt({combined_a_to_b:.4f} x {combined_b_to_a:.4f}) "
                f"= {reciprocal_wtm:.4f}"
            ),
        }

        report = {
            "match_id": str(match_id),
            "user_a_id": user_a_id,
            "user_b_id": user_b_id,
            "report_level": 3,
            "report_type": "reasoning_chain",
            "phase_1_visual": phase_1_visual,
            "phase_2_psychometric": phase_2_psychometric,
            "phase_3_biological": phase_3_biological,
            "final_calculation": final_calculation,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "level3_generation_complete",
            wtm=match.wtm_score,
            reciprocal_wtm=round(reciprocal_wtm, 4),
        )

        return report

    # ══════════════════════════════════════════════════════════════════════
    # Level 2A: gemini_narrative.md — Protocol A (cynical psych)
    # ══════════════════════════════════════════════════════════════════════

    async def generate_level2a_report(
        self,
        match_id: str,
        db_session: AsyncSession,
    ) -> str:
        """Generate the Level 2A narrative using Gemini Protocol A.

        The system prompt instructs Gemini to act as "the Harmonia Engine,
        a cynical evolutionary psychologist."  The full evidence maps for
        both users are injected so the narrative can cite specific snippets.

        The report covers:
          - Sin deltas and their relationship implications
          - Perceived similarity analysis with overlapping trait evidence
          - Friction analysis with probability estimates
          - Verdict: Viable or Dead on Arrival

        Parameters
        ----------
        match_id:
            UUID of the match record.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        str
            The Markdown narrative report.
        """
        log = logger.bind(match_id=match_id, report_level="L2A")
        log.info("level2a_generation_start")

        match = await self._fetch_match(match_id, db_session)
        user_a_id = str(match.user_a_id)
        user_b_id = str(match.user_b_id)

        # Fetch users for display context
        user_a = await self._fetch_user(user_a_id, db_session)
        user_b = await self._fetch_user(user_b_id, db_session)

        user_a_name = user_a.display_name if user_a else "User A"
        user_b_name = user_b.display_name if user_b else "User B"

        # Fetch profiles
        profile_a = await self._fetch_profile(user_a_id, db_session)
        profile_b = await self._fetch_profile(user_b_id, db_session)

        sins_a = profile_a.sins if profile_a else {}
        sins_b = profile_b.sins if profile_b else {}

        # Build evidence maps
        evidence_map_a = await self._evidence_builder.build_evidence_map(
            user_a_id, db_session
        )
        evidence_map_b = await self._evidence_builder.build_evidence_map(
            user_b_id, db_session
        )

        # Compute sin deltas
        sin_deltas = self._compute_sin_deltas(sins_a, sins_b)

        # Friction flags
        friction_flags = self._compute_friction_flags(sins_a, sins_b)

        # Build the Protocol A prompt
        prompt = self._build_protocol_a_prompt(
            match=match,
            user_a_name=user_a_name,
            user_b_name=user_b_name,
            sins_a=sins_a,
            sins_b=sins_b,
            sin_deltas=sin_deltas,
            friction_flags=friction_flags,
            evidence_map_a=evidence_map_a,
            evidence_map_b=evidence_map_b,
        )

        # Call Gemini
        narrative = await self._call_gemini(prompt)

        log.info("level2a_generation_complete", length=len(narrative))
        return narrative

    # ══════════════════════════════════════════════════════════════════════
    # Level 2B: hla_gemini_analysis.md — Protocol B (MHC geneticist)
    # ══════════════════════════════════════════════════════════════════════

    async def generate_level2b_report(
        self,
        match_id: str,
        db_session: AsyncSession,
    ) -> str:
        """Generate the Level 2B HLA analysis using Gemini Protocol B.

        The system prompt instructs Gemini to act as "an expert Geneticist
        specializing in the Major Histocompatibility Complex."  No
        personality evidence is included — this is purely biological.

        The report covers:
          - Data integrity audit
          - Allelic dissimilarity assessment
          - Peptide-binding groove analysis
          - Olfactory / pheromonal prediction
          - Reproductive fitness estimate
          - Summary verdict

        Parameters
        ----------
        match_id:
            UUID of the match record.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        str
            The Markdown HLA analysis report.
        """
        log = logger.bind(match_id=match_id, report_level="L2B")
        log.info("level2b_generation_start")

        match = await self._fetch_match(match_id, db_session)
        user_a_id = str(match.user_a_id)
        user_b_id = str(match.user_b_id)

        # Fetch HLA data
        hla_a = await self._fetch_hla_data(user_a_id, db_session)
        hla_b = await self._fetch_hla_data(user_b_id, db_session)

        if hla_a is None or hla_b is None:
            missing = []
            if hla_a is None:
                missing.append(user_a_id)
            if hla_b is None:
                missing.append(user_b_id)
            log.warning("level2b_hla_data_missing", missing_users=missing)
            return (
                "# HLA Analysis Report\n\n"
                "**Status:** Incomplete — HLA data is missing for one or "
                "both users.\n\n"
                f"Missing data for user(s): {', '.join(missing)}\n\n"
                "This report cannot be generated without complete HLA "
                "genotype data from both participants."
            )

        alleles_a = self._extract_alleles(hla_a)
        alleles_b = self._extract_alleles(hla_b)
        combined = alleles_a + alleles_b
        n_unique = len(set(combined))
        n_total = max(len(combined), 12)
        s_bio = (n_unique / n_total) * 100.0 if n_total > 0 else 0.0
        heterozygosity = n_unique / n_total if n_total > 0 else 0.0

        # Build the Protocol B prompt
        prompt = self._build_protocol_b_prompt(
            match=match,
            alleles_a=alleles_a,
            alleles_b=alleles_b,
            n_unique=n_unique,
            n_total=n_total,
            s_bio=s_bio,
            heterozygosity=heterozygosity,
            hla_source_a=hla_a.get("source", "unknown"),
            hla_source_b=hla_b.get("source", "unknown"),
        )

        # Call Gemini
        analysis = await self._call_gemini(prompt)

        log.info("level2b_generation_complete", length=len(analysis))
        return analysis

    # ══════════════════════════════════════════════════════════════════════
    # Level 1: customer_summary.json — Sanitised user-facing output
    # ══════════════════════════════════════════════════════════════════════

    async def generate_level1_summary(
        self,
        match_id: str,
        db_session: AsyncSession,
    ) -> dict:
        """Generate the Level 1 customer-facing summary.

        **CRITICAL:** This output contains NO evidence snippets, NO raw
        sin scores, NO sin labels, and NO quoted fragments from user
        responses.  Everything is expressed in friendly natural language.

        Contents:
          - display_score (0-100)
          - badges (e.g., "Strong Chemistry", "Personality Match")
          - synopsis (headline + body)
          - compatibility_breakdown (physical/personality/chemistry)
          - shared_traits (natural language list)
          - conversation_starters

        Parameters
        ----------
        match_id:
            UUID of the match record.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            The sanitised customer summary.
        """
        log = logger.bind(match_id=match_id, report_level="L1")
        log.info("level1_generation_start")

        match = await self._fetch_match(match_id, db_session)
        user_a_id = str(match.user_a_id)
        user_b_id = str(match.user_b_id)

        # Fetch profiles
        profile_a = await self._fetch_profile(user_a_id, db_session)
        profile_b = await self._fetch_profile(user_b_id, db_session)

        sins_a = profile_a.sins if profile_a else {}
        sins_b = profile_b.sins if profile_b else {}

        # ── Display score ───────────────────────────────────────────────
        display_score = max(0, min(100, round(match.wtm_score)))

        # ── Physical score ──────────────────────────────────────────────
        physical_score = round(
            (match.s_vis_a_to_b + match.s_vis_b_to_a) / 2.0
        )
        physical_score = max(0, min(100, physical_score))

        # ── Personality score ───────────────────────────────────────────
        personality_score = max(0, min(100, round(match.s_psych)))

        # ── Chemistry score ─────────────────────────────────────────────
        chemistry_score: int | None = None
        if match.s_bio is not None:
            chemistry_score = max(0, min(100, round(match.s_bio)))

        # ── Badges ──────────────────────────────────────────────────────
        badges: list[str] = []

        if chemistry_score is not None and chemistry_score >= _BADGE_CHEMISTRY_THRESHOLD:
            badges.append("Strong Chemistry")

        if personality_score >= _BADGE_PERSONALITY_THRESHOLD:
            badges.append("Personality Match")

        if display_score >= _BADGE_SPARK_WTM_THRESHOLD:
            badges.append("Instant Spark")

        if physical_score >= _BADGE_PHYSICAL_THRESHOLD:
            badges.append("Visual Type Match")

        # ── Shared traits (natural language, NO sin labels) ─────────────
        shared_traits = self._generate_shared_traits(sins_a, sins_b)

        # ── Synopsis ────────────────────────────────────────────────────
        synopsis = self._generate_synopsis(
            display_score, shared_traits, badges
        )

        # ── Compatibility breakdown ─────────────────────────────────────
        compatibility_breakdown: dict[str, dict] = {
            "physical": {
                "score": physical_score,
                "label": self._score_to_label(physical_score),
            },
            "personality": {
                "score": personality_score,
                "label": self._score_to_label(personality_score),
            },
        }

        if chemistry_score is not None:
            compatibility_breakdown["chemistry"] = {
                "score": chemistry_score,
                "label": self._chemistry_label(chemistry_score),
            }

        # ── Conversation starters ───────────────────────────────────────
        conversation_starters = self._generate_conversation_starters(
            sins_a, sins_b, shared_traits
        )

        summary = {
            "display_score": display_score,
            "badges": badges,
            "synopsis": synopsis,
            "compatibility_breakdown": compatibility_breakdown,
            "shared_traits": shared_traits,
            "conversation_starters": conversation_starters,
        }

        log.info(
            "level1_generation_complete",
            display_score=display_score,
            badge_count=len(badges),
            trait_count=len(shared_traits),
        )

        return summary

    # ══════════════════════════════════════════════════════════════════════
    # Standalone evidence map endpoint
    # ══════════════════════════════════════════════════════════════════════

    async def get_evidence_map_for_user(
        self,
        match_id: str,
        user_id: str,
        db_session: AsyncSession,
    ) -> dict:
        """Return the standalone evidence map for a specific user within
        a match.  Intended for admin debugging of individual profiles.

        Parameters
        ----------
        match_id:
            UUID of the match (used to validate the user is part of the
            match).
        user_id:
            UUID of the user whose evidence map to retrieve.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            The structured evidence map for the user, plus match context.
        """
        log = logger.bind(match_id=match_id, user_id=user_id)
        log.info("evidence_map_fetch_start")

        # Validate the user is part of this match
        match = await self._fetch_match(match_id, db_session)
        if str(match.user_a_id) != str(user_id) and str(match.user_b_id) != str(user_id):
            raise ValueError(
                f"User {user_id} is not a participant in match {match_id}"
            )

        evidence_map = await self._evidence_builder.build_evidence_map(
            user_id, db_session
        )

        # Fetch the profile for additional context
        profile = await self._fetch_profile(user_id, db_session)

        result = {
            "match_id": str(match_id),
            "user_id": str(user_id),
            "evidence_map": evidence_map["evidence_map"],
            "profile_summary": {
                "quality_tier": profile.quality_tier if profile else None,
                "quality_score": profile.quality_score if profile else None,
                "aggregated_sins": profile.sins if profile else {},
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "evidence_map_fetch_complete",
            question_count=len(evidence_map["evidence_map"]),
        )

        return result

    # ══════════════════════════════════════════════════════════════════════
    # Internal: Prompt builders
    # ══════════════════════════════════════════════════════════════════════

    def _build_protocol_a_prompt(
        self,
        match: Match,
        user_a_name: str,
        user_b_name: str,
        sins_a: dict,
        sins_b: dict,
        sin_deltas: list[dict],
        friction_flags: list[dict],
        evidence_map_a: dict,
        evidence_map_b: dict,
    ) -> str:
        """Build the Protocol A system+user prompt for Level 2A narrative.

        Protocol A persona: "You are the Harmonia Engine, a cynical
        evolutionary psychologist..."
        """
        # Format sin comparison table
        sin_table_lines: list[str] = []
        for sin in SIN_NAMES:
            sa = sins_a.get(sin, {})
            sb = sins_b.get(sin, {})
            score_a = sa.get("score", 0.0)
            score_b = sb.get("score", 0.0)
            conf_a = sa.get("confidence", 0.0)
            conf_b = sb.get("confidence", 0.0)
            delta = abs(score_a - score_b)
            sin_table_lines.append(
                f"  {sin.upper():10s}  A={score_a:+.2f} (conf {conf_a:.2f})  "
                f"B={score_b:+.2f} (conf {conf_b:.2f})  delta={delta:.2f}"
            )
        sin_table = "\n".join(sin_table_lines)

        # Format friction flags
        friction_text = "None detected." if not friction_flags else "\n".join(
            f"  - {f['sin']}: delta={f['delta']:.2f} — {f['description']}"
            for f in friction_flags
        )

        # Format evidence maps (abbreviated for prompt length)
        evidence_a_text = self._format_evidence_for_prompt(evidence_map_a)
        evidence_b_text = self._format_evidence_for_prompt(evidence_map_b)

        prompt = f"""You are the Harmonia Engine, a cynical evolutionary psychologist with decades of experience in mate-selection research. You write forensic audit reports that dissect compatibility between two people with surgical precision. You are brutally honest, darkly witty, and grounded in evolutionary psychology. You cite specific evidence from user responses when making claims.

TASK: Generate a forensic compatibility narrative for the match between {user_a_name} (User A) and {user_b_name} (User B).

=== MATCH SCORES ===
S_vis (A→B): {match.s_vis_a_to_b:.2f}
S_vis (B→A): {match.s_vis_b_to_a:.2f}
S_psych: {match.s_psych:.2f}
S_bio: {match.s_bio if match.s_bio is not None else 'N/A'}
WtM Score: {match.wtm_score:.2f}

=== SIN SCORE COMPARISON ===
{sin_table}

=== FRICTION FLAGS ===
{friction_text}

=== USER A ({user_a_name}) — EVIDENCE MAP ===
{evidence_a_text}

=== USER B ({user_b_name}) — EVIDENCE MAP ===
{evidence_b_text}

INSTRUCTIONS:
1. For every sin where the delta between the two users exceeds {_FRICTION_DELTA_THRESHOLD}, cite BOTH users' evidence snippets and explain the relationship implication.
2. Build a perceived similarity analysis using overlapping trait evidence — quote the relevant snippets that show alignment.
3. Produce a friction analysis with probability estimates, grounded in specific snippets from both users' responses.
4. Deliver a final verdict: Viable or Dead on Arrival, with evidence-backed reasoning.

Format the output as a Markdown document with clear section headings. Be specific, cite quotes, and do not hedge. Every claim must be traceable to a snippet from the evidence maps."""

        return prompt

    def _build_protocol_b_prompt(
        self,
        match: Match,
        alleles_a: list[str],
        alleles_b: list[str],
        n_unique: int,
        n_total: int,
        s_bio: float,
        heterozygosity: float,
        hla_source_a: str,
        hla_source_b: str,
    ) -> str:
        """Build the Protocol B system+user prompt for Level 2B HLA analysis.

        Protocol B persona: "You are an expert Geneticist specializing
        in the Major Histocompatibility Complex..."
        """
        alleles_a_str = ", ".join(alleles_a) if alleles_a else "None available"
        alleles_b_str = ", ".join(alleles_b) if alleles_b else "None available"

        # Compute per-locus breakdown
        locus_breakdown_lines: list[str] = []
        for locus in ["A", "B", "C"]:
            locus_a = [a for a in alleles_a if a.startswith(f"{locus}*")]
            locus_b = [a for a in alleles_b if a.startswith(f"{locus}*")]
            shared = set(locus_a) & set(locus_b)
            unique_to_pair = set(locus_a) | set(locus_b)
            locus_breakdown_lines.append(
                f"  Locus {locus}: "
                f"A=[{', '.join(locus_a)}] "
                f"B=[{', '.join(locus_b)}] "
                f"Shared={len(shared)} Unique={len(unique_to_pair)}"
            )
        locus_breakdown = "\n".join(locus_breakdown_lines)

        prompt = f"""You are an expert Geneticist specializing in the Major Histocompatibility Complex (MHC/HLA system) and its role in human mate selection, immune complementarity, and olfactory attraction. You write detailed technical reports that bridge molecular immunology with evolutionary reproductive biology.

TASK: Generate a comprehensive HLA compatibility analysis for this matched pair.

=== DATA INTEGRITY ===
User A source: {hla_source_a}
User B source: {hla_source_b}
User A allele count: {len(alleles_a)}
User B allele count: {len(alleles_b)}

=== ALLELE DATA ===
User A: {alleles_a_str}
User B: {alleles_b_str}

=== PER-LOCUS BREAKDOWN ===
{locus_breakdown}

=== COMPUTED METRICS ===
Unique alleles (combined): {n_unique}
Total allele slots: {n_total}
S_bio = ({n_unique} / {n_total}) * 100 = {s_bio:.2f}
Heterozygosity Index = {n_unique} / {n_total} = {heterozygosity:.4f}
Match S_bio on record: {match.s_bio}

INSTRUCTIONS:
Generate a detailed Markdown report with the following sections:
1. **Data Integrity Audit** — Assess completeness and source reliability of both genotypes.
2. **Allelic Dissimilarity Assessment** — Per-locus analysis of shared vs. unique alleles. Higher dissimilarity is generally favorable for offspring immune diversity.
3. **Peptide-Binding Groove Analysis** — For known alleles, describe binding characteristics and disease associations. Note any alleles with clinically significant associations.
4. **Olfactory/Pheromonal Prediction** — Based on the heterozygosity index, predict the strength of MHC-mediated olfactory attraction signals.
5. **Reproductive Fitness Estimate** — Assess the immunological diversity advantage for potential offspring based on the combined allele pool.
6. **Summary Verdict** — Overall biological compatibility assessment with confidence level.

Be scientifically rigorous. Reference established MHC research where applicable. Do not overstate certainty — note limitations of imputed vs. directly sequenced data."""

        return prompt

    # ══════════════════════════════════════════════════════════════════════
    # Internal: Calculation helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _check_shared_direction(
        score_a: float, score_b: float, threshold: float = 0.5
    ) -> str | None:
        """Determine whether two sin scores share a meaningful direction.

        Returns "vice" if both > threshold, "virtue" if both < -threshold,
        or None otherwise.
        """
        if score_a > threshold and score_b > threshold:
            return "vice"
        if score_a < -threshold and score_b < -threshold:
            return "virtue"
        return None

    @staticmethod
    def _get_quality_multiplier(tier_a: str, tier_b: str) -> float:
        """Look up quality multiplier for a tier pair."""
        multipliers: dict[tuple[str, str], float] = {
            ("high", "high"): 1.0,
            ("high", "moderate"): 0.9,
            ("moderate", "high"): 0.9,
            ("moderate", "moderate"): 0.8,
            ("high", "low"): 0.7,
            ("low", "high"): 0.7,
            ("moderate", "low"): 0.6,
            ("low", "moderate"): 0.6,
            ("low", "low"): 0.5,
        }
        return multipliers.get((tier_a, tier_b), 0.5)

    @staticmethod
    def _compute_sin_deltas(
        sins_a: dict, sins_b: dict
    ) -> list[dict]:
        """Compute per-sin deltas between two profiles."""
        deltas: list[dict] = []
        for sin in SIN_NAMES:
            sa = sins_a.get(sin, {})
            sb = sins_b.get(sin, {})
            score_a = float(sa.get("score", 0.0))
            score_b = float(sb.get("score", 0.0))
            delta = abs(score_a - score_b)
            deltas.append({
                "sin": sin,
                "score_a": score_a,
                "score_b": score_b,
                "delta": round(delta, 4),
            })
        return deltas

    @staticmethod
    def _compute_friction_flags(
        sins_a: dict, sins_b: dict
    ) -> list[dict]:
        """Detect friction flags where sin deltas exceed the threshold."""
        friction_descriptions: dict[str, str] = {
            "wrath": "Conflict handling mismatch — one partner is confrontational while the other avoids conflict",
            "sloth": "Energy level mismatch — one partner is proactive while the other is passive",
            "pride": "Ego alignment issue — differing levels of self-assurance may cause tension",
            "lust": "Spontaneity gap — one partner craves novelty while the other prefers deliberation",
            "greed": "Resource attitude mismatch — different approaches to money and generosity",
            "gluttony": "Indulgence gap — differing attitudes toward moderation and excess",
            "envy": "Ambition mismatch — different drives around competition and contentment",
        }

        flags: list[dict] = []
        for sin in SIN_NAMES:
            sa = sins_a.get(sin, {})
            sb = sins_b.get(sin, {})
            score_a = float(sa.get("score", 0.0))
            score_b = float(sb.get("score", 0.0))
            delta = abs(score_a - score_b)

            if delta > _FRICTION_DELTA_THRESHOLD:
                flags.append({
                    "sin": sin,
                    "delta": round(delta, 4),
                    "score_a": score_a,
                    "score_b": score_b,
                    "description": friction_descriptions.get(
                        sin, f"Significant divergence on {sin} dimension"
                    ),
                })

        return flags

    def _format_evidence_for_prompt(self, evidence_data: dict) -> str:
        """Format an evidence map dict into a readable text block for
        injection into a Gemini prompt."""
        lines: list[str] = []
        evidence_map = evidence_data.get("evidence_map", {})

        for q_label, q_data in evidence_map.items():
            response_text = q_data.get("response_text", "")
            recognitions = q_data.get("sin_recognitions", [])

            lines.append(f"--- {q_label} ---")
            if response_text:
                # Truncate very long responses for prompt size management
                display_text = (
                    response_text[:500] + "..."
                    if len(response_text) > 500
                    else response_text
                )
                lines.append(f"Response: \"{display_text}\"")

            if not recognitions:
                lines.append("  (No sin recognitions recorded)")
            else:
                for rec in recognitions:
                    snippet = rec.get("evidence_snippet", "")
                    loc = rec.get("snippet_location", {})
                    start = loc.get("start", -1)
                    end = loc.get("end", -1)
                    lines.append(
                        f"  [{rec.get('sin', '?').upper()}] "
                        f"score={rec.get('score', 0):.1f} "
                        f"conf={rec.get('confidence', 0):.2f} "
                        f"snippet=\"{snippet}\" "
                        f"[chars {start}-{end}] "
                        f"— {rec.get('interpretation', '')}"
                    )
            lines.append("")

        return "\n".join(lines) if lines else "(No evidence data available)"

    # ══════════════════════════════════════════════════════════════════════
    # Internal: Level 1 helpers (all sanitised — NO sin labels/scores)
    # ══════════════════════════════════════════════════════════════════════

    def _generate_shared_traits(
        self, sins_a: dict, sins_b: dict
    ) -> list[str]:
        """Generate natural-language shared trait descriptions.

        Only traits where both users share the same direction (both
        virtuous or both vice-leaning) are included.  The output uses
        friendly language with NO sin labels.

        Parameters
        ----------
        sins_a, sins_b:
            Aggregated sin dicts for both users.

        Returns
        -------
        list[str]
            Up to 4 natural-language trait descriptions.
        """
        contributing: list[tuple[str, str, float]] = []

        for sin in SIN_NAMES:
            sa = sins_a.get(sin, {})
            sb = sins_b.get(sin, {})
            score_a = float(sa.get("score", 0.0))
            score_b = float(sb.get("score", 0.0))
            conf_a = float(sa.get("confidence", 0.0))
            conf_b = float(sb.get("confidence", 0.0))

            direction = self._check_shared_direction(score_a, score_b)
            if direction is None:
                continue

            # Compute a ranking metric: trait similarity * avg confidence * weight
            trait_sim = 1.0 - (abs(score_a - score_b) / 10.0)
            avg_conf = (conf_a + conf_b) / 2.0
            ranking = trait_sim * avg_conf * _SIN_WEIGHTS[sin]

            contributing.append((sin, direction, ranking))

        # Sort by ranking descending, take top 4
        contributing.sort(key=lambda x: x[2], reverse=True)
        top = contributing[:4]

        shared: list[str] = []
        for sin, direction, _ in top:
            description = _TRAIT_DESCRIPTIONS.get(sin, {}).get(direction, "")
            if description:
                shared.append(f"You're both {description}")

        return shared

    @staticmethod
    def _generate_synopsis(
        display_score: int,
        shared_traits: list[str],
        badges: list[str],
    ) -> dict:
        """Generate a headline and body for the match synopsis.

        All language is friendly and positive — no sin labels, no scores
        mentioned in the text.
        """
        # Headline based on score tier
        if display_score >= 80:
            headline = "An exceptional connection"
        elif display_score >= 65:
            headline = "You two have real potential"
        elif display_score >= 50:
            headline = "There's something here worth exploring"
        elif display_score >= 35:
            headline = "An interesting match with some sparks"
        else:
            headline = "A connection worth a conversation"

        # Body based on available traits and badges
        body_parts: list[str] = []

        if shared_traits:
            # Reference the trait themes without sin labels
            if len(shared_traits) >= 3:
                body_parts.append(
                    "You share several key personality traits that suggest "
                    "a natural compatibility."
                )
            elif len(shared_traits) == 2:
                body_parts.append(
                    "You share a couple of important traits that could form "
                    "the foundation of a strong connection."
                )
            else:
                body_parts.append(
                    "You have at least one significant personality trait "
                    "in common."
                )

        if "Strong Chemistry" in badges:
            body_parts.append(
                "Your biological compatibility suggests strong natural "
                "chemistry."
            )

        if "Visual Type Match" in badges:
            body_parts.append(
                "There's a strong mutual physical attraction signal."
            )

        if not body_parts:
            body_parts.append(
                "Every great relationship starts with a conversation. "
                "See what you discover about each other."
            )

        return {
            "headline": headline,
            "body": " ".join(body_parts),
        }

    @staticmethod
    def _score_to_label(score: int) -> str:
        """Convert a 0-100 compatibility score into a friendly label."""
        if score >= 80:
            return "Strong attraction"
        if score >= 65:
            return "Good alignment"
        if score >= 50:
            return "Moderate attraction"
        if score >= 35:
            return "Some alignment"
        return "Room to grow"

    @staticmethod
    def _chemistry_label(score: int) -> str:
        """Convert a chemistry score into a display label."""
        if score >= 75:
            return "Strong chemistry signal"
        if score >= 50:
            return "Good chemistry"
        if score >= 25:
            return "Some chemistry"
        return "Mild chemistry"

    def _generate_conversation_starters(
        self,
        sins_a: dict,
        sins_b: dict,
        shared_traits: list[str],
    ) -> list[str]:
        """Generate conversation starters based on shared trait themes.

        Starters are generic enough to avoid revealing the scoring
        mechanism but specific enough to be useful.  NO sin labels
        appear in the output.
        """
        # Map sin directions to conversation starter templates
        starter_templates: dict[str, dict[str, list[str]]] = {
            "greed": {
                "virtue": [
                    "Ask about their most meaningful act of generosity",
                    "Share your philosophy on experiences vs. possessions",
                ],
                "vice": [
                    "Discuss your financial goals and what motivates them",
                    "Ask about their dream investment or business idea",
                ],
            },
            "pride": {
                "virtue": [
                    "Ask about a time they gave credit to someone else",
                    "Share what being humble means to you",
                ],
                "vice": [
                    "Ask about their proudest achievement",
                    "Share what drives your ambition",
                ],
            },
            "lust": {
                "virtue": [
                    "Ask about their most carefully planned decision",
                    "Discuss what thoughtfulness means in relationships",
                ],
                "vice": [
                    "Ask about their ideal spontaneous weekend",
                    "Share your most adventurous experience",
                ],
            },
            "wrath": {
                "virtue": [
                    "Ask how they handle disagreements with friends",
                    "Share your approach to keeping the peace",
                ],
                "vice": [
                    "Ask what they're passionate enough to argue about",
                    "Discuss a time being direct really paid off",
                ],
            },
            "gluttony": {
                "virtue": [
                    "Ask about their self-care routine",
                    "Discuss your favourite way to recharge",
                ],
                "vice": [
                    "Ask about their favourite indulgence",
                    "Share your thoughts on treating yourself",
                ],
            },
            "envy": {
                "virtue": [
                    "Ask what makes them feel most content",
                    "Share what you're grateful for right now",
                ],
                "vice": [
                    "Ask about their biggest goal this year",
                    "Discuss what success looks like to both of you",
                ],
            },
            "sloth": {
                "virtue": [
                    "Ask about a project they're excited to work on",
                    "Share what gets you out of bed in the morning",
                ],
                "vice": [
                    "Ask about their ideal lazy Sunday",
                    "Share your favourite way to do absolutely nothing",
                ],
            },
        }

        starters: list[str] = []

        # Find which sins are shared and in which direction
        for sin in SIN_NAMES:
            sa = sins_a.get(sin, {})
            sb = sins_b.get(sin, {})
            score_a = float(sa.get("score", 0.0))
            score_b = float(sb.get("score", 0.0))
            direction = self._check_shared_direction(score_a, score_b)

            if direction is not None:
                templates = starter_templates.get(sin, {}).get(direction, [])
                if templates:
                    # Take the first template not already used
                    for t in templates:
                        if t not in starters:
                            starters.append(t)
                            break

            if len(starters) >= 4:
                break

        # Fallback starters if we don't have enough
        fallbacks = [
            "Ask about their favourite way to spend a weekend",
            "Share something that made you laugh recently",
            "Ask what they value most in a close friendship",
            "Discuss a book, movie, or show you've both enjoyed",
        ]
        for fb in fallbacks:
            if len(starters) >= 4:
                break
            if fb not in starters:
                starters.append(fb)

        return starters[:4]
