"""
Harmonia V3 — Profile Aggregation Pipeline (PIIP Spec)

Implements the 6-step profile-building pipeline:
  1. Validate each Q&A pair
  2. Organise scores by trait
  3. Aggregate each trait (CWMV or simple mean)
  4. Detect response styles (ERS, MRS, patterns)
  5. Calculate composite quality score (0-100)
  6. Compile final profile JSON

Persistence is handled by ``save_profile`` which writes to the
``personality_profiles`` table with automatic version incrementing.
"""

from __future__ import annotations

import math
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session_factory
from app.models.profile import PersonalityProfile

logger = structlog.get_logger("harmonia.profile_service")

# ──────────────────────────────────────────────────────────────────────────────
# Placeholder / low-signal patterns used during validation (step 1).
# ──────────────────────────────────────────────────────────────────────────────

_PLACEHOLDER_PHRASES: set[str] = {
    "i don't know",
    "no comment",
    "n/a",
    "nothing to say",
    "test",
    "asdf",
    "lorem ipsum",
}


class ProfileService:
    """Orchestrates the PIIP profile-aggregation pipeline.

    All scoring constants, thresholds, and sin definitions are exposed as
    class-level attributes so they can be introspected or overridden in tests.
    """

    # ── Constants ─────────────────────────────────────────────────────────

    SIN_NAMES: list[str] = [
        "greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth",
    ]

    SIN_WEIGHTS: dict[str, float] = {
        "wrath": 1.5,
        "sloth": 1.3,
        "pride": 1.2,
        "lust": 1.0,
        "greed": 0.9,
        "gluttony": 0.8,
        "envy": 0.7,
    }

    MIN_QUESTIONS: int = 4
    MIN_WORD_COUNT: int = 25
    LOW_WORD_COUNT: int = 50
    HIGH_VARIANCE_THRESHOLD: float = 3.0
    ERS_THRESHOLD: float = 0.40
    MRS_THRESHOLD: float = 0.50

    # ══════════════════════════════════════════════════════════════════════
    # 1. build_profile — full pipeline entry point
    # ══════════════════════════════════════════════════════════════════════

    async def build_profile(
        self,
        user_id: str,
        parsed_responses: list[dict],
        response_metadata: list[dict] | None = None,
    ) -> dict:
        """Run the full 6-step pipeline and return the compiled profile dict.

        Parameters
        ----------
        user_id:
            Unique identifier for the user.
        parsed_responses:
            List of per-question parsing results.  Each dict is expected to
            contain at least::

                {
                    "question_number": int,
                    "response_text": str,
                    "word_count": int,            # optional — derived if absent
                    "scores": [
                        {
                            "sin": str,
                            "score": float,       # -5 … +5
                            "confidence": float,  # 0 … 1
                            "evidence": str,
                        },
                        ...
                    ],
                }

        response_metadata:
            Optional list of dicts carrying timing / engagement signals per
            question, e.g. ``{"question_number": 1, "response_time_s": 42.5,
            "word_count": 67}``.

        Returns
        -------
        dict
            The fully compiled profile JSON (see step 6).
        """
        log = logger.bind(user_id=user_id)
        log.info("pipeline_start", n_responses=len(parsed_responses))

        # Step 1 — Validate each Q&A pair
        valid_responses = self._validate_responses(parsed_responses, log)
        log.info(
            "step1_validation_complete",
            valid=len(valid_responses),
            rejected=len(parsed_responses) - len(valid_responses),
        )

        if len(valid_responses) < self.MIN_QUESTIONS:
            log.warning(
                "insufficient_valid_responses",
                valid=len(valid_responses),
                required=self.MIN_QUESTIONS,
            )
            return self._compile_profile(
                user_id=user_id,
                trait_data={sin: _empty_trait(sin) for sin in self.SIN_NAMES},
                quality_score=0.0,
                quality_tier="rejected",
                style_info={"flags": ["insufficient_responses"], "details": {}},
                flags=["insufficient_responses"],
            )

        # Step 2 — Organise scores by trait (7 lists, up to 6 scores each)
        trait_buckets = self._organise_by_trait(valid_responses, log)

        # Step 3 — Aggregate each trait (CWMV or simple mean)
        trait_data: dict[str, dict] = {}
        for sin in self.SIN_NAMES:
            scores_for_sin = trait_buckets.get(sin, [])
            if not scores_for_sin:
                trait_data[sin] = _empty_trait(sin)
                continue
            confidences = [s.get("confidence", 0.5) for s in scores_for_sin]
            if len(confidences) >= 2 and statistics.stdev(confidences) >= 0.10:
                trait_data[sin] = self._aggregate_trait_cwmv(scores_for_sin)
            else:
                trait_data[sin] = self._aggregate_trait_simple(scores_for_sin)

        log.info("step3_aggregation_complete", traits=list(trait_data.keys()))

        # Step 4 — Detect response styles (ERS, MRS, patterns)
        all_scores = self._collect_all_scores(trait_buckets)
        style_info = self._detect_response_styles(all_scores, response_metadata)
        log.info("step4_style_detection", flags=style_info["flags"])

        # Tier-1 outlier flags
        flags = self._detect_outliers(trait_buckets)
        flags.extend(style_info["flags"])
        log.info("step4_outlier_flags", flags=flags)

        # Step 5 — Calculate composite quality score
        quality_score, quality_tier = self._calculate_quality_score(
            trait_data, style_info, response_metadata,
        )
        log.info(
            "step5_quality",
            quality_score=round(quality_score, 2),
            quality_tier=quality_tier,
        )

        # Step 6 — Compile final profile JSON with metadata
        profile = self._compile_profile(
            user_id=user_id,
            trait_data=trait_data,
            quality_score=quality_score,
            quality_tier=quality_tier,
            style_info=style_info,
            flags=flags,
        )
        log.info("pipeline_complete", quality_tier=quality_tier)
        return profile

    # ══════════════════════════════════════════════════════════════════════
    # 2. _aggregate_trait_cwmv — Confidence-Weighted Mean Voting
    # ══════════════════════════════════════════════════════════════════════

    def _aggregate_trait_cwmv(self, scores: list[dict]) -> dict:
        """Confidence-Weighted Mean Voting.

        Formula::

            Score_trait = Σ(score × confidence) / Σ(confidence)

        Variance penalty: if the SD of raw scores exceeds
        ``HIGH_VARIANCE_THRESHOLD`` (3.0), the aggregate confidence is reduced
        by ``min(0.3, (SD - 3.0) × 0.1)``.

        Parameters
        ----------
        scores:
            List of score dicts, each with ``"score"`` and ``"confidence"`` keys.

        Returns
        -------
        dict
            Aggregated trait with ``score``, ``confidence``, ``variance``,
            ``n_questions``, and ``method`` keys.
        """
        raw_scores = [s.get("score", 0.0) for s in scores]
        confidences = [s.get("confidence", 0.5) for s in scores]

        weighted_sum = sum(sc * cf for sc, cf in zip(raw_scores, confidences))
        confidence_sum = sum(confidences)

        if confidence_sum == 0:
            aggregated_score = 0.0
            aggregated_confidence = 0.0
        else:
            aggregated_score = weighted_sum / confidence_sum
            aggregated_confidence = confidence_sum / len(confidences)

        # Variance penalty
        variance = statistics.stdev(raw_scores) if len(raw_scores) >= 2 else 0.0
        if variance > self.HIGH_VARIANCE_THRESHOLD:
            penalty = min(0.3, (variance - self.HIGH_VARIANCE_THRESHOLD) * 0.1)
            aggregated_confidence = max(0.0, aggregated_confidence - penalty)

        return {
            "score": round(aggregated_score, 4),
            "confidence": round(aggregated_confidence, 4),
            "variance": round(variance, 4),
            "n_questions": len(scores),
            "method": "cwmv",
        }

    # ══════════════════════════════════════════════════════════════════════
    # 3. _aggregate_trait_simple — Simple mean fallback
    # ══════════════════════════════════════════════════════════════════════

    def _aggregate_trait_simple(self, scores: list[dict]) -> dict:
        """Simple arithmetic mean — used when confidence SD < 0.10.

        Parameters
        ----------
        scores:
            List of score dicts, each with ``"score"`` and ``"confidence"`` keys.

        Returns
        -------
        dict
            Aggregated trait with ``score``, ``confidence``, ``variance``,
            ``n_questions``, and ``method`` keys.
        """
        raw_scores = [s.get("score", 0.0) for s in scores]
        confidences = [s.get("confidence", 0.5) for s in scores]

        aggregated_score = statistics.mean(raw_scores) if raw_scores else 0.0
        aggregated_confidence = statistics.mean(confidences) if confidences else 0.0
        variance = statistics.stdev(raw_scores) if len(raw_scores) >= 2 else 0.0

        return {
            "score": round(aggregated_score, 4),
            "confidence": round(aggregated_confidence, 4),
            "variance": round(variance, 4),
            "n_questions": len(scores),
            "method": "simple_mean",
        }

    # ══════════════════════════════════════════════════════════════════════
    # 4. _detect_outliers — Tier 1 flags
    # ══════════════════════════════════════════════════════════════════════

    def _detect_outliers(self, trait_scores: dict[str, list[dict]]) -> list[str]:
        """Tier 1 flags: zero variance, extreme scores, score reversals.

        Checks
        ------
        - **Zero variance**: all scores for a trait are identical (requires
          all 6 question entries present).
        - **Extreme scores**: any individual score at +/-5 with confidence > 0.9.
        - **Score reversals**: adjacent questions (sorted by question_number)
          with a delta >= 8 on the 11-point scale.

        Parameters
        ----------
        trait_scores:
            Dict mapping sin name to list of score-entry dicts (with
            ``"score"``, ``"confidence"``, and ``"question_number"`` keys).

        Returns
        -------
        list[str]
            Human-readable flag strings.
        """
        flags: list[str] = []

        for sin, entries in trait_scores.items():
            raw = [e.get("score", 0.0) for e in entries]
            if not raw:
                continue

            # Zero variance — all scores identical for a trait (need all 6)
            if len(raw) >= 6 and len(set(raw)) == 1:
                flags.append(f"zero_variance:{sin}")

            # Extreme scores: ±5 with high confidence
            for entry in entries:
                score = entry.get("score", 0.0)
                confidence = entry.get("confidence", 0.0)
                if abs(score) >= 5.0 and confidence > 0.9:
                    flags.append(
                        f"extreme_score:{sin}:q{entry.get('question_number', '?')}"
                    )

            # Score reversals: adjacent questions with delta >= 8
            sorted_entries = sorted(entries, key=lambda e: e.get("question_number", 0))
            for i in range(len(sorted_entries) - 1):
                s1 = sorted_entries[i].get("score", 0.0)
                s2 = sorted_entries[i + 1].get("score", 0.0)
                if abs(s2 - s1) >= 8.0:
                    q1 = sorted_entries[i].get("question_number", "?")
                    q2 = sorted_entries[i + 1].get("question_number", "?")
                    flags.append(f"score_reversal:{sin}:q{q1}-q{q2}")

        return flags

    # ══════════════════════════════════════════════════════════════════════
    # 5. _detect_response_styles — ERS, MRS, pattern detection
    # ══════════════════════════════════════════════════════════════════════

    def _detect_response_styles(
        self,
        all_scores: list[dict],
        response_metadata: list[dict] | None = None,
    ) -> dict:
        """Detect ERS, MRS, and pattern-based response styles.

        Detection rules
        ---------------
        - **ERS (Extreme Response Style)**: >40% of all scores at ±4 or ±5.
        - **MRS (Midpoint Response Style)**: >50% of scores near midpoint
          (-0.5 to +0.5) AND fast completion (median response time < 15 s).
        - **Pattern — identical sins**: all 7 sins receive the same mean
          score for a question.
        - **Pattern — alternating extremes**: at least 4 consecutive
          alternations between positive and negative extremes (magnitude >= 4).

        Parameters
        ----------
        all_scores:
            Flat list of every individual score dict across all traits.
        response_metadata:
            Optional per-question metadata with ``response_time_s`` and
            ``word_count`` fields.

        Returns
        -------
        dict
            ``{"flags": [...], "details": {...}}``
        """
        flags: list[str] = []
        details: dict[str, Any] = {}

        if not all_scores:
            return {"flags": flags, "details": details}

        raw_values = [s.get("score", 0.0) for s in all_scores]
        total = len(raw_values)

        # ── ERS (Extreme Response Style) ──────────────────────────────────
        extreme_count = sum(1 for v in raw_values if abs(v) >= 4.0)
        ers_ratio = extreme_count / total if total else 0.0
        details["ers_ratio"] = round(ers_ratio, 4)
        if ers_ratio > self.ERS_THRESHOLD:
            flags.append("ers_extreme_response_style")

        # ── MRS (Midpoint Response Style) ─────────────────────────────────
        midpoint_count = sum(1 for v in raw_values if -0.5 <= v <= 0.5)
        mrs_ratio = midpoint_count / total if total else 0.0
        details["mrs_ratio"] = round(mrs_ratio, 4)

        fast_completion = self._has_fast_completion(response_metadata)
        details["fast_completion"] = fast_completion
        if mrs_ratio > self.MRS_THRESHOLD and fast_completion:
            flags.append("mrs_midpoint_response_style")

        # ── Pattern: identical scores across all 7 sins for a question ────
        per_question: dict[int, list[float]] = {}
        for s in all_scores:
            qn = s.get("question_number")
            if qn is not None:
                per_question.setdefault(qn, []).append(s.get("score", 0.0))

        for qn, qscores in per_question.items():
            if len(qscores) == 7 and len(set(qscores)) == 1:
                flags.append(f"pattern_identical_sin_scores:q{qn}")
                details.setdefault("identical_sin_questions", []).append(qn)

        # ── Pattern: alternating extremes ─────────────────────────────────
        if self._has_alternating_extremes(raw_values):
            flags.append("pattern_alternating_extremes")

        return {"flags": flags, "details": details}

    # ══════════════════════════════════════════════════════════════════════
    # 6. _calculate_quality_score — composite 0-100
    # ══════════════════════════════════════════════════════════════════════

    def _calculate_quality_score(
        self,
        trait_aggregates: dict[str, dict],
        response_styles: dict,
        response_metadata: list[dict] | None = None,
    ) -> tuple[float, str]:
        """Compute composite quality score (0-100) from four equal components.

        Components (each contributes 0-25 points):

        1. **Internal consistency** — mean confidence across traits,
           normalised to 0-25 (confidence is already 0-1, so multiply by 25).
        2. **Response variance** — optimal range [1.0, 6.0] yields full 25;
           too low suggests mechanical responding, too high suggests noise.
        3. **Response style** — deduct 25 per flag detected (max deduction 25).
        4. **Engagement** — word count + response time signals from metadata.

        Tiers
        -----
        - ``"high"`` : score >= 80
        - ``"moderate"`` : 60 <= score < 80
        - ``"low"`` : score < 60

        Parameters
        ----------
        trait_aggregates:
            Dict mapping sin name to aggregated trait dict (from step 3).
        response_styles:
            Dict with ``"flags"`` key (from ``_detect_response_styles``).
        response_metadata:
            Optional per-question metadata.

        Returns
        -------
        tuple[float, str]
            ``(score, tier)``
        """
        style_flags = response_styles.get("flags", [])

        # ── 1. Internal consistency (0-25) ────────────────────────────────
        confidences: list[float] = []
        for td in trait_aggregates.values():
            conf = td.get("confidence", 0.0)
            if td.get("n_questions", 0) > 0:
                confidences.append(conf)
        mean_conf = statistics.mean(confidences) if confidences else 0.0
        consistency_score = mean_conf * 25.0  # confidence is 0-1

        # ── 2. Response variance (0-25) ───────────────────────────────────
        variances: list[float] = []
        for td in trait_aggregates.values():
            v = td.get("variance", 0.0)
            if td.get("n_questions", 0) >= 2:
                variances.append(v)
        if variances:
            avg_var = statistics.mean(variances)
            # Optimal range [1.0, 6.0] -> full score; outside -> linear penalty
            if 1.0 <= avg_var <= 6.0:
                variance_score = 25.0
            elif avg_var < 1.0:
                # Too little variance — could be mechanical responding
                variance_score = max(0.0, 25.0 * avg_var)
            else:
                # Too much variance — noisy / incoherent
                variance_score = max(0.0, 25.0 * (1.0 - (avg_var - 6.0) / 4.0))
        else:
            variance_score = 0.0

        # ── 3. Response style (0-25) ──────────────────────────────────────
        style_penalty = min(25.0, len(style_flags) * 25.0)
        style_score = max(0.0, 25.0 - style_penalty)

        # ── 4. Engagement (0-25) ──────────────────────────────────────────
        engagement_score = self._compute_engagement(response_metadata)

        # ── Composite ─────────────────────────────────────────────────────
        total = consistency_score + variance_score + style_score + engagement_score
        total = max(0.0, min(100.0, total))

        if total >= 80.0:
            tier = "high"
        elif total >= 60.0:
            tier = "moderate"
        else:
            tier = "low"

        return round(total, 2), tier

    # ══════════════════════════════════════════════════════════════════════
    # 7. save_profile — persist to personality_profiles with versioning
    # ══════════════════════════════════════════════════════════════════════

    async def save_profile(
        self,
        user_id: str,
        profile_data: dict,
        source: str = "real_user",
    ) -> dict:
        """Save a profile to the ``personality_profiles`` table.

        If the user already has a profile, the existing record's version is
        read, the new profile receives ``version + 1``, and the old row is
        replaced (the table has a unique constraint on ``user_id``).

        Parameters
        ----------
        user_id:
            UUID string for the user.
        profile_data:
            The compiled profile dict returned by ``build_profile``.
        source:
            One of ``"real_user"`` or ``"claude_agent"``.

        Returns
        -------
        dict
            ``{"profile_id": str, "version": int, "quality_tier": str}``
        """
        log = logger.bind(user_id=user_id)
        log.info("save_profile_start", source=source)

        async with async_session_factory() as session:
            # Determine version — look up any existing profile for this user
            stmt = (
                select(PersonalityProfile.version)
                .where(PersonalityProfile.user_id == uuid.UUID(user_id))
                .order_by(PersonalityProfile.version.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            existing_version = result.scalar_one_or_none()
            new_version = (existing_version or 0) + 1

            # If an existing profile exists, remove it so the unique constraint
            # on user_id is satisfied (upsert semantics).
            if existing_version is not None:
                delete_stmt = (
                    select(PersonalityProfile)
                    .where(PersonalityProfile.user_id == uuid.UUID(user_id))
                )
                existing = (await session.execute(delete_stmt)).scalar_one_or_none()
                if existing is not None:
                    await session.delete(existing)
                    await session.flush()
                log.info(
                    "save_profile_version_increment",
                    old_version=existing_version,
                    new_version=new_version,
                )

            profile_id = uuid.uuid4()
            new_profile = PersonalityProfile(
                id=profile_id,
                user_id=uuid.UUID(user_id),
                version=new_version,
                sins=profile_data.get("sins", {}),
                quality_score=profile_data.get("quality_score", 0.0),
                quality_tier=profile_data.get("quality_tier", "low"),
                response_styles=profile_data.get("response_styles"),
                flags=profile_data.get("flags"),
                metadata_=profile_data.get("metadata"),
                source=source,
            )

            session.add(new_profile)
            await session.commit()

            log.info(
                "save_profile_complete",
                profile_id=str(profile_id),
                version=new_version,
                quality_tier=new_profile.quality_tier,
            )

            return {
                "profile_id": str(profile_id),
                "version": new_version,
                "quality_tier": new_profile.quality_tier,
            }

    # ══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════

    # ── Step 1: validation ────────────────────────────────────────────────

    def _validate_responses(
        self,
        parsed_responses: list[dict],
        log: Any,
    ) -> list[dict]:
        """Return only the Q&A dicts that pass all validation gates.

        Gates
        -----
        1. Word count >= ``MIN_WORD_COUNT`` (25).
        2. Not placeholder / nonsense text.
        3. At least one score entry with non-trivial signal.
        """
        valid: list[dict] = []
        for resp in parsed_responses:
            qn = resp.get("question_number", "?")
            text = resp.get("response_text", "")
            word_count = resp.get("word_count") or len(text.split())

            # Gate 1: minimum word count
            if word_count < self.MIN_WORD_COUNT:
                log.debug("validation_reject_word_count", question=qn, words=word_count)
                continue

            # Gate 2: placeholder / nonsense text
            if self._is_placeholder(text):
                log.debug("validation_reject_placeholder", question=qn)
                continue

            # Gate 3: must have at least one score with signal
            scores = resp.get("scores", [])
            if not self._has_signal(scores):
                log.debug("validation_reject_no_signal", question=qn)
                continue

            valid.append(resp)
        return valid

    @staticmethod
    def _is_placeholder(text: str) -> bool:
        """Return True if the text looks like placeholder / junk content."""
        normalised = text.strip().lower()
        if normalised in _PLACEHOLDER_PHRASES:
            return True
        # Repetitive single-character padding (e.g. "aaaa…")
        unique_chars = set(normalised.replace(" ", ""))
        if len(unique_chars) <= 2 and len(normalised) > 10:
            return True
        return False

    @staticmethod
    def _has_signal(scores: list[dict]) -> bool:
        """Return True if at least one score entry conveys non-trivial signal."""
        for s in scores:
            confidence = s.get("confidence", 0.0)
            score = s.get("score", 0.0)
            if abs(score) > 0.0 and confidence > 0.0:
                return True
        return False

    # ── Step 2: organise by trait ─────────────────────────────────────────

    def _organise_by_trait(
        self,
        valid_responses: list[dict],
        log: Any,
    ) -> dict[str, list[dict]]:
        """Bucket individual score dicts into per-sin lists (max 6 each)."""
        buckets: dict[str, list[dict]] = {sin: [] for sin in self.SIN_NAMES}
        for resp in valid_responses:
            qn = resp.get("question_number")
            for score_entry in resp.get("scores", []):
                sin = score_entry.get("sin", "").lower()
                if sin not in buckets:
                    log.warning("unknown_sin_ignored", sin=sin, question=qn)
                    continue
                # Attach question_number for downstream outlier detection
                entry = {**score_entry, "question_number": qn}
                if len(buckets[sin]) < 6:
                    buckets[sin].append(entry)
        log.info(
            "step2_organise_complete",
            counts={sin: len(v) for sin, v in buckets.items()},
        )
        return buckets

    # ── Step 6: compile ───────────────────────────────────────────────────

    def _compile_profile(
        self,
        user_id: str,
        trait_data: dict[str, dict],
        quality_score: float,
        quality_tier: str,
        style_info: dict,
        flags: list[str],
        source: str = "real_user",
    ) -> dict:
        """Return the final profile JSON structure.

        This matches the ``PersonalityProfile`` model columns and the
        ``ProfileResponse`` schema.
        """
        sins: dict[str, dict] = {}
        for sin in self.SIN_NAMES:
            td = trait_data.get(sin, _empty_trait(sin))
            sins[sin] = {
                "score": td["score"],
                "confidence": td["confidence"],
                "variance": td["variance"],
                "n_questions": td["n_questions"],
                "method": td["method"],
                "weight": self.SIN_WEIGHTS.get(sin, 1.0),
            }

        return {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "version": 1,
            "sins": sins,
            "quality_score": quality_score,
            "quality_tier": quality_tier,
            "response_styles": style_info,
            "flags": flags,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "pipeline": "piip_v1",
                "sin_weights": self.SIN_WEIGHTS,
                "thresholds": {
                    "min_questions": self.MIN_QUESTIONS,
                    "min_word_count": self.MIN_WORD_COUNT,
                    "high_variance": self.HIGH_VARIANCE_THRESHOLD,
                    "ers": self.ERS_THRESHOLD,
                    "mrs": self.MRS_THRESHOLD,
                },
            },
        }

    # ── Misc helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _collect_all_scores(trait_buckets: dict[str, list[dict]]) -> list[dict]:
        """Flatten all per-sin score entries into a single list."""
        out: list[dict] = []
        for entries in trait_buckets.values():
            out.extend(entries)
        return out

    @staticmethod
    def _has_fast_completion(metadata: list[dict] | None) -> bool:
        """Heuristic: if median response_time_s < 15 s, flag fast completion."""
        if not metadata:
            return False
        times = [
            m["response_time_s"]
            for m in metadata
            if "response_time_s" in m and m["response_time_s"] is not None
        ]
        if not times:
            return False
        median_time = statistics.median(times)
        return median_time < 15.0

    @staticmethod
    def _has_alternating_extremes(values: list[float], threshold: float = 4.0) -> bool:
        """True if scores alternate between positive and negative extremes.

        Looks for a run of at least 4 alternations where each adjacent pair
        flips sign and both have magnitude >= ``threshold``.
        """
        if len(values) < 4:
            return False
        alternations = 0
        for i in range(len(values) - 1):
            a, b = values[i], values[i + 1]
            if abs(a) >= threshold and abs(b) >= threshold and (a * b < 0):
                alternations += 1
            else:
                alternations = 0
            if alternations >= 3:
                return True
        return False

    @staticmethod
    def _compute_engagement(metadata: list[dict] | None) -> float:
        """Compute the engagement component (0-25).

        Sub-components (each worth up to 12.5):
          - Word count signal: mean words normalised to [25, 150] range
          - Response time signal: median time normalised to [15, 120] s range
        """
        if not metadata:
            # Without metadata, assign a moderate default
            return 12.5

        # ── Word count sub-component (0-12.5) ────────────────────────────
        word_counts = [
            m["word_count"]
            for m in metadata
            if "word_count" in m and m["word_count"] is not None
        ]
        if word_counts:
            mean_wc = statistics.mean(word_counts)
            # Map [25, 150] -> [0, 1]
            wc_norm = max(0.0, min(1.0, (mean_wc - 25) / (150.0 - 25)))
            wc_score = wc_norm * 12.5
        else:
            wc_score = 6.25  # moderate default

        # ── Response time sub-component (0-12.5) ─────────────────────────
        times = [
            m["response_time_s"]
            for m in metadata
            if "response_time_s" in m and m["response_time_s"] is not None
        ]
        if times:
            median_time = statistics.median(times)
            # Map [15, 120] -> [0, 1]; below 15 is suspicious, above 120 is fine
            if median_time < 15.0:
                time_norm = max(0.0, median_time / 15.0) * 0.5  # penalty for speed
            else:
                time_norm = min(1.0, (median_time - 15.0) / (120.0 - 15.0))
            time_score = time_norm * 12.5
        else:
            time_score = 6.25  # moderate default

        return round(wc_score + time_score, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helper
# ──────────────────────────────────────────────────────────────────────────────

def _empty_trait(sin: str) -> dict:
    """Return a zeroed-out trait dict for a sin with no data."""
    return {
        "score": 0.0,
        "confidence": 0.0,
        "variance": 0.0,
        "n_questions": 0,
        "method": "none",
    }
