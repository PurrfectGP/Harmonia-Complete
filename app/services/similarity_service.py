"""
Harmonia V3 — Perceived similarity calculation and match explanation engine.

Computes a quality-adjusted similarity score between two personality profiles
based on shared sin-trait alignment, and generates natural-language match
explanations for customer-facing match cards.

5-step calculation pipeline:
  1. Determine shared direction (both > +0.5 = shared vice,
     both < -0.5 = shared virtue).  Neutral or opposing -> skip.
  2. Trait similarity:  1 - (|score_a - score_b| / 10)
  3. Confidence weighting:  trait_similarity * avg_confidence
  4. Apply sin-specific weights
  5. Normalise:  sum(contributions) / 7.4
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class SimilarityService:
    """Calculate perceived similarity between two personality profiles and
    produce human-readable match explanations.

    Only traits where both users share the same direction (both virtuous OR
    both vice-leaning) contribute positively — differences are never
    penalised.  This creates the "astrology effect."
    """

    # ── Sin axis definitions ────────────────────────────────────────
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

    NEUTRAL_THRESHOLD: float = 0.5   # scores between -0.5 and +0.5 treated as no signal
    TOTAL_WEIGHT: float = 7.4        # sum of all sin weights
    DEFAULT_STAGE2_THRESHOLD: float = 0.40

    # ── Quality multiplier table (tier_a, tier_b) -> multiplier ─────
    QUALITY_MULTIPLIERS: dict[tuple[str, str], float] = {
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

    # ── Trait-to-description mapping for match explanations ─────────
    TRAIT_DESCRIPTIONS: dict[str, dict[str, str]] = {
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

    # ── Public API ──────────────────────────────────────────────────

    def calculate_similarity(
        self, profile_a: dict, profile_b: dict
    ) -> dict:
        """Main entry point.  Compute perceived-similarity between two
        personality profiles.

        Parameters
        ----------
        profile_a, profile_b : dict
            Each must contain:
              - ``sins``: dict mapping sin name -> {"score": float, "confidence": float}
              - ``quality_tier``: str  ("high" | "moderate" | "low")

        Returns
        -------
        dict with keys:
            raw_score, adjusted_score, quality_multiplier,
            breakdown (list), overlap_count, tier, display_mode
        """
        sins_a = profile_a["sins"]
        sins_b = profile_b["sins"]
        tier_a = profile_a["quality_tier"]
        tier_b = profile_b["quality_tier"]

        logger.info(
            "similarity.calculate_start",
            tier_a=tier_a,
            tier_b=tier_b,
        )

        # ------ per-trait contributions (steps 1-4) ------------------
        breakdown: list[dict] = []
        total_contribution: float = 0.0
        overlap_count: int = 0

        for sin in self.SIN_NAMES:
            entry_a = sins_a.get(sin, {})
            entry_b = sins_b.get(sin, {})
            score_a: float = float(entry_a.get("score", 0.0))
            score_b: float = float(entry_b.get("score", 0.0))
            conf_a: float = float(entry_a.get("confidence", 0.0))
            conf_b: float = float(entry_b.get("confidence", 0.0))

            # Step 1: direction alignment check
            shared_direction = self._check_shared_direction(score_a, score_b)
            if shared_direction is None:
                # One in neutral zone or opposite directions -> skip
                breakdown.append(
                    {
                        "sin": sin,
                        "score_a": score_a,
                        "score_b": score_b,
                        "shared_direction": None,
                        "trait_similarity": 0.0,
                        "confidence_weighted": 0.0,
                        "weighted_contribution": 0.0,
                    }
                )
                continue

            overlap_count += 1

            # Step 2: trait similarity = 1 - (|score_a - score_b| / 10)
            trait_similarity = 1.0 - (abs(score_a - score_b) / 10.0)

            # Step 3: confidence-weighted = trait_similarity * avg confidence
            avg_confidence = (conf_a + conf_b) / 2.0
            confidence_weighted = trait_similarity * avg_confidence

            # Step 4: apply sin weight
            sin_weight = self.SIN_WEIGHTS[sin]
            weighted_contribution = confidence_weighted * sin_weight

            total_contribution += weighted_contribution

            breakdown.append(
                {
                    "sin": sin,
                    "score_a": score_a,
                    "score_b": score_b,
                    "shared_direction": shared_direction,
                    "trait_similarity": round(trait_similarity, 4),
                    "confidence_weighted": round(confidence_weighted, 4),
                    "weighted_contribution": round(weighted_contribution, 4),
                }
            )

        # Step 5: normalise
        raw_score = total_contribution / self.TOTAL_WEIGHT

        # ------ quality adjustment -----------------------------------
        quality_multiplier = self._calculate_quality_multiplier(tier_a, tier_b)
        adjusted_score = raw_score * quality_multiplier

        # ------ tier / display mode ----------------------------------
        tier, display_mode, _n_traits = self._evaluate_threshold(adjusted_score)

        logger.info(
            "similarity.calculate_done",
            raw_score=round(raw_score, 4),
            adjusted_score=round(adjusted_score, 4),
            quality_multiplier=quality_multiplier,
            overlap_count=overlap_count,
            tier=tier,
            display_mode=display_mode,
        )

        return {
            "raw_score": round(raw_score, 4),
            "adjusted_score": round(adjusted_score, 4),
            "quality_multiplier": quality_multiplier,
            "breakdown": breakdown,
            "overlap_count": overlap_count,
            "tier": tier,
            "display_mode": display_mode,
        }

    def generate_match_explanation(
        self, breakdown: list[dict], tier: str, display_mode: str
    ) -> dict:
        """Generate natural-language trait descriptions for a match card.

        Selects the top *n* traits (determined by display_mode) by
        ``weighted_contribution`` from the breakdown and produces
        "You're both ..." phrases using :pyattr:`TRAIT_DESCRIPTIONS`.

        Parameters
        ----------
        breakdown : list[dict]
            The ``breakdown`` list returned by :pymeth:`calculate_similarity`.
        tier : str
            Match tier (e.g. "strong_fit", "good_fit", etc.).
        display_mode : str
            Display mode ("highlight", "standard", "minimal", "chemistry_focus").

        Returns
        -------
        dict
            Contains ``shared_traits`` (list of description strings),
            ``trait_count`` (int), ``tier`` (str), and ``display_mode`` (str).
        """
        # Map display_mode to number of traits to show
        mode_to_n_traits: dict[str, int] = {
            "highlight": 4,
            "standard": 3,
            "minimal": 2,
            "chemistry_focus": 1,
        }
        n_traits = mode_to_n_traits.get(display_mode, 3)

        logger.debug(
            "similarity.generate_explanation",
            tier=tier,
            display_mode=display_mode,
            n_traits=n_traits,
        )

        # Filter to traits that actually contributed (shared direction)
        contributing = [
            entry for entry in breakdown if entry.get("shared_direction") is not None
        ]

        # Sort by weighted_contribution descending
        contributing.sort(key=lambda e: e["weighted_contribution"], reverse=True)

        # Take top n
        top_traits = contributing[:n_traits]

        shared_traits: list[str] = []
        for entry in top_traits:
            sin = entry["sin"]
            direction = entry["shared_direction"]
            description = self.TRAIT_DESCRIPTIONS.get(sin, {}).get(direction, "")
            if description:
                shared_traits.append(f"You're both {description}")

        logger.debug(
            "similarity.explanations_generated",
            count=len(shared_traits),
            traits=[e["sin"] for e in top_traits],
        )

        return {
            "shared_traits": shared_traits,
            "trait_count": len(shared_traits),
            "tier": tier,
            "display_mode": display_mode,
        }

    def get_hla_display(self, hla_score: float | None) -> dict:
        """Return display metadata for an HLA chemistry score.

        Parameters
        ----------
        hla_score : float or None
            HLA-derived chemistry score in [0, 100], or None if unavailable.

        Returns
        -------
        dict
            Keys: ``show`` (bool), and optionally ``emoji`` (str), ``label`` (str).

        Thresholds
        ----------
        >= 75 : show=True, emoji="fire", label="Strong chemistry signal"
        50-74 : show=True, emoji="sparkles", label="Good chemistry"
        25-49 : show=True, emoji="dizzy", label="Some chemistry"
        <  25 : show=False
        """
        if hla_score is None or hla_score < 25:
            logger.debug("similarity.hla_display", show=False, hla_score=hla_score)
            return {"show": False}

        if hla_score >= 75:
            display = {
                "show": True,
                "emoji": "\U0001f525",   # fire
                "label": "Strong chemistry signal",
            }
        elif hla_score >= 50:
            display = {
                "show": True,
                "emoji": "\u2728",       # sparkles
                "label": "Good chemistry",
            }
        else:  # 25-49
            display = {
                "show": True,
                "emoji": "\U0001f4ab",   # dizzy / shooting star
                "label": "Some chemistry",
            }

        logger.debug("similarity.hla_display", show=True, label=display["label"])
        return display

    def assemble_match_card(
        self, similarity_result: dict, hla_result: dict | None = None
    ) -> dict:
        """Combine personality explanation and HLA display into a
        customer-facing match card JSON payload.

        Parameters
        ----------
        similarity_result : dict
            Result from :pymeth:`calculate_similarity`.
        hla_result : dict or None
            Result from :pymeth:`get_hla_display`, or None if HLA data is
            unavailable.

        Returns
        -------
        dict
            Customer-facing match card with personality and chemistry sections.
        """
        breakdown = similarity_result["breakdown"]
        tier = similarity_result["tier"]
        display_mode = similarity_result["display_mode"]

        # Generate the personality explanation
        explanation = self.generate_match_explanation(breakdown, tier, display_mode)

        card: dict = {
            "personality": {
                "tier": tier,
                "display_mode": display_mode,
                "score": similarity_result["adjusted_score"],
                "overlap_count": similarity_result["overlap_count"],
                "shared_traits": explanation["shared_traits"],
            },
        }

        # Attach chemistry section only when HLA data is present and visible
        hla_display = hla_result if hla_result is not None else {"show": False}
        if hla_display.get("show"):
            card["chemistry"] = {
                "emoji": hla_display["emoji"],
                "label": hla_display["label"],
            }

        logger.info(
            "similarity.match_card_assembled",
            tier=tier,
            display_mode=display_mode,
            trait_count=len(explanation["shared_traits"]),
            has_chemistry=hla_display.get("show", False),
        )

        return card

    # ── Internal helpers ────────────────────────────────────────────

    def _check_shared_direction(
        self, score_a: float, score_b: float
    ) -> str | None:
        """Determine whether two sin scores share a meaningful direction.

        Returns
        -------
        "vice" if both > +NEUTRAL_THRESHOLD,
        "virtue" if both < -NEUTRAL_THRESHOLD,
        None otherwise (neutral zone or opposing directions).
        """
        a_positive = score_a > self.NEUTRAL_THRESHOLD
        a_negative = score_a < -self.NEUTRAL_THRESHOLD
        b_positive = score_b > self.NEUTRAL_THRESHOLD
        b_negative = score_b < -self.NEUTRAL_THRESHOLD

        if a_positive and b_positive:
            return "vice"
        if a_negative and b_negative:
            return "virtue"
        return None

    def _calculate_quality_multiplier(
        self, tier_a: str, tier_b: str
    ) -> float:
        """Look up the quality multiplier for the given tier combination.

        Quality multiplier table
        ------------------------
        High / High       = 1.0
        High / Moderate   = 0.9
        Moderate / Moderate = 0.8
        High / Low        = 0.7
        Moderate / Low    = 0.6
        Low / Low         = 0.5

        Parameters
        ----------
        tier_a : str
            Quality tier of profile A ("high", "moderate", or "low").
        tier_b : str
            Quality tier of profile B ("high", "moderate", or "low").

        Returns
        -------
        float
            The multiplier to apply to the raw similarity score.
        """
        return self.QUALITY_MULTIPLIERS.get((tier_a, tier_b), 0.5)

    def _evaluate_threshold(
        self, adjusted_score: float
    ) -> tuple[str, str, int]:
        """Classify an adjusted similarity score into a match tier.

        This is a *soft gate* — low similarity changes the display messaging
        but never blocks the match.

        Parameters
        ----------
        adjusted_score : float
            Quality-adjusted similarity score.

        Returns
        -------
        tuple of (tier, display_mode, n_traits_to_show)
            - strong_fit  (>= 0.60): "highlight" mode, 4 traits
            - good_fit    (>= 0.40): "standard" mode, 3 traits
            - moderate_fit(>= 0.25): "minimal" mode, 2 traits
            - low_fit     (<  0.25): "chemistry_focus" mode, 1 trait
        """
        if adjusted_score >= 0.60:
            return ("strong_fit", "highlight", 4)
        if adjusted_score >= 0.40:
            return ("good_fit", "standard", 3)
        if adjusted_score >= 0.25:
            return ("moderate_fit", "minimal", 2)
        return ("low_fit", "chemistry_focus", 1)
