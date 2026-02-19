"""Visual trait extraction for Harmonia V3 preference profiling.

Analyses detected facial traits across a user's support set and computes
weighted preference profiles that feed into the Whole-the-Match scoring
pipeline.

The trait preference profile captures which visual traits a user is drawn
to (MANDATORY/PREFERRED) and which they actively dislike (AVERSION/NEGATIVE),
based on the correlation between trait presence and the user's ratings.

Note:
    The ``_detect_traits`` method is a placeholder.  In production it should
    be replaced with a real vision model call (e.g. Gemini Vision, a dedicated
    facial attribute classifier, or a pre-computed trait cache).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Trait categories and enums
# ---------------------------------------------------------------------------
class TraitCategory(str, Enum):
    """High-level trait categories for facial attribute analysis."""

    GLASSES = "glasses"
    FACIAL_HAIR = "facial_hair"
    HAIR_COLOR = "hair_color"
    HAIR_LENGTH = "hair_length"
    SMILE = "smile"
    EYE_COLOR = "eye_color"
    FACE_SHAPE = "face_shape"
    SKIN_TONE = "skin_tone"
    AGE_RANGE = "age_range"
    JAWLINE = "jawline"
    CHEEKBONES = "cheekbones"
    NOSE_SHAPE = "nose_shape"
    EYEBROW_SHAPE = "eyebrow_shape"
    LIP_SHAPE = "lip_shape"


class PreferenceLevel(str, Enum):
    """Preference classification for a trait."""

    MANDATORY = "MANDATORY"      # >80% in liked faces
    PREFERRED = "PREFERRED"      # 60-80% in liked faces
    AVERSION = "AVERSION"        # >80% in disliked faces
    NEGATIVE = "NEGATIVE"        # 60-80% in disliked faces
    NEUTRAL = "NEUTRAL"          # No strong signal


# ---------------------------------------------------------------------------
# Trait values by category (used by the placeholder detector)
# ---------------------------------------------------------------------------
TRAIT_VALUES: dict[str, list[str]] = {
    TraitCategory.GLASSES: ["none", "glasses", "sunglasses"],
    TraitCategory.FACIAL_HAIR: ["none", "stubble", "beard", "mustache", "goatee"],
    TraitCategory.HAIR_COLOR: ["black", "brown", "blonde", "red", "gray", "other"],
    TraitCategory.HAIR_LENGTH: ["short", "medium", "long", "bald"],
    TraitCategory.SMILE: ["none", "slight", "broad"],
    TraitCategory.EYE_COLOR: ["brown", "blue", "green", "hazel", "gray"],
    TraitCategory.FACE_SHAPE: ["oval", "round", "square", "heart", "oblong"],
    TraitCategory.SKIN_TONE: ["light", "medium", "olive", "tan", "dark"],
    TraitCategory.AGE_RANGE: ["20s", "30s", "40s", "50s"],
    TraitCategory.JAWLINE: ["soft", "defined", "angular"],
    TraitCategory.CHEEKBONES: ["flat", "moderate", "prominent"],
    TraitCategory.NOSE_SHAPE: ["straight", "button", "aquiline", "wide", "narrow"],
    TraitCategory.EYEBROW_SHAPE: ["straight", "arched", "thick", "thin"],
    TraitCategory.LIP_SHAPE: ["thin", "medium", "full"],
}


@dataclass
class TraitPreference:
    """A single trait preference with computed weight and classification."""

    trait_category: str
    trait_value: str
    frequency_liked: float        # 0-1, frequency in liked faces (ratings 4-5)
    frequency_disliked: float     # 0-1, frequency in disliked faces (ratings 1-2)
    rating_correlation: float     # Pearson r between trait presence and rating
    weight: float                 # W_trait = F_trait * R_corr * sign(avg_rating - 3.0)
    level: PreferenceLevel        # Classification
    avg_rating_when_present: float  # Mean rating when this trait is present


@dataclass
class TraitProfile:
    """Complete trait preference profile for a user."""

    user_id: str
    preferences: list[TraitPreference] = field(default_factory=list)
    num_support_images: int = 0
    num_liked: int = 0
    num_disliked: int = 0
    num_neutral: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the profile to a plain dict for storage."""
        return {
            "user_id": self.user_id,
            "num_support_images": self.num_support_images,
            "num_liked": self.num_liked,
            "num_disliked": self.num_disliked,
            "num_neutral": self.num_neutral,
            "preferences": [
                {
                    "trait_category": p.trait_category,
                    "trait_value": p.trait_value,
                    "frequency_liked": round(p.frequency_liked, 4),
                    "frequency_disliked": round(p.frequency_disliked, 4),
                    "rating_correlation": round(p.rating_correlation, 4),
                    "weight": round(p.weight, 4),
                    "level": p.level.value,
                    "avg_rating_when_present": round(p.avg_rating_when_present, 2),
                }
                for p in self.preferences
            ],
            "mandatory_traits": self.mandatory_traits,
            "preferred_traits": self.preferred_traits,
            "aversion_traits": self.aversion_traits,
            "negative_traits": self.negative_traits,
        }

    @property
    def mandatory_traits(self) -> list[str]:
        """Traits classified as MANDATORY."""
        return [
            f"{p.trait_category}:{p.trait_value}"
            for p in self.preferences
            if p.level == PreferenceLevel.MANDATORY
        ]

    @property
    def preferred_traits(self) -> list[str]:
        """Traits classified as PREFERRED."""
        return [
            f"{p.trait_category}:{p.trait_value}"
            for p in self.preferences
            if p.level == PreferenceLevel.PREFERRED
        ]

    @property
    def aversion_traits(self) -> list[str]:
        """Traits classified as AVERSION."""
        return [
            f"{p.trait_category}:{p.trait_value}"
            for p in self.preferences
            if p.level == PreferenceLevel.AVERSION
        ]

    @property
    def negative_traits(self) -> list[str]:
        """Traits classified as NEGATIVE."""
        return [
            f"{p.trait_category}:{p.trait_value}"
            for p in self.preferences
            if p.level == PreferenceLevel.NEGATIVE
        ]


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------
class TraitExtractor:
    """Extracts and analyses visual trait preferences from a user's support set.

    For each facial trait detected across the support images, the extractor:

    1. Computes frequency in liked (rating 4-5) vs disliked (rating 1-2) faces
    2. Computes a correlation between trait presence and ratings
    3. Calculates a preference weight:
       ``W_trait = F_trait * R_corr * sign(avg_rating - 3.0)``
    4. Classifies the trait preference level:
       - MANDATORY:  >80% frequency in liked faces
       - PREFERRED:  60-80% frequency in liked faces
       - AVERSION:   >80% frequency in disliked faces
       - NEGATIVE:   60-80% frequency in disliked faces
       - NEUTRAL:    No strong signal in either direction

    Note:
        The ``_detect_traits`` method is a placeholder.  Replace with a real
        vision model in production (e.g. Gemini Vision API, a dedicated
        facial attribute classifier, or a pre-computed attribute cache).
    """

    # Rating thresholds
    LIKED_THRESHOLD = 4     # Ratings >= this are "liked"
    DISLIKED_THRESHOLD = 2  # Ratings <= this are "disliked"

    # Classification thresholds
    HIGH_FREQUENCY = 0.80   # >80% = MANDATORY or AVERSION
    MED_FREQUENCY = 0.60    # 60-80% = PREFERRED or NEGATIVE

    def __init__(self) -> None:
        """Initialise the trait extractor."""
        logger.info("trait_extractor.init")

    # ------------------------------------------------------------------ #
    # Placeholder trait detection
    # ------------------------------------------------------------------ #
    def _detect_traits(self, image_path: str) -> dict[str, str]:
        """Detect facial traits in a single image.

        **Placeholder implementation.**  Uses a deterministic hash of the image
        path to produce stable pseudo-random traits for development and testing.
        In production, replace this with a real vision model call.

        Args:
            image_path: Path to the face image.

        Returns:
            Dict mapping trait category names to detected trait values, e.g.:
            ``{"glasses": "none", "hair_color": "brown", "smile": "broad", ...}``
        """
        # Deterministic seed from image path for reproducible placeholder results
        seed = int(hashlib.sha256(image_path.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        traits: dict[str, str] = {}
        for category, values in TRAIT_VALUES.items():
            cat_name = category.value if isinstance(category, TraitCategory) else category
            traits[cat_name] = rng.choice(values)

        return traits

    def _detect_traits_batch(
        self, image_paths: list[str]
    ) -> list[dict[str, str]]:
        """Detect traits for a batch of images.

        Args:
            image_paths: List of paths to face images.

        Returns:
            List of trait dicts, one per image.
        """
        return [self._detect_traits(path) for path in image_paths]

    # ------------------------------------------------------------------ #
    # Correlation computation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pearson_correlation(
        trait_presence: list[float], ratings: list[float]
    ) -> float:
        """Compute Pearson correlation between trait presence (0/1) and ratings.

        Returns 0.0 if variance is zero (constant values).
        """
        n = len(trait_presence)
        if n < 2:
            return 0.0

        mean_x = sum(trait_presence) / n
        mean_y = sum(ratings) / n

        var_x = sum((x - mean_x) ** 2 for x in trait_presence)
        var_y = sum((y - mean_y) ** 2 for y in ratings)

        if var_x == 0 or var_y == 0:
            return 0.0

        cov = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(trait_presence, ratings)
        )

        return cov / math.sqrt(var_x * var_y)

    # ------------------------------------------------------------------ #
    # Preference classification
    # ------------------------------------------------------------------ #
    @classmethod
    def _classify_preference(
        cls,
        frequency_liked: float,
        frequency_disliked: float,
        num_liked: int,
        num_disliked: int,
    ) -> PreferenceLevel:
        """Classify a trait's preference level based on its frequency distribution.

        Args:
            frequency_liked: Fraction of liked faces (4-5) with this trait.
            frequency_disliked: Fraction of disliked faces (1-2) with this trait.
            num_liked: Total number of liked faces.
            num_disliked: Total number of disliked faces.

        Returns:
            The ``PreferenceLevel`` classification.
        """
        # Need at least some data in a bucket to classify in that direction
        if num_liked > 0 and frequency_liked > cls.HIGH_FREQUENCY:
            return PreferenceLevel.MANDATORY
        if num_disliked > 0 and frequency_disliked > cls.HIGH_FREQUENCY:
            return PreferenceLevel.AVERSION
        if num_liked > 0 and frequency_liked > cls.MED_FREQUENCY:
            return PreferenceLevel.PREFERRED
        if num_disliked > 0 and frequency_disliked > cls.MED_FREQUENCY:
            return PreferenceLevel.NEGATIVE

        return PreferenceLevel.NEUTRAL

    # ------------------------------------------------------------------ #
    # Main analysis pipeline
    # ------------------------------------------------------------------ #
    def analyse_support_set(
        self,
        user_id: str,
        image_paths: list[str],
        ratings: list[int],
    ) -> TraitProfile:
        """Analyse a user's support set and build a trait preference profile.

        Args:
            user_id: Unique identifier for the user.
            image_paths: Paths to the user's rated face images.
            ratings: Integer ratings (1-5) for each image.

        Returns:
            A ``TraitProfile`` with computed preferences for every detected trait.

        Raises:
            ValueError: If image_paths and ratings have mismatched lengths.
        """
        if len(image_paths) != len(ratings):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and ratings "
                f"({len(ratings)}) must have the same length."
            )

        logger.info(
            "trait_extractor.analyse_start",
            user_id=user_id,
            num_images=len(image_paths),
        )

        # Step 1: Detect traits for all support images
        all_traits = self._detect_traits_batch(image_paths)

        # Partition indices by rating bucket
        liked_indices = [i for i, r in enumerate(ratings) if r >= self.LIKED_THRESHOLD]
        disliked_indices = [i for i, r in enumerate(ratings) if r <= self.DISLIKED_THRESHOLD]
        neutral_indices = [
            i for i, r in enumerate(ratings)
            if self.DISLIKED_THRESHOLD < r < self.LIKED_THRESHOLD
        ]

        num_liked = len(liked_indices)
        num_disliked = len(disliked_indices)
        num_neutral = len(neutral_indices)

        # Step 2: For each unique (category, value) pair, compute frequencies
        # Collect all unique trait-value pairs
        trait_value_occurrences: dict[tuple[str, str], list[int]] = defaultdict(list)
        for img_idx, traits in enumerate(all_traits):
            for category, value in traits.items():
                trait_value_occurrences[(category, value)].append(img_idx)

        preferences: list[TraitPreference] = []

        for (category, value), occurrence_indices in trait_value_occurrences.items():
            occurrence_set = set(occurrence_indices)

            # Frequency in liked faces
            liked_with_trait = len(occurrence_set & set(liked_indices))
            freq_liked = liked_with_trait / num_liked if num_liked > 0 else 0.0

            # Frequency in disliked faces
            disliked_with_trait = len(occurrence_set & set(disliked_indices))
            freq_disliked = (
                disliked_with_trait / num_disliked if num_disliked > 0 else 0.0
            )

            # Binary trait presence for correlation: 1.0 if image has this trait
            trait_presence = [
                1.0 if i in occurrence_set else 0.0
                for i in range(len(image_paths))
            ]
            float_ratings = [float(r) for r in ratings]
            r_corr = self._pearson_correlation(trait_presence, float_ratings)

            # Average rating when trait is present
            present_ratings = [ratings[i] for i in occurrence_indices]
            avg_rating = (
                sum(present_ratings) / len(present_ratings)
                if present_ratings
                else 3.0
            )

            # Compute weight: W_trait = F_trait * R_corr * sign(avg_rating - 3.0)
            sign = 1.0 if avg_rating >= 3.0 else -1.0
            f_trait = max(freq_liked, freq_disliked)  # Dominant frequency
            weight = f_trait * abs(r_corr) * sign

            # Classify preference level
            level = self._classify_preference(
                freq_liked, freq_disliked, num_liked, num_disliked
            )

            pref = TraitPreference(
                trait_category=category,
                trait_value=value,
                frequency_liked=freq_liked,
                frequency_disliked=freq_disliked,
                rating_correlation=r_corr,
                weight=weight,
                level=level,
                avg_rating_when_present=avg_rating,
            )
            preferences.append(pref)

        # Sort by absolute weight descending (strongest signals first)
        preferences.sort(key=lambda p: abs(p.weight), reverse=True)

        profile = TraitProfile(
            user_id=user_id,
            preferences=preferences,
            num_support_images=len(image_paths),
            num_liked=num_liked,
            num_disliked=num_disliked,
            num_neutral=num_neutral,
        )

        logger.info(
            "trait_extractor.analyse_complete",
            user_id=user_id,
            num_traits=len(preferences),
            mandatory=len(profile.mandatory_traits),
            preferred=len(profile.preferred_traits),
            aversion=len(profile.aversion_traits),
            negative=len(profile.negative_traits),
        )

        return profile

    def extract_profile_dict(
        self,
        user_id: str,
        image_paths: list[str],
        ratings: list[int],
    ) -> dict[str, Any]:
        """Convenience method: analyse and return serialised dict.

        Args:
            user_id: Unique identifier for the user.
            image_paths: Paths to the user's rated face images.
            ratings: Integer ratings (1-5) for each image.

        Returns:
            Serialised trait preference profile as a dict.
        """
        profile = self.analyse_support_set(user_id, image_paths, ratings)
        return profile.to_dict()
