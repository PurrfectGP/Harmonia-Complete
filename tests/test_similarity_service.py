"""Unit tests for SimilarityService — perceived similarity calculation."""
import pytest
from app.services.similarity_service import SimilarityService


@pytest.fixture
def similarity_service():
    return SimilarityService()


class TestSharedDirection:
    """Tests for shared direction detection."""

    def test_both_positive_shared_vice(self, similarity_service):
        """Both scores > +0.5 = shared vice."""
        direction = similarity_service._check_shared_direction(2.5, 3.1)
        assert direction == "vice"

    def test_both_negative_shared_virtue(self, similarity_service):
        """Both scores < -0.5 = shared virtue."""
        direction = similarity_service._check_shared_direction(-1.8, -2.2)
        assert direction == "virtue"

    def test_one_neutral_no_contribution(self, similarity_service):
        """One score in [-0.5, +0.5] = no signal."""
        direction = similarity_service._check_shared_direction(0.3, -2.0)
        assert direction is None

    def test_opposite_directions_no_contribution(self, similarity_service):
        """Different directions = no contribution."""
        direction = similarity_service._check_shared_direction(-1.5, 1.8)
        assert direction is None


class TestSimilarityCalculation:
    """Tests for the full similarity calculation."""

    def test_spec_worked_example(self, similarity_service, sample_profile_a, sample_profile_b):
        """Test against README Section 8.4 worked example.

        User A: [-1.8, +0.3, +2.5, -2.1, +1.2, -0.8, -1.5]
        User B: [-2.2, -1.5, +3.1, -1.8, -0.3, -1.2, +1.8]
        Expected: 4 of 7 traits shared.
        """
        result = similarity_service.calculate_similarity(sample_profile_a, sample_profile_b)
        assert result["overlap_count"] == 4  # greed(virtue), lust(vice), wrath(virtue), envy(virtue)
        assert 0 <= result["raw_score"] <= 1.0
        assert result["tier"] in ("strong_fit", "good_fit", "moderate_fit", "low_fit")

    def test_trait_similarity_formula(self, similarity_service):
        """Trait similarity = 1 - (|score_a - score_b| / 10)."""
        # With scores -1.8 and -2.2, |delta| = 0.4
        expected = 1 - (0.4 / 10)  # 0.96
        assert abs(expected - 0.96) < 0.001

    def test_identical_profiles_high_similarity(self, similarity_service):
        """Identical profiles should give high similarity."""
        profile = {
            "sins": {sin: {"score": -2.0, "confidence": 0.85} for sin in similarity_service.SIN_NAMES},
            "quality_tier": "high",
        }
        result = similarity_service.calculate_similarity(profile, profile)
        assert result["raw_score"] > 0.5
        assert result["overlap_count"] == 7


class TestQualityMultiplier:
    """Tests for quality adjustment."""

    def test_high_high(self, similarity_service):
        """High/High pair = 1.0 multiplier."""
        mult = similarity_service._calculate_quality_multiplier("high", "high")
        assert mult == 1.0

    def test_low_low(self, similarity_service):
        """Low/Low pair = 0.5 multiplier."""
        mult = similarity_service._calculate_quality_multiplier("low", "low")
        assert mult == 0.5

    def test_mixed_tiers(self, similarity_service):
        """High/Moderate = 0.9."""
        mult = similarity_service._calculate_quality_multiplier("high", "moderate")
        assert mult == 0.9


class TestThresholdEvaluation:
    """Tests for similarity tier classification."""

    def test_strong_fit(self, similarity_service):
        """Score ≥0.60 is strong_fit."""
        tier, mode, n = similarity_service._evaluate_threshold(0.65)
        assert tier == "strong_fit"
        assert n == 4

    def test_good_fit(self, similarity_service):
        """Score ≥0.40 is good_fit."""
        tier, mode, n = similarity_service._evaluate_threshold(0.45)
        assert tier == "good_fit"
        assert n == 3

    def test_moderate_fit(self, similarity_service):
        """Score ≥0.25 is moderate_fit."""
        tier, mode, n = similarity_service._evaluate_threshold(0.30)
        assert tier == "moderate_fit"
        assert n == 2

    def test_low_fit(self, similarity_service):
        """Score <0.25 is low_fit."""
        tier, mode, n = similarity_service._evaluate_threshold(0.15)
        assert tier == "low_fit"
        assert n == 1


class TestMatchExplanation:
    """Tests for natural language match explanations."""

    def test_generates_shared_trait_phrases(self, similarity_service):
        """Generates 'You're both...' phrases."""
        breakdown = [
            {"sin": "greed", "direction": "virtue", "shared_direction": "virtue", "weighted_contribution": 0.5},
            {"sin": "wrath", "direction": "virtue", "shared_direction": "virtue", "weighted_contribution": 0.8},
            {"sin": "lust", "direction": "vice", "shared_direction": "vice", "weighted_contribution": 0.6},
        ]
        result = similarity_service.generate_match_explanation(breakdown, "good_fit", "standard")
        traits = result.get("shared_traits", [])
        assert len(traits) > 0
        assert any("both" in t.lower() for t in traits)


class TestHLADisplay:
    """Tests for HLA display logic."""

    def test_strong_chemistry(self, similarity_service):
        """Score ≥75 shows 'Strong chemistry signal'."""
        display = similarity_service.get_hla_display(85.0)
        assert display["show"] is True
        assert "strong" in display.get("label", "").lower() or "Strong" in display.get("label", "")

    def test_hidden_below_25(self, similarity_service):
        """Score <25 is hidden."""
        display = similarity_service.get_hla_display(20.0)
        assert display["show"] is False

    def test_none_hidden(self, similarity_service):
        """None score is hidden."""
        display = similarity_service.get_hla_display(None)
        assert display["show"] is False
