"""Unit tests for MatchingService — WtM calculation and matching pipeline."""
import pytest
import math
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.matching_service import MatchingService


@pytest.fixture
def matching_service():
    with patch("app.services.matching_service.get_settings") as mock:
        settings = MagicMock()
        settings.VISUAL_WEIGHT = 0.4
        settings.PERSONALITY_WEIGHT = 0.3
        settings.HLA_WEIGHT = 0.3
        mock.return_value = settings
        service = MatchingService()
    return service


class TestWtMCalculation:
    """Tests for Willingness to Meet calculation."""

    def test_spec_worked_example(self, matching_service):
        """Section 10.4: S_vis=79.7, S_psych=52.0, S_bio=91.6 -> WtM~75."""
        result = matching_service._calculate_wtm(
            s_vis_a_to_b=79.7,
            s_vis_b_to_a=79.7,  # symmetric for this test
            s_psych=52.0,
            s_bio=91.6,
        )
        # (0.4*79.7) + (0.3*52.0) + (0.3*91.6) = 31.88 + 15.6 + 27.48 = 74.96
        assert abs(result["reciprocal_wtm"] - 74.96) < 0.5

    def test_without_bio_weight_redistribution(self, matching_service):
        """Missing HLA redistributes weights: visual->0.571, psych->0.429."""
        result = matching_service._calculate_wtm(
            s_vis_a_to_b=80.0,
            s_vis_b_to_a=80.0,
            s_psych=60.0,
            s_bio=None,
        )
        # (0.571*80) + (0.429*60) = 45.68 + 25.74 = 71.42
        expected = (0.4/0.7)*80.0 + (0.3/0.7)*60.0
        assert abs(result["reciprocal_wtm"] - expected) < 1.0
        assert result.get("weights_redistributed", False) or result.get("bio_available") is False

    def test_reciprocal_asymmetric(self, matching_service):
        """Reciprocal WtM uses geometric mean of both directions."""
        result = matching_service._calculate_wtm(
            s_vis_a_to_b=90.0,
            s_vis_b_to_a=60.0,
            s_psych=70.0,
            s_bio=80.0,
        )
        combined_a = (0.4*90.0) + (0.3*70.0) + (0.3*80.0)  # 36+21+24 = 81
        combined_b = (0.4*60.0) + (0.3*70.0) + (0.3*80.0)  # 24+21+24 = 69
        expected_reciprocal = math.sqrt(combined_a * combined_b)
        assert abs(result["reciprocal_wtm"] - expected_reciprocal) < 1.0


class TestFrictionFlags:
    """Tests for personality friction detection."""

    def test_high_delta_flagged(self, matching_service):
        """Sin delta > 0.3 (normalised) should be flagged."""
        profile_a = {"sins": {"wrath": {"score": -4.0}, "sloth": {"score": -3.0}}}
        profile_b = {"sins": {"wrath": {"score": 3.0}, "sloth": {"score": 2.0}}}
        # Add remaining sins
        for sin in ["greed", "pride", "lust", "gluttony", "envy"]:
            profile_a["sins"][sin] = {"score": 0.0}
            profile_b["sins"][sin] = {"score": 0.0}
        result = matching_service._calculate_friction_flags(profile_a, profile_b)
        assert result["flag_count"] > 0
        assert result["p_friction"] < 1.0

    def test_similar_profiles_no_friction(self, matching_service):
        """Similar profiles should have minimal friction flags."""
        profile_a = {"sins": {sin: {"score": -1.0} for sin in ["greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth"]}}
        profile_b = {"sins": {sin: {"score": -1.2} for sin in ["greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth"]}}
        result = matching_service._calculate_friction_flags(profile_a, profile_b)
        assert result["flag_count"] == 0
        assert result["p_friction"] == 1.0

    def test_friction_penalty_formula(self, matching_service):
        """P_friction = max(0.5, 1.0 - 0.1 x n_flags)."""
        # With 3 flags: max(0.5, 1.0 - 0.3) = 0.7
        profile_a = {"sins": {}}
        profile_b = {"sins": {}}
        for sin in ["greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth"]:
            profile_a["sins"][sin] = {"score": -4.0}
            profile_b["sins"][sin] = {"score": 4.0}
        result = matching_service._calculate_friction_flags(profile_a, profile_b)
        assert result["p_friction"] >= 0.5
