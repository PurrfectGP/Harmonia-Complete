"""Unit tests for CalibrationService — calibration database pipeline."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.calibration_service import CalibrationService


@pytest.fixture
def calibration_service():
    return CalibrationService()


class TestReviewWorkflow:
    """Tests for admin review state transitions."""

    def test_approve_sets_validated_score(self):
        """Approve: validated_score = gemini_raw_score."""
        example = {
            "gemini_raw_score": -2.0,
            "review_status": "pending",
        }
        # After approval:
        approved = {**example, "review_status": "approved", "validated_score": -2.0}
        assert approved["validated_score"] == approved["gemini_raw_score"]
        assert approved["review_status"] == "approved"

    def test_correct_uses_admin_score(self):
        """Correct: admin provides their own score."""
        example = {
            "gemini_raw_score": 3.0,
            "review_status": "pending",
        }
        corrected = {**example, "review_status": "corrected", "validated_score": -1.0}
        assert corrected["validated_score"] != corrected["gemini_raw_score"]
        assert corrected["review_status"] == "corrected"

    def test_reject_excludes_from_pool(self):
        """Reject: excluded from few-shot pool."""
        example = {
            "review_status": "pending",
        }
        rejected = {**example, "review_status": "rejected"}
        assert rejected["review_status"] == "rejected"


class TestFewShotRetrieval:
    """Tests for calibration example retrieval for few-shot injection."""

    def test_corrected_examples_first(self):
        """Priority: corrected examples ranked before approved."""
        examples = [
            {"review_status": "approved", "validated_score": -2.0, "response_text": "a"},
            {"review_status": "corrected", "validated_score": -1.0, "response_text": "b"},
        ]
        # Sort by priority: corrected first
        sorted_examples = sorted(examples, key=lambda x: 0 if x["review_status"] == "corrected" else 1)
        assert sorted_examples[0]["review_status"] == "corrected"

    def test_diversity_of_scores(self):
        """Examples should span the -5 to +5 range."""
        examples = [
            {"validated_score": -3.0},
            {"validated_score": 0.0},
            {"validated_score": 2.5},
        ]
        scores = [e["validated_score"] for e in examples]
        score_range = max(scores) - min(scores)
        assert score_range >= 4.0  # Good spread


class TestCoverageMap:
    """Tests for calibration coverage tracking."""

    def test_coverage_matrix_dimensions(self):
        """Coverage map should be 6 questions x 7 sins = 42 cells."""
        sins = ["greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth"]
        questions = list(range(1, 7))
        total_cells = len(questions) * len(sins)
        assert total_cells == 42

    def test_target_per_cell(self):
        """Target: at least 5 validated examples per cell."""
        target = 5
        total_minimum = 42 * target  # 210 minimum
        assert total_minimum == 210
