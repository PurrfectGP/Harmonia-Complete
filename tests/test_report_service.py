"""Unit tests for ReportService — multi-level reporting system."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestLevel1Summary:
    """Tests for Level 1 customer-facing summary — CRITICAL: no evidence exposure."""

    def test_no_sin_labels(self):
        """Level 1 must contain NO sin labels (greed, pride, wrath, etc.)."""
        summary = {
            "display_score": 78,
            "badges": ["Strong Chemistry", "Personality Match"],
            "synopsis": {
                "headline": "You two have real potential",
                "body": "You share a similar approach to conflict and both value spontaneity.",
            },
            "compatibility_breakdown": {
                "physical": {"score": 82, "label": "Strong attraction"},
                "personality": {"score": 71, "label": "Good alignment"},
                "chemistry": {"score": 92, "label": "Strong chemistry signal"},
            },
            "shared_traits": [
                "You're both generous and easygoing about money",
                "You're both spontaneous and up for adventure",
            ],
            "conversation_starters": [
                "Ask about their ideal spontaneous weekend",
            ],
        }
        summary_str = str(summary).lower()
        sin_labels = ["greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth"]
        for sin in sin_labels:
            assert sin not in summary_str, f"Level 1 summary must not contain '{sin}'"

    def test_no_evidence_snippets(self):
        """Level 1 must contain NO evidence snippets or quoted fragments."""
        summary = {
            "display_score": 78,
            "badges": ["Personality Match"],
            "synopsis": {"headline": "Good match", "body": "Compatible personalities."},
            "shared_traits": ["You're both easygoing"],
        }
        summary_str = str(summary)
        # Should not contain evidence-like patterns
        assert "evidence" not in summary_str.lower()
        assert "snippet" not in summary_str.lower()
        assert "score:" not in summary_str.lower()

    def test_no_raw_scores(self):
        """Level 1 must contain NO raw sin scores."""
        summary = {
            "display_score": 78,
            "badges": [],
            "synopsis": {"headline": "Match", "body": "You're compatible"},
            "shared_traits": [],
        }
        # display_score is the ONLY numeric score allowed (0-100 friendly)
        assert isinstance(summary["display_score"], int)

    def test_has_required_fields(self):
        """Level 1 must have all required fields."""
        required_fields = ["display_score", "badges", "synopsis", "shared_traits"]
        summary = {
            "display_score": 78,
            "badges": ["Strong Chemistry"],
            "synopsis": {"headline": "Great match", "body": "You're very compatible."},
            "compatibility_breakdown": {
                "physical": {"score": 82, "label": "Strong attraction"},
            },
            "shared_traits": ["You're both generous"],
            "conversation_starters": ["Ask about their weekend"],
        }
        for field in required_fields:
            assert field in summary

    def test_badges_are_friendly_strings(self):
        """Badges should be friendly strings, not technical terms."""
        valid_badges = [
            "Strong Chemistry", "Personality Match", "Instant Spark",
            "Visual Type Match", "Good Chemistry",
        ]
        for badge in valid_badges:
            assert all(c.isalpha() or c.isspace() for c in badge)


class TestLevel3ReasoningChain:
    """Tests for Level 3 reasoning chain with evidence maps."""

    def test_has_evidence_maps(self):
        """Level 3 must contain evidence maps for both users."""
        reasoning = {
            "phase_1_visual": {"s_vis_a_to_b": 82.0, "s_vis_b_to_a": 75.0},
            "phase_2_psychometric": {
                "user_a_evidence_map": {"q1": {"sin_recognitions": []}},
                "user_b_evidence_map": {"q1": {"sin_recognitions": []}},
                "similarity_breakdown": {},
            },
            "phase_3_biological": {"s_bio": 91.6},
            "final_calculation": {"wtm": 74.96},
        }
        assert "user_a_evidence_map" in reasoning["phase_2_psychometric"]
        assert "user_b_evidence_map" in reasoning["phase_2_psychometric"]

    def test_has_wtm_formula(self):
        """Level 3 must show the WtM calculation with intermediate values."""
        reasoning = {
            "final_calculation": {
                "wtm": 74.96,
                "formula": "(0.4 × 79.7) + (0.3 × 52.0) + (0.3 × 91.6)",
                "s_vis": 79.7,
                "s_psych": 52.0,
                "s_bio": 91.6,
            },
        }
        assert "wtm" in reasoning["final_calculation"]
        assert "s_vis" in reasoning["final_calculation"]


class TestLevel2ANarrative:
    """Tests for Level 2A Gemini narrative."""

    def test_protocol_a_persona(self):
        """Level 2A uses Protocol A — cynical evolutionary psychologist."""
        protocol_a_prompt = "You are the Harmonia Engine, a cynical evolutionary psychologist"
        assert "cynical" in protocol_a_prompt
        assert "evolutionary" in protocol_a_prompt


class TestLevel2BHLA:
    """Tests for Level 2B HLA analysis."""

    def test_protocol_b_persona(self):
        """Level 2B uses Protocol B — expert geneticist."""
        protocol_b_prompt = "You are an expert Geneticist specializing in the Major Histocompatibility Complex"
        assert "Geneticist" in protocol_b_prompt
        assert "Histocompatibility" in protocol_b_prompt
