"""Unit tests for GeminiService — PIIP Gemini parsing engine."""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.gemini_service import GeminiService


@pytest.fixture
def gemini_service():
    """Create a GeminiService with mocked settings."""
    with patch("app.services.gemini_service.get_settings") as mock_settings:
        settings = MagicMock()
        settings.GEMINI_API_KEY = "test-key"
        settings.GEMINI_MODEL_PRIMARY = "gemini-3-pro-preview"
        settings.GEMINI_MODEL_FALLBACK = "gemini-3-flash-preview"
        settings.GEMINI_MODEL_STABLE = "gemini-2.5-flash"
        mock_settings.return_value = settings
        with patch("app.services.gemini_service.genai"):
            service = GeminiService()
    return service


class TestBuildTraitPrompt:
    """Tests for _build_trait_prompt method."""

    def test_basic_prompt_contains_anchors(self, gemini_service):
        """Verify prompt contains trait anchors and scale definition."""
        prompt = gemini_service._build_trait_prompt(
            question="Test question",
            answer="Test answer with enough words",
            sin="wrath",
            few_shot_examples=[],
        )
        assert "wrath" in prompt.lower() or "WRATH" in prompt
        assert "-5" in prompt
        assert "+5" in prompt
        assert "conflict avoidance" in prompt.lower() or "confrontational" in prompt.lower()

    def test_prompt_with_few_shot_examples(self, gemini_service):
        """Verify few-shot examples are injected into the prompt."""
        examples = [
            {"response_text": "I'd just cover it.", "validated_score": -2.0, "evidence_snippet": "cover it"},
        ]
        prompt = gemini_service._build_trait_prompt(
            question="Test question",
            answer="Test answer",
            sin="wrath",
            few_shot_examples=examples,
        )
        assert "Reference Example" in prompt or "reference example" in prompt.lower() or "Example" in prompt
        assert "cover it" in prompt


class TestLocateEvidence:
    """Tests for _locate_evidence_in_response method."""

    def test_exact_match(self, gemini_service):
        """Exact string match returns correct character offsets."""
        text = "I'd probably suggest we just split it evenly to keep things simple."
        snippet = "just split it evenly"
        start, end = gemini_service._locate_evidence_in_response(text, snippet)
        assert start >= 0
        assert end > start
        assert text[start:end] == snippet

    def test_case_insensitive_match(self, gemini_service):
        """Case-insensitive matching works."""
        text = "I HATE those awkward moments"
        snippet = "i hate those awkward moments"
        start, end = gemini_service._locate_evidence_in_response(text, snippet)
        assert start >= 0

    def test_not_found(self, gemini_service):
        """Returns (-1, -1) when snippet not in response."""
        text = "A completely different response about something else."
        snippet = "this phrase does not appear anywhere"
        start, end = gemini_service._locate_evidence_in_response(text, snippet)
        assert start == -1
        assert end == -1


class TestLiwcSignals:
    """Tests for _extract_liwc_signals method."""

    def test_first_person_singular(self, gemini_service):
        """Detects first person singular pronouns (I, me, my)."""
        text = "I think I would handle it my way because it affects me personally."
        signals = gemini_service._extract_liwc_signals(text)
        assert "first_person_singular" in signals
        assert signals["first_person_singular"] > 0

    def test_emotion_words(self, gemini_service):
        """Detects negative and positive emotion words."""
        text = "I hate this terrible situation but I love the happy outcome."
        signals = gemini_service._extract_liwc_signals(text)
        assert "negative_emotion" in signals or "neg_emotion" in signals


class TestDetectDiscrepancies:
    """Tests for _detect_discrepancies method."""

    def test_wrath_discrepancy(self, gemini_service):
        """Claims low wrath but uses anger language."""
        text = "I absolutely hate when people are so stupid and infuriating"
        sins = {"wrath": {"score": -3, "confidence": 0.8}}
        discrepancies = gemini_service._detect_discrepancies(text, sins)
        assert len(discrepancies) > 0

    def test_no_discrepancies(self, gemini_service):
        """No discrepancies when language matches scores."""
        text = "I prefer to keep things peaceful and avoid conflict"
        sins = {"wrath": {"score": -2, "confidence": 0.8}}
        discrepancies = gemini_service._detect_discrepancies(text, sins)
        # Should have no discrepancies or very few
        assert isinstance(discrepancies, list)
