"""Shared pytest fixtures for Harmonia V3 tests."""
import pytest
import uuid


@pytest.fixture
def sample_user_id():
    return str(uuid.uuid4())


@pytest.fixture
def sample_user_id_b():
    return str(uuid.uuid4())


@pytest.fixture
def sample_hla_alleles_a():
    """User A alleles from the spec worked example (Section 10.4)."""
    return {
        "HLA-A": ["A*01:01", "A*02:01"],
        "HLA-B": ["B*08:01", "B*07:02"],
        "HLA-DRB1": ["DRB1*15:01", "DRB1*03:01"],
    }


@pytest.fixture
def sample_hla_alleles_b():
    """User B alleles from the spec worked example (Section 10.4)."""
    return {
        "HLA-A": ["A*03:01", "A*24:02"],
        "HLA-B": ["B*44:02", "B*35:01"],
        "HLA-DRB1": ["DRB1*04:01", "DRB1*07:01"],
    }


@pytest.fixture
def sample_profile_a():
    """Personality profile A with sin scores and confidence."""
    return {
        "sins": {
            "greed": {"score": -1.8, "confidence": 0.78},
            "pride": {"score": 0.3, "confidence": 0.65},
            "lust": {"score": 2.5, "confidence": 0.82},
            "wrath": {"score": -2.1, "confidence": 0.88},
            "gluttony": {"score": 1.2, "confidence": 0.70},
            "envy": {"score": -0.8, "confidence": 0.75},
            "sloth": {"score": -1.5, "confidence": 0.72},
        },
        "quality_tier": "high",
        "quality_score": 84.2,
    }


@pytest.fixture
def sample_profile_b():
    """Personality profile B with sin scores and confidence."""
    return {
        "sins": {
            "greed": {"score": -2.2, "confidence": 0.80},
            "pride": {"score": -1.5, "confidence": 0.70},
            "lust": {"score": 3.1, "confidence": 0.85},
            "wrath": {"score": -1.8, "confidence": 0.90},
            "gluttony": {"score": -0.3, "confidence": 0.68},
            "envy": {"score": -1.2, "confidence": 0.77},
            "sloth": {"score": 1.8, "confidence": 0.74},
        },
        "quality_tier": "moderate",
        "quality_score": 72.5,
    }


def _sins_dict_to_scores_list(sins_dict):
    """Convert a {sin: {score, confidence, evidence}} dict to the scores list
    format expected by ProfileService.build_profile."""
    return [
        {"sin": sin, "score": data["score"], "confidence": data["confidence"], "evidence": data.get("evidence", "")}
        for sin, data in sins_dict.items()
    ]


@pytest.fixture
def sample_parsed_response_q1():
    """Parsed response for question 1 (group dinner check) from the spec."""
    sins_dict = {
        "greed": {"score": -2, "confidence": 0.78, "evidence": "Life's too short for that kind of pettiness"},
        "pride": {"score": 0, "confidence": 0.50, "evidence": ""},
        "lust": {"score": 0, "confidence": 0.45, "evidence": ""},
        "wrath": {"score": -3, "confidence": 0.88, "evidence": "I hate those awkward moments where everyone's trying to calculate their exact share"},
        "gluttony": {"score": 0, "confidence": 0.40, "evidence": ""},
        "envy": {"score": 0, "confidence": 0.42, "evidence": ""},
        "sloth": {"score": 1, "confidence": 0.72, "evidence": "just split it evenly to keep things simple"},
    }
    return {
        "question_number": 1,
        "question_text": "You're at a group dinner with friends. The bill arrives and it's not split evenly...",
        "response_text": (
            "I'd probably suggest we just split it evenly to keep things simple. "
            "I hate those awkward moments where everyone's trying to calculate their exact share. "
            "Life's too short for that kind of pettiness. If someone ordered way more, they can "
            "throw in extra if they want, but I'm not going to be the one calling them out."
        ),
        "word_count": 54,
        "scores": _sins_dict_to_scores_list(sins_dict),
        "sins": sins_dict,
    }


@pytest.fixture
def sample_parsed_responses(sample_parsed_response_q1):
    """All 6 parsed responses (Q1 from spec, Q2-Q6 with realistic values)."""
    responses = [sample_parsed_response_q1]
    for qn in range(2, 7):
        sins_dict = {
            "greed": {"score": -1.5, "confidence": 0.75, "evidence": "example evidence"},
            "pride": {"score": 0.5, "confidence": 0.60, "evidence": "example evidence"},
            "lust": {"score": 1.0, "confidence": 0.70, "evidence": "example evidence"},
            "wrath": {"score": -2.0, "confidence": 0.85, "evidence": "example evidence"},
            "gluttony": {"score": 0.3, "confidence": 0.55, "evidence": "example evidence"},
            "envy": {"score": -0.5, "confidence": 0.65, "evidence": "example evidence"},
            "sloth": {"score": 0.8, "confidence": 0.68, "evidence": "example evidence"},
        }
        responses.append({
            "question_number": qn,
            "question_text": f"Question {qn} text",
            "response_text": "A realistic response with enough words to pass the minimum word count validation for this question. " * 3,
            "word_count": 54,
            "scores": _sins_dict_to_scores_list(sins_dict),
            "sins": sins_dict,
        })
    return responses


@pytest.fixture
def sample_calibration_examples():
    """Sample calibration examples for testing few-shot retrieval."""
    return [
        {
            "id": str(uuid.uuid4()),
            "question_number": 1,
            "response_text": "I'd just throw in my card and cover it...",
            "sin": "wrath",
            "gemini_raw_score": -2.0,
            "gemini_raw_confidence": 0.85,
            "gemini_raw_evidence": "I'd rather just eat the cost",
            "validated_score": -2.0,
            "review_status": "approved",
            "review_notes": None,
        },
        {
            "id": str(uuid.uuid4()),
            "question_number": 1,
            "response_text": "Honestly I'd speak up. If someone ordered three cocktails...",
            "sin": "wrath",
            "gemini_raw_score": 3.0,
            "gemini_raw_confidence": 0.80,
            "gemini_raw_evidence": "I'd speak up",
            "validated_score": 2.0,
            "review_status": "corrected",
            "review_notes": "Sarcasm, not genuine anger — should be +2 not +3",
        },
    ]
