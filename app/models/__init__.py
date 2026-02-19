"""
Harmonia V3 — ORM model registry.

Importing every model here ensures that Alembic (and any other tool that
inspects ``Base.metadata``) discovers all tables automatically.
"""

from app.models.user import User
from app.models.questionnaire import QuestionnaireResponse, Question
from app.models.profile import PersonalityProfile
from app.models.visual import VisualPreference, VisualRating
from app.models.hla import HLAData
from app.models.match import Match, Swipe
from app.models.evidence import ParsingEvidence
from app.models.calibration import CalibrationExample

__all__ = [
    "User",
    "QuestionnaireResponse",
    "Question",
    "PersonalityProfile",
    "VisualPreference",
    "VisualRating",
    "HLAData",
    "Match",
    "Swipe",
    "ParsingEvidence",
    "CalibrationExample",
]
