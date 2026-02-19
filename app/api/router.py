"""
Harmonia V3 — Main API Router

Aggregates all sub-routers under a single prefix so that ``app.main``
can mount the entire API surface with one ``include_router`` call.
"""

from fastapi import APIRouter

from app.api import users, questionnaire, visual, hla, matching, reports
from app.api.admin import calibration

router = APIRouter()

router.include_router(users.router, prefix="/users", tags=["Users"])
router.include_router(questionnaire.router, prefix="/questionnaire", tags=["Questionnaire"])
router.include_router(visual.router, prefix="/visual", tags=["Visual Intelligence"])
router.include_router(hla.router, prefix="/hla", tags=["HLA Genetics"])
router.include_router(matching.router, prefix="/match", tags=["Matching"])
router.include_router(reports.router, prefix="/reports", tags=["Reports"])
router.include_router(calibration.router, prefix="/admin/calibration", tags=["Admin - Calibration"])
