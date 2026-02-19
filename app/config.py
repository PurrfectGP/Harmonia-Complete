"""
Harmonia V3 — Application Configuration

Loads all configuration from environment variables (and an optional .env file)
using Pydantic Settings.  A cached ``get_settings()`` helper is provided so that
FastAPI dependency-injection (and any other call-site) always receives the same
validated instance without re-parsing the environment on every request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Harmonia V3 platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Gemini LLM
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str
    GEMINI_MODEL_PRIMARY: str = "gemini-3-pro-preview"
    GEMINI_MODEL_FALLBACK: str = "gemini-3-flash-preview"
    GEMINI_MODEL_STABLE: str = "gemini-2.5-flash"

    # ------------------------------------------------------------------ #
    # Anthropic (cluster generation)
    # ------------------------------------------------------------------ #
    ANTHROPIC_API_KEY: str = ""

    # ------------------------------------------------------------------ #
    # Database – Cloud SQL via Unix socket or private IP
    # ------------------------------------------------------------------ #
    DATABASE_URL: str
    DB_USER: str = "harmonia_user"
    DB_PASSWORD: str = ""
    DB_NAME: str = "harmonia"

    # ------------------------------------------------------------------ #
    # Redis – Memorystore for Redis
    # ------------------------------------------------------------------ #
    REDIS_URL: str

    # ------------------------------------------------------------------ #
    # Whole-the-Match (WtM) scoring weights
    # ------------------------------------------------------------------ #
    VISUAL_WEIGHT: float = 0.4       # S_vis weight
    PERSONALITY_WEIGHT: float = 0.3  # S_psych weight
    HLA_WEIGHT: float = 0.3          # S_bio weight

    # ------------------------------------------------------------------ #
    # Meta-learning / MetaFBP hyper-parameters
    # ------------------------------------------------------------------ #
    ADAPTATION_STRENGTH_LAMBDA: float = 0.01
    INNER_LOOP_STEPS: int = 5
    INNER_LR: float = 0.01
    FEATURE_DIM: int = 512
    METAFBP_EXTRACTOR_PATH: str = "models/universal_extractor.pth"
    METAFBP_GENERATOR_PATH: str = "models/meta_generator.pth"

    # ------------------------------------------------------------------ #
    # Security
    # ------------------------------------------------------------------ #
    FERNET_KEY: str  # Used for HLA encryption at rest

    # ------------------------------------------------------------------ #
    # Runtime environment
    # ------------------------------------------------------------------ #
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Google Cloud Platform
    # ------------------------------------------------------------------ #
    GCP_PROJECT_ID: str = ""
    GCP_REGION: str = "europe-west2"  # London
    GCS_BUCKET_NAME: str = ""
    CLOUD_SQL_INSTANCE_CONNECTION: str = ""
    CLOUD_SQL_USE_UNIX_SOCKET: bool = True
    GCS_MODEL_WEIGHTS_PREFIX: str = "models/"

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    ALLOWED_ORIGINS: str = "*"

    # ------------------------------------------------------------------ #
    # Sin weights for personality (PIIP) matching
    # ------------------------------------------------------------------ #
    SIN_WEIGHTS: Dict[str, float] = {
        "wrath": 1.5,
        "sloth": 1.3,
        "pride": 1.2,
        "lust": 1.0,
        "greed": 0.9,
        "gluttony": 0.8,
        "envy": 0.7,
    }

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def allowed_origins_list(self) -> list[str]:
        """Return ALLOWED_ORIGINS as a list split on commas."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @field_validator("VISUAL_WEIGHT", "PERSONALITY_WEIGHT", "HLA_WEIGHT")
    @classmethod
    def _weight_must_be_between_0_and_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Weight must be between 0 and 1, got {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated ``Settings`` instance.

    Using ``@lru_cache`` guarantees the .env file is read and validated
    exactly once per process lifetime.  Import this function anywhere you
    need access to configuration::

        from app.config import get_settings
        settings = get_settings()
    """
    return Settings()  # type: ignore[call-arg]
