"""
Harmonia V3 — Visual Intelligence Service

Manages the visual attractiveness pipeline:

  1. **Calibration** — Accepts user ratings on reference images, extracts
     facial traits, runs MetaFBP inner-loop adaptation, and caches the
     personalised model weights in Redis.

  2. **Scoring** — Loads adapted weights from Redis, runs inference on a
     target image, combines the MetaFBP component with trait-match
     components to produce S_vis.

  3. **Cache management** — Invalidates stale adapted weights and triggers
     background re-adaptation when the user provides new ratings.

Scoring formula:
    S_vis = (MetaFBP_Component * 0.6) + (T_match_positive * 0.25) + (T_match_negative * 0.15)

Where:
    MetaFBP_Component = (raw_score - 1) * 25   (maps [1,5] -> [0,100])
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.visual import VisualPreference, VisualRating

logger = structlog.get_logger("harmonia.visual_service")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_REDIS_KEY_PREFIX = "harmonia:visual:adapted_weights:"
_REDIS_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

# S_vis component weights (from spec)
_METAFBP_WEIGHT = 0.60
_T_POSITIVE_WEIGHT = 0.25
_T_NEGATIVE_WEIGHT = 0.15

# MetaFBP raw score mapping: [1, 5] -> [0, 100]
_METAFBP_RAW_MIN = 1.0
_METAFBP_RAW_MAX = 5.0
_METAFBP_SCALE = 25.0  # (raw - 1) * 25


class VisualService:
    """Orchestrates the visual attractiveness scoring pipeline.

    The service lazily initialises the MetaFBP inference engine and
    trait extractor so that the application can start even if model
    weights have not been downloaded yet.
    """

    def __init__(self) -> None:
        """Initialise the service with lazy-loaded ML components.

        Both ``MetaFBPInference`` and ``TraitExtractor`` are instantiated
        on first use rather than at construction time, because the model
        weight files may not be available immediately (they are downloaded
        from GCS during the lifespan startup phase).
        """
        self._metafbp: Any | None = None
        self._trait_extractor: Any | None = None
        self._redis: Any | None = None

    # ── Lazy loaders ──────────────────────────────────────────────────────

    def _get_metafbp(self) -> Any:
        """Return the MetaFBPInference instance, creating it on first call."""
        if self._metafbp is None:
            try:
                from app.ml.metafbp import MetaFBPInference

                settings = get_settings()
                self._metafbp = MetaFBPInference(
                    extractor_path=settings.METAFBP_EXTRACTOR_PATH,
                    generator_path=settings.METAFBP_GENERATOR_PATH,
                    feature_dim=settings.FEATURE_DIM,
                    inner_lr=settings.INNER_LR,
                    inner_steps=settings.INNER_LOOP_STEPS,
                    adaptation_lambda=settings.ADAPTATION_STRENGTH_LAMBDA,
                )
                logger.info("metafbp_initialised")
            except Exception:
                logger.exception("metafbp_init_failed")
                raise RuntimeError(
                    "MetaFBP model weights not available. "
                    "Ensure model files are present before scoring."
                )
        return self._metafbp

    def _get_trait_extractor(self) -> Any:
        """Return the TraitExtractor instance, creating it on first call."""
        if self._trait_extractor is None:
            try:
                from app.ml.trait_extraction import TraitExtractor

                self._trait_extractor = TraitExtractor()
                logger.info("trait_extractor_initialised")
            except Exception:
                logger.exception("trait_extractor_init_failed")
                raise RuntimeError(
                    "TraitExtractor could not be initialised. "
                    "Check dependencies and model files."
                )
        return self._trait_extractor

    async def _get_redis(self) -> Any:
        """Return an async Redis client, creating it on first call."""
        if self._redis is None:
            import redis.asyncio as aioredis

            settings = get_settings()
            self._redis = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            logger.info("visual_service_redis_connected")
        return self._redis

    # ── Public API ────────────────────────────────────────────────────────

    async def calibrate_user(
        self,
        user_id: str,
        ratings: list[dict],
        db_session: AsyncSession,
    ) -> dict:
        """Calibrate a user's visual preference model from their ratings.

        Steps:
          1. Persist each rating to the ``visual_ratings`` table.
          2. Run trait extraction on the rated images.
          3. Run MetaFBP inner-loop adaptation using the support set.
          4. Cache adapted weights in Redis.
          5. Store the derived trait model in ``visual_preferences``.

        Parameters
        ----------
        user_id:
            UUID of the user being calibrated.
        ratings:
            List of dicts, each containing::

                {
                    "image_id": str,
                    "image_path": str,
                    "rating": int,   # 1-5
                }

        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            Calibration summary including trait model and rating stats.
        """
        log = logger.bind(user_id=user_id, n_ratings=len(ratings))
        log.info("calibration_start")

        # ── Step 1: Persist ratings ────────────────────────────────────
        for r in ratings:
            rating_record = VisualRating(
                user_id=user_id,
                image_id=r["image_id"],
                image_path=r["image_path"],
                rating=r["rating"],
            )
            db_session.add(rating_record)

        await db_session.flush()
        log.info("ratings_persisted", count=len(ratings))

        # ── Step 2: Trait extraction ───────────────────────────────────
        trait_extractor = self._get_trait_extractor()

        trait_results: list[dict] = []
        for r in ratings:
            try:
                traits = trait_extractor.extract(r["image_path"])
                trait_results.append({
                    "image_id": r["image_id"],
                    "rating": r["rating"],
                    "traits": traits,
                })
            except Exception:
                log.warning(
                    "trait_extraction_failed",
                    image_id=r["image_id"],
                    image_path=r["image_path"],
                )
                trait_results.append({
                    "image_id": r["image_id"],
                    "rating": r["rating"],
                    "traits": {},
                })

        # Derive preference model from ratings + traits
        trait_model = self._derive_trait_model(trait_results)
        log.info(
            "trait_model_derived",
            mandatory=len(trait_model.get("mandatory_traits", {})),
            preferred=len(trait_model.get("preferred_traits", {})),
            aversion=len(trait_model.get("aversion_traits", {})),
        )

        # ── Step 3: MetaFBP adaptation ────────────────────────────────
        metafbp = self._get_metafbp()

        # Build support set: list of (image_path, rating) tuples
        support_set = [
            (r["image_path"], r["rating"]) for r in ratings
        ]
        adapted_state = metafbp.adapt(support_set)
        log.info("metafbp_adaptation_complete")

        # ── Step 4: Cache adapted weights in Redis ────────────────────
        await self._cache_adapted_weights(user_id, adapted_state)

        # ── Step 5: Store trait model in visual_preferences ───────────
        redis_key = f"{_REDIS_KEY_PREFIX}{user_id}"

        # Compute support-set statistics
        rating_values = [r["rating"] for r in ratings]
        support_stats = {
            "n_images": len(ratings),
            "mean_rating": round(sum(rating_values) / max(1, len(rating_values)), 4),
            "rating_distribution": {
                str(i): rating_values.count(i) for i in range(1, 6)
            },
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Upsert visual preference record
        stmt = select(VisualPreference).where(
            VisualPreference.user_id == user_id
        )
        result = await db_session.execute(stmt)
        existing_pref = result.scalar_one_or_none()

        if existing_pref is not None:
            existing_pref.support_set_stats = support_stats
            existing_pref.mandatory_traits = trait_model.get("mandatory_traits")
            existing_pref.preferred_traits = trait_model.get("preferred_traits")
            existing_pref.aversion_traits = trait_model.get("aversion_traits")
            existing_pref.adapted_weights_key = redis_key
            log.info("visual_preference_updated")
        else:
            new_pref = VisualPreference(
                user_id=user_id,
                support_set_stats=support_stats,
                mandatory_traits=trait_model.get("mandatory_traits"),
                preferred_traits=trait_model.get("preferred_traits"),
                aversion_traits=trait_model.get("aversion_traits"),
                adapted_weights_key=redis_key,
            )
            db_session.add(new_pref)
            log.info("visual_preference_created")

        await db_session.flush()

        log.info("calibration_complete")

        return {
            "user_id": user_id,
            "ratings_stored": len(ratings),
            "support_set_stats": support_stats,
            "trait_model": trait_model,
            "adapted_weights_cached": True,
            "redis_key": redis_key,
        }

    async def score_target(
        self,
        user_id: str,
        target_image_path: str,
        db_session: AsyncSession,
    ) -> dict:
        """Score a target image for a calibrated user.

        Steps:
          1. Load cached adapted weights from Redis.
          2. Run MetaFBP inference on the target image.
          3. Detect traits on the target image.
          4. Load user's trait preferences from the database.
          5. Calculate trait match components.
          6. Combine into final S_vis score.

        Formula:
            S_vis = (MetaFBP_Component * 0.6) + (T_match_positive * 0.25) + (T_match_negative * 0.15)
            MetaFBP_Component = (raw_score - 1) * 25   (maps [1,5] -> [0,100])

        Parameters
        ----------
        user_id:
            UUID of the user whose adapted model to use.
        target_image_path:
            Path to the target image to score.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            Scoring result with final S_vis and component breakdown.
        """
        log = logger.bind(user_id=user_id, target=target_image_path)
        log.info("scoring_start")

        # ── Step 1: Load adapted weights ──────────────────────────────
        adapted_state = await self._load_cached_weights(user_id)

        if adapted_state is None:
            log.warning("adapted_weights_not_found")
            return {
                "user_id": user_id,
                "target_image_path": target_image_path,
                "error": "not_calibrated",
                "message": "User has not been calibrated. Call calibrate_user first.",
            }

        # ── Step 2: MetaFBP inference ─────────────────────────────────
        metafbp = self._get_metafbp()
        raw_score = metafbp.score(target_image_path, adapted_state)

        # Clamp to valid range [1, 5]
        raw_score = max(_METAFBP_RAW_MIN, min(_METAFBP_RAW_MAX, float(raw_score)))

        # Map to [0, 100]
        metafbp_component = (raw_score - _METAFBP_RAW_MIN) * _METAFBP_SCALE

        log.info(
            "metafbp_scored",
            raw_score=round(raw_score, 4),
            component=round(metafbp_component, 4),
        )

        # ── Step 3: Trait detection on target ─────────────────────────
        trait_extractor = self._get_trait_extractor()
        try:
            target_traits = trait_extractor.extract(target_image_path)
        except Exception:
            log.warning("target_trait_extraction_failed")
            target_traits = {}

        # ── Step 4: Load user preferences ─────────────────────────────
        stmt = select(VisualPreference).where(
            VisualPreference.user_id == user_id
        )
        result = await db_session.execute(stmt)
        pref_record = result.scalar_one_or_none()

        if pref_record is not None:
            user_prefs = {
                "mandatory_traits": pref_record.mandatory_traits or {},
                "preferred_traits": pref_record.preferred_traits or {},
                "aversion_traits": pref_record.aversion_traits or {},
            }
        else:
            user_prefs = {
                "mandatory_traits": {},
                "preferred_traits": {},
                "aversion_traits": {},
            }

        # ── Step 5: Trait match ───────────────────────────────────────
        t_match_positive, t_match_negative = self._calculate_trait_match(
            user_prefs, target_traits
        )

        # ── Step 6: Combine into S_vis ────────────────────────────────
        s_vis = (
            (metafbp_component * _METAFBP_WEIGHT)
            + (t_match_positive * _T_POSITIVE_WEIGHT)
            + (t_match_negative * _T_NEGATIVE_WEIGHT)
        )

        # Clamp to [0, 100]
        s_vis = max(0.0, min(100.0, s_vis))

        log.info(
            "scoring_complete",
            s_vis=round(s_vis, 4),
            metafbp_component=round(metafbp_component, 4),
            t_positive=round(t_match_positive, 4),
            t_negative=round(t_match_negative, 4),
        )

        return {
            "user_id": user_id,
            "target_image_path": target_image_path,
            "s_vis": round(s_vis, 4),
            "components": {
                "metafbp_raw_score": round(raw_score, 4),
                "metafbp_component": round(metafbp_component, 4),
                "t_match_positive": round(t_match_positive, 4),
                "t_match_negative": round(t_match_negative, 4),
            },
            "weights": {
                "metafbp": _METAFBP_WEIGHT,
                "t_positive": _T_POSITIVE_WEIGHT,
                "t_negative": _T_NEGATIVE_WEIGHT,
            },
            "target_traits": target_traits,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

    async def invalidate_cache(self, user_id: str) -> None:
        """Invalidate cached adapted weights and trigger re-adaptation.

        Removes the Redis cache key for the user's adapted model weights.
        After invalidation, the next call to ``score_target`` will fail
        with ``not_calibrated`` until ``calibrate_user`` is called again
        (or background re-adaptation completes).

        Parameters
        ----------
        user_id:
            UUID of the user whose cache to invalidate.
        """
        log = logger.bind(user_id=user_id)
        log.info("cache_invalidation_start")

        redis = await self._get_redis()
        redis_key = f"{_REDIS_KEY_PREFIX}{user_id}"

        deleted = await redis.delete(redis_key)

        if deleted:
            log.info("cache_invalidated", redis_key=redis_key)
        else:
            log.info("cache_key_not_found", redis_key=redis_key)

        # Trigger background re-adaptation
        # In production this would enqueue a task to a worker queue
        # (e.g. Cloud Tasks, Celery). For now we log the intent.
        log.info(
            "background_readaptation_triggered",
            note="Async re-adaptation would be enqueued here",
        )

    # ── Trait match calculation ────────────────────────────────────────────

    def _calculate_trait_match(
        self,
        user_prefs: dict,
        target_traits: dict,
    ) -> tuple[float, float]:
        """Calculate positive and negative trait match scores.

        Parameters
        ----------
        user_prefs:
            Dict with keys ``mandatory_traits``, ``preferred_traits``,
            and ``aversion_traits``.  Each maps trait names to weight
            floats (0-1).
        target_traits:
            Dict mapping trait names to detected confidence values (0-1)
            for the target image.

        Returns
        -------
        tuple[float, float]
            ``(T_match_positive, T_match_negative)`` both in [0, 100].

            - **T_match_positive**: Weighted match score of preferred and
              mandatory traits that are present in the target.  Mandatory
              traits carry double weight.
            - **T_match_negative**: Inverse of aversion traits present.
              100 means no aversion traits detected; 0 means all aversion
              traits are strongly present.
        """
        mandatory = user_prefs.get("mandatory_traits", {})
        preferred = user_prefs.get("preferred_traits", {})
        aversion = user_prefs.get("aversion_traits", {})

        # ── Positive component ────────────────────────────────────────
        # Mandatory traits count 2x, preferred count 1x
        positive_entries: list[tuple[float, float]] = []  # (weight, match)

        for trait, importance in mandatory.items():
            weight = float(importance) * 2.0  # double weight for mandatory
            detected = float(target_traits.get(trait, 0.0))
            positive_entries.append((weight, detected))

        for trait, importance in preferred.items():
            if trait in mandatory:
                continue  # already counted as mandatory
            weight = float(importance)
            detected = float(target_traits.get(trait, 0.0))
            positive_entries.append((weight, detected))

        if positive_entries:
            total_weight = sum(w for w, _ in positive_entries)
            if total_weight > 0:
                weighted_match = sum(w * m for w, m in positive_entries)
                t_positive = (weighted_match / total_weight) * 100.0
            else:
                t_positive = 50.0  # neutral when no weights
        else:
            # No trait preferences defined — assign neutral score
            t_positive = 50.0

        # ── Negative component ────────────────────────────────────────
        # Inverse of aversion trait presence: 100 = no aversions, 0 = all present
        if aversion:
            aversion_entries: list[tuple[float, float]] = []
            for trait, severity in aversion.items():
                weight = float(severity)
                detected = float(target_traits.get(trait, 0.0))
                aversion_entries.append((weight, detected))

            total_aversion_weight = sum(w for w, _ in aversion_entries)
            if total_aversion_weight > 0:
                weighted_aversion = sum(w * d for w, d in aversion_entries)
                aversion_ratio = weighted_aversion / total_aversion_weight
                t_negative = (1.0 - aversion_ratio) * 100.0
            else:
                t_negative = 100.0  # no aversion weight means no penalty
        else:
            # No aversion traits defined — full score (no penalty)
            t_negative = 100.0

        # Clamp both to [0, 100]
        t_positive = max(0.0, min(100.0, t_positive))
        t_negative = max(0.0, min(100.0, t_negative))

        return t_positive, t_negative

    # ── Redis cache helpers ───────────────────────────────────────────────

    async def _cache_adapted_weights(
        self,
        user_id: str,
        adapted_state: dict,
    ) -> None:
        """Serialize and cache adapted model weights in Redis.

        Parameters
        ----------
        user_id:
            UUID of the user.
        adapted_state:
            The adapted model state dict from MetaFBP adaptation.
            Must be JSON-serialisable (tensors should be converted to
            lists by the MetaFBP module before returning).
        """
        log = logger.bind(user_id=user_id)

        redis = await self._get_redis()
        redis_key = f"{_REDIS_KEY_PREFIX}{user_id}"

        try:
            serialised = json.dumps(adapted_state)
            await redis.setex(redis_key, _REDIS_TTL_SECONDS, serialised)
            log.info(
                "adapted_weights_cached",
                redis_key=redis_key,
                ttl_seconds=_REDIS_TTL_SECONDS,
                payload_bytes=len(serialised),
            )
        except Exception:
            log.exception("adapted_weights_cache_failed", redis_key=redis_key)
            raise

    async def _load_cached_weights(
        self,
        user_id: str,
    ) -> dict | None:
        """Load adapted model weights from Redis.

        Parameters
        ----------
        user_id:
            UUID of the user.

        Returns
        -------
        dict or None
            The deserialised adapted state, or None if the key does not
            exist or has expired.
        """
        log = logger.bind(user_id=user_id)

        redis = await self._get_redis()
        redis_key = f"{_REDIS_KEY_PREFIX}{user_id}"

        try:
            raw = await redis.get(redis_key)
            if raw is None:
                log.debug("adapted_weights_cache_miss", redis_key=redis_key)
                return None

            adapted_state = json.loads(raw)
            log.debug(
                "adapted_weights_cache_hit",
                redis_key=redis_key,
                payload_bytes=len(raw),
            )
            return adapted_state
        except Exception:
            log.exception("adapted_weights_load_failed", redis_key=redis_key)
            return None

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _derive_trait_model(trait_results: list[dict]) -> dict:
        """Derive mandatory, preferred, and aversion trait preferences from
        rated image traits.

        Strategy:
          - Traits consistently present in highly-rated images (4-5) and
            absent in low-rated images (1-2) become **mandatory**.
          - Traits more common in high-rated images become **preferred**.
          - Traits more common in low-rated images become **aversion**.

        Parameters
        ----------
        trait_results:
            List of dicts with ``rating`` (int) and ``traits`` (dict)
            per rated image.

        Returns
        -------
        dict
            ``{"mandatory_traits": {...}, "preferred_traits": {...},
              "aversion_traits": {...}}``
        """
        # Collect trait presence by rating bucket
        high_traits: dict[str, list[float]] = {}  # ratings 4-5
        low_traits: dict[str, list[float]] = {}   # ratings 1-2
        all_traits: dict[str, list[float]] = {}    # all ratings

        for entry in trait_results:
            rating = entry.get("rating", 3)
            traits = entry.get("traits", {})

            for trait_name, confidence in traits.items():
                conf_val = float(confidence)
                all_traits.setdefault(trait_name, []).append(conf_val)

                if rating >= 4:
                    high_traits.setdefault(trait_name, []).append(conf_val)
                elif rating <= 2:
                    low_traits.setdefault(trait_name, []).append(conf_val)

        mandatory_traits: dict[str, float] = {}
        preferred_traits: dict[str, float] = {}
        aversion_traits: dict[str, float] = {}

        # Count high-rated and low-rated images for normalisation
        n_high = sum(1 for e in trait_results if e.get("rating", 3) >= 4)
        n_low = sum(1 for e in trait_results if e.get("rating", 3) <= 2)

        for trait_name in all_traits:
            high_mean = (
                sum(high_traits.get(trait_name, [])) / max(1, len(high_traits.get(trait_name, [])))
                if trait_name in high_traits
                else 0.0
            )
            low_mean = (
                sum(low_traits.get(trait_name, [])) / max(1, len(low_traits.get(trait_name, [])))
                if trait_name in low_traits
                else 0.0
            )

            high_freq = len(high_traits.get(trait_name, [])) / max(1, n_high)
            low_freq = len(low_traits.get(trait_name, [])) / max(1, n_low)

            # Mandatory: high freq in liked images AND high mean AND low
            # freq in disliked images
            if high_freq >= 0.8 and high_mean >= 0.7 and low_freq <= 0.2:
                mandatory_traits[trait_name] = round(high_mean, 4)
            # Preferred: notably more common in high-rated
            elif high_freq >= 0.5 and high_mean > low_mean + 0.2:
                preferred_traits[trait_name] = round(high_mean - low_mean, 4)
            # Aversion: notably more common in low-rated
            elif low_freq >= 0.5 and low_mean > high_mean + 0.2:
                aversion_traits[trait_name] = round(low_mean - high_mean, 4)

        return {
            "mandatory_traits": mandatory_traits,
            "preferred_traits": preferred_traits,
            "aversion_traits": aversion_traits,
        }
