"""
Harmonia V3 — CalibrationService: Ground-Truth Calibration Example Lifecycle

Manages the full lifecycle of calibration examples used for few-shot injection
into the Gemini parsing prompts.  The workflow is:

  1. **Ingest** — examples are created from real or synthetic parsing runs
     with ``review_status="pending"``.
  2. **Review** — an admin approves, corrects, or rejects each example.
  3. **Retrieve** — the GeminiService fetches top-N validated examples per
     (question_number, sin) cell for few-shot prompt injection.
  4. **Stats / Metrics** — coverage maps, correction drift, and effectiveness
     reporting for the calibration programme.

The target is 5 validated examples per cell in a 6-question x 7-sin matrix
(42 cells, 210 examples total for full coverage).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import case, distinct, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.calibration import CalibrationExample

logger = structlog.get_logger("harmonia.calibration_service")

# ── Constants ────────────────────────────────────────────────────────────────

SIN_NAMES: list[str] = [
    "greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth",
]

QUESTION_NUMBERS: list[int] = [1, 2, 3, 4, 5, 6]

TARGET_EXAMPLES_PER_CELL: int = 5

VALID_REVIEW_STATUSES: set[str] = {"pending", "approved", "corrected", "rejected"}

VALID_REVIEW_ACTIONS: set[str] = {"approve", "correct", "reject"}


class CalibrationService:
    """Orchestrates the calibration-example lifecycle: ingestion, admin
    review, few-shot retrieval, and programme-wide statistics.

    All public methods accept an optional ``db_session`` parameter.  When
    ``None`` is passed, the method opens (and commits/closes) its own
    session via ``async_session_factory``.  When an existing session is
    provided, the caller is responsible for commit/rollback.
    """

    # ══════════════════════════════════════════════════════════════════════
    # 1. ingest_from_parsing — create a new pending calibration example
    # ══════════════════════════════════════════════════════════════════════

    async def ingest_from_parsing(
        self,
        user_id: str,
        question_number: int,
        response_text: str,
        sin: str,
        gemini_score: float,
        gemini_confidence: float,
        gemini_evidence: str,
        source_profile_id: str = None,
        db_session: AsyncSession | None = None,
    ) -> str:
        """Create a new calibration example with ``review_status='pending'``.

        Parameters
        ----------
        user_id:
            The user whose response is being calibrated.
        question_number:
            Ordinal question position (1-6).
        response_text:
            The user's free-text answer.
        sin:
            The sin dimension this example targets.
        gemini_score:
            The raw score produced by Gemini for this (question, sin) pair.
        gemini_confidence:
            The raw confidence from Gemini (0-1).
        gemini_evidence:
            The evidence snippet extracted by Gemini.
        source_profile_id:
            Optional UUID of the personality profile this example came from.
        db_session:
            An active async SQLAlchemy session, or ``None`` to auto-manage.

        Returns
        -------
        str
            The UUID of the newly created calibration example.
        """
        log = logger.bind(
            user_id=user_id,
            question_number=question_number,
            sin=sin,
        )
        log.info(
            "ingest_start",
            gemini_score=gemini_score,
            gemini_confidence=gemini_confidence,
        )

        example_id = uuid.uuid4()

        example = CalibrationExample(
            id=example_id,
            question_number=question_number,
            response_text=response_text,
            sin=sin,
            gemini_raw_score=gemini_score,
            gemini_raw_confidence=gemini_confidence,
            gemini_raw_evidence=gemini_evidence,
            review_status="pending",
            source_profile_id=(
                uuid.UUID(source_profile_id)
                if source_profile_id
                else None
            ),
        )

        if db_session is not None:
            db_session.add(example)
            await db_session.flush()
        else:
            async with async_session_factory() as session:
                session.add(example)
                await session.commit()

        log.info("ingest_complete", example_id=str(example_id))
        return str(example_id)

    # ══════════════════════════════════════════════════════════════════════
    # 2. get_review_queue — list pending examples for admin review
    # ══════════════════════════════════════════════════════════════════════

    async def get_review_queue(
        self,
        filters: dict | None = None,
        db_session: AsyncSession | None = None,
    ) -> list[dict]:
        """Return calibration examples matching the given filters, sorted by
        creation date (oldest first).

        Parameters
        ----------
        filters:
            Optional dict with any of: ``status`` (str or list[str]),
            ``question_number`` (int), ``sin`` (str), ``limit`` (int,
            default 50), ``offset`` (int, default 0).
        db_session:
            An active async SQLAlchemy session, or ``None`` to auto-manage.

        Returns
        -------
        list[dict]
            Each dict contains all columns from the calibration_examples
            table, serialised as JSON-safe values.
        """
        filters = filters or {}
        log = logger.bind(filters=filters)
        log.info("get_review_queue_start")

        status = filters.get("status", "pending")
        question_number = filters.get("question_number")
        sin = filters.get("sin")
        limit = filters.get("limit", 50)
        offset = filters.get("offset", 0)

        stmt = select(CalibrationExample).order_by(
            CalibrationExample.created_at.asc()
        )

        # Status filter — single string or list of statuses
        if isinstance(status, list):
            stmt = stmt.where(CalibrationExample.review_status.in_(status))
        elif status:
            stmt = stmt.where(CalibrationExample.review_status == status)

        if question_number is not None:
            stmt = stmt.where(
                CalibrationExample.question_number == question_number
            )

        if sin is not None:
            stmt = stmt.where(CalibrationExample.sin == sin)

        stmt = stmt.offset(offset).limit(limit)

        async def _execute(session: AsyncSession) -> list[dict]:
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_dict(row) for row in rows]

        if db_session is not None:
            results = await _execute(db_session)
        else:
            async with async_session_factory() as session:
                results = await _execute(session)

        log.info("get_review_queue_complete", count=len(results))
        return results

    # ══════════════════════════════════════════════════════════════════════
    # 3. review_example — approve / correct / reject a single example
    # ══════════════════════════════════════════════════════════════════════

    async def review_example(
        self,
        example_id: str,
        action: str,
        validated_score: float | None = None,
        validated_by: str | None = None,
        review_notes: str | None = None,
        db_session: AsyncSession | None = None,
    ) -> None:
        """Apply an admin review action to a calibration example.

        Actions
        -------
        - ``"approve"``: Accept the Gemini raw score as-is.
          ``validated_score`` is set to ``gemini_raw_score``.
        - ``"correct"``: Override the score.  The caller **must** supply
          ``validated_score`` and optionally ``review_notes``.
        - ``"reject"``: Exclude the example from the few-shot pool.

        Parameters
        ----------
        example_id:
            UUID of the calibration example.
        action:
            One of ``"approve"``, ``"correct"``, ``"reject"``.
        validated_score:
            Required when ``action="correct"``.
        validated_by:
            Identifier for the reviewing admin.
        review_notes:
            Optional notes explaining the correction or rejection.
        db_session:
            An active async SQLAlchemy session, or ``None`` to auto-manage.

        Raises
        ------
        ValueError
            If the action is invalid or required fields are missing.
        LookupError
            If no example is found with the given ID.
        """
        if action not in VALID_REVIEW_ACTIONS:
            raise ValueError(
                f"Invalid action {action!r}. Must be one of: "
                f"{', '.join(sorted(VALID_REVIEW_ACTIONS))}"
            )

        if action == "correct" and validated_score is None:
            raise ValueError(
                "validated_score is required when action='correct'"
            )

        log = logger.bind(
            example_id=example_id,
            action=action,
            validated_by=validated_by,
        )
        log.info("review_start")

        async def _apply(session: AsyncSession) -> None:
            stmt = select(CalibrationExample).where(
                CalibrationExample.id == uuid.UUID(example_id)
            )
            result = await session.execute(stmt)
            example = result.scalar_one_or_none()

            if example is None:
                raise LookupError(
                    f"Calibration example {example_id} not found"
                )

            now = datetime.now(timezone.utc)

            if action == "approve":
                example.review_status = "approved"
                example.validated_score = example.gemini_raw_score
                example.validated_by = validated_by
                example.validated_at = now
                example.review_notes = review_notes

            elif action == "correct":
                example.review_status = "corrected"
                example.validated_score = validated_score
                example.validated_by = validated_by
                example.validated_at = now
                example.review_notes = review_notes

            elif action == "reject":
                example.review_status = "rejected"
                example.validated_by = validated_by
                example.validated_at = now
                example.review_notes = review_notes

            await session.flush()

        if db_session is not None:
            await _apply(db_session)
        else:
            async with async_session_factory() as session:
                await _apply(session)
                await session.commit()

        log.info("review_complete", new_status=action)

    # ══════════════════════════════════════════════════════════════════════
    # 4. get_examples — retrieve top-N validated examples for few-shot
    # ══════════════════════════════════════════════════════════════════════

    async def get_examples(
        self,
        question_number: int,
        sin: str,
        n: int = 3,
        db_session: AsyncSession | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Retrieve the top *n* validated calibration examples for a given
        (question_number, sin) cell, suitable for few-shot prompt injection.

        Selection priority
        ------------------
        1. **Corrected** examples first (human-validated overrides are the
           highest-quality anchors).
        2. **Approved** examples sorted by descending Gemini confidence.
        3. **Score diversity** — from the eligible pool, select examples
           that maximise the spread of ``validated_score`` so the LLM sees
           both low and high anchors.

        Parameters
        ----------
        question_number:
            The ordinal question position (1-6).
        sin:
            The sin dimension.
        n:
            Maximum number of examples to return (default 3).
        db_session:
            An active async SQLAlchemy session, or ``None`` to auto-manage.

        Returns
        -------
        list[dict]
            Each dict contains ``response_text``, ``validated_score``,
            ``evidence_snippet``, and ``review_notes``.
        """
        log = logger.bind(
            question_number=question_number,
            sin=sin,
            n=n,
        )
        log.info("get_examples_start")

        # Build a query that fetches all approved/corrected examples for
        # this cell, ordered: corrected first, then approved by confidence.
        stmt = (
            select(CalibrationExample)
            .where(
                CalibrationExample.question_number == question_number,
                CalibrationExample.sin == sin,
                CalibrationExample.review_status.in_(["approved", "corrected"]),
            )
            .order_by(
                # Corrected examples first (sort key: 0 for corrected, 1 for approved)
                case(
                    (CalibrationExample.review_status == "corrected", 0),
                    else_=1,
                ),
                # Then by descending confidence
                CalibrationExample.gemini_raw_confidence.desc(),
            )
        )

        async def _execute(session: AsyncSession) -> list[dict]:
            result = await session.execute(stmt)
            rows = result.scalars().all()

            if not rows:
                return []

            # Apply score diversity selection: pick examples that maximise
            # the spread of validated_score values.
            selected = self._select_diverse_examples(rows, n)

            return [
                {
                    "response_text": row.response_text,
                    "validated_score": row.validated_score,
                    "evidence_snippet": row.gemini_raw_evidence,
                    "review_notes": row.review_notes,
                }
                for row in selected
            ]

        if db_session is not None:
            results = await _execute(db_session)
        else:
            async with async_session_factory() as session:
                results = await _execute(session)

        log.info("get_examples_complete", count=len(results))
        return results

    # ══════════════════════════════════════════════════════════════════════
    # 5. get_calibration_stats — programme-wide statistics
    # ══════════════════════════════════════════════════════════════════════

    async def get_calibration_stats(
        self,
        db_session: AsyncSession | None = None,
    ) -> dict:
        """Return aggregate statistics for the calibration programme.

        Returns
        -------
        dict
            Contains:
            - ``total``, ``pending``, ``approved``, ``corrected``,
              ``rejected``: overall counts.
            - ``by_question``: dict keyed by question_number with status
              breakdowns.
            - ``by_sin``: dict keyed by sin name with status breakdowns.
            - ``avg_correction_magnitude``: mean absolute difference
              between ``gemini_raw_score`` and ``validated_score`` for
              corrected examples.
            - ``coverage_map``: 6x7 matrix (question x sin) showing
              the count of approved+corrected examples per cell, with
              a ``target`` of 5 per cell.
        """
        log = logger.bind()
        log.info("get_calibration_stats_start")

        async def _execute(session: AsyncSession) -> dict:
            # ── Overall status counts ─────────────────────────────────
            status_stmt = select(
                CalibrationExample.review_status,
                func.count().label("cnt"),
            ).group_by(CalibrationExample.review_status)

            status_result = await session.execute(status_stmt)
            status_counts = {row[0]: row[1] for row in status_result.all()}

            total = sum(status_counts.values())
            pending = status_counts.get("pending", 0)
            approved = status_counts.get("approved", 0)
            corrected = status_counts.get("corrected", 0)
            rejected = status_counts.get("rejected", 0)

            # ── Breakdown by question ─────────────────────────────────
            by_q_stmt = select(
                CalibrationExample.question_number,
                CalibrationExample.review_status,
                func.count().label("cnt"),
            ).group_by(
                CalibrationExample.question_number,
                CalibrationExample.review_status,
            )

            by_q_result = await session.execute(by_q_stmt)
            by_question: dict[int, dict[str, int]] = {}
            for qn, status, cnt in by_q_result.all():
                by_question.setdefault(qn, {})[status] = cnt

            # ── Breakdown by sin ──────────────────────────────────────
            by_s_stmt = select(
                CalibrationExample.sin,
                CalibrationExample.review_status,
                func.count().label("cnt"),
            ).group_by(
                CalibrationExample.sin,
                CalibrationExample.review_status,
            )

            by_s_result = await session.execute(by_s_stmt)
            by_sin: dict[str, dict[str, int]] = {}
            for sin_name, status, cnt in by_s_result.all():
                by_sin.setdefault(sin_name, {})[status] = cnt

            # ── Average correction magnitude ──────────────────────────
            corr_stmt = select(
                func.avg(
                    func.abs(
                        CalibrationExample.validated_score
                        - CalibrationExample.gemini_raw_score
                    )
                )
            ).where(CalibrationExample.review_status == "corrected")

            corr_result = await session.execute(corr_stmt)
            avg_correction = corr_result.scalar_one_or_none()
            avg_correction_magnitude = (
                round(float(avg_correction), 4) if avg_correction else 0.0
            )

            # ── Coverage map (6 questions x 7 sins) ───────────────────
            cov_stmt = select(
                CalibrationExample.question_number,
                CalibrationExample.sin,
                func.count().label("cnt"),
            ).where(
                CalibrationExample.review_status.in_(["approved", "corrected"])
            ).group_by(
                CalibrationExample.question_number,
                CalibrationExample.sin,
            )

            cov_result = await session.execute(cov_stmt)
            coverage_raw: dict[tuple[int, str], int] = {}
            for qn, sin_name, cnt in cov_result.all():
                coverage_raw[(qn, sin_name)] = cnt

            coverage_map: dict[str, dict[str, dict[str, Any]]] = {}
            total_cells = len(QUESTION_NUMBERS) * len(SIN_NAMES)
            covered_cells = 0
            fully_covered_cells = 0

            for qn in QUESTION_NUMBERS:
                q_key = f"q{qn}"
                coverage_map[q_key] = {}
                for sin_name in SIN_NAMES:
                    count = coverage_raw.get((qn, sin_name), 0)
                    coverage_map[q_key][sin_name] = {
                        "count": count,
                        "target": TARGET_EXAMPLES_PER_CELL,
                        "met": count >= TARGET_EXAMPLES_PER_CELL,
                    }
                    if count > 0:
                        covered_cells += 1
                    if count >= TARGET_EXAMPLES_PER_CELL:
                        fully_covered_cells += 1

            return {
                "total": total,
                "pending": pending,
                "approved": approved,
                "corrected": corrected,
                "rejected": rejected,
                "by_question": by_question,
                "by_sin": by_sin,
                "avg_correction_magnitude": avg_correction_magnitude,
                "coverage_map": coverage_map,
                "coverage_summary": {
                    "total_cells": total_cells,
                    "covered_cells": covered_cells,
                    "fully_covered_cells": fully_covered_cells,
                    "coverage_pct": round(
                        covered_cells / total_cells * 100, 1
                    ) if total_cells else 0.0,
                    "full_coverage_pct": round(
                        fully_covered_cells / total_cells * 100, 1
                    ) if total_cells else 0.0,
                },
            }

        if db_session is not None:
            stats = await _execute(db_session)
        else:
            async with async_session_factory() as session:
                stats = await _execute(session)

        log.info(
            "get_calibration_stats_complete",
            total=stats["total"],
            coverage_pct=stats["coverage_summary"]["coverage_pct"],
        )
        return stats

    # ══════════════════════════════════════════════════════════════════════
    # 6. bulk_review — batch review multiple examples
    # ══════════════════════════════════════════════════════════════════════

    async def bulk_review(
        self,
        reviews: list[dict],
        reviewer: str,
        db_session: AsyncSession | None = None,
    ) -> dict:
        """Apply review actions to multiple calibration examples in a single
        transaction.

        Parameters
        ----------
        reviews:
            List of dicts, each with at minimum ``example_id`` and
            ``action``.  Optionally ``validated_score`` and
            ``review_notes``.
        reviewer:
            Identifier for the reviewing admin (applied to all reviews).
        db_session:
            An active async SQLAlchemy session, or ``None`` to auto-manage.

        Returns
        -------
        dict
            ``{"total": int, "succeeded": int, "failed": int,
              "errors": list[dict]}``
        """
        log = logger.bind(reviewer=reviewer, batch_size=len(reviews))
        log.info("bulk_review_start")

        succeeded = 0
        failed = 0
        errors: list[dict] = []

        async def _apply_all(session: AsyncSession) -> None:
            nonlocal succeeded, failed

            for review in reviews:
                example_id = review.get("example_id")
                action = review.get("action")

                if not example_id or not action:
                    failed += 1
                    errors.append({
                        "example_id": example_id,
                        "error": "Missing example_id or action",
                    })
                    continue

                try:
                    await self.review_example(
                        example_id=example_id,
                        action=action,
                        validated_score=review.get("validated_score"),
                        validated_by=reviewer,
                        review_notes=review.get("review_notes"),
                        db_session=session,
                    )
                    succeeded += 1
                except (ValueError, LookupError) as exc:
                    failed += 1
                    errors.append({
                        "example_id": example_id,
                        "error": str(exc),
                    })
                    log.warning(
                        "bulk_review_item_failed",
                        example_id=example_id,
                        error=str(exc),
                    )

        if db_session is not None:
            await _apply_all(db_session)
        else:
            async with async_session_factory() as session:
                await _apply_all(session)
                await session.commit()

        result = {
            "total": len(reviews),
            "succeeded": succeeded,
            "failed": failed,
            "errors": errors,
        }

        log.info(
            "bulk_review_complete",
            succeeded=succeeded,
            failed=failed,
        )
        return result

    # ══════════════════════════════════════════════════════════════════════
    # 7. get_effectiveness_metrics — calibration programme health
    # ══════════════════════════════════════════════════════════════════════

    async def get_effectiveness_metrics(
        self,
        db_session: AsyncSession | None = None,
    ) -> dict:
        """Return effectiveness metrics for the calibration programme.

        Returns
        -------
        dict
            Contains:
            - ``avg_correction_magnitude``: overall mean |validated -
              gemini_raw| for corrected examples.
            - ``correction_magnitude_by_sin``: per-sin breakdown.
            - ``score_distributions_by_sin``: per-sin stats for approved
              and corrected examples (mean, std, min, max, count).
            - ``drift_report``: comparison of early vs. recent correction
              magnitudes to detect temporal drift.
        """
        log = logger.bind()
        log.info("get_effectiveness_metrics_start")

        async def _execute(session: AsyncSession) -> dict:
            # ── Overall correction magnitude ──────────────────────────
            corr_stmt = select(
                func.avg(
                    func.abs(
                        CalibrationExample.validated_score
                        - CalibrationExample.gemini_raw_score
                    )
                ).label("avg_mag"),
                func.count().label("cnt"),
            ).where(CalibrationExample.review_status == "corrected")

            corr_result = await session.execute(corr_stmt)
            corr_row = corr_result.one()
            avg_correction_magnitude = (
                round(float(corr_row.avg_mag), 4)
                if corr_row.avg_mag is not None
                else 0.0
            )
            total_corrections = corr_row.cnt

            # ── Per-sin correction magnitude ──────────────────────────
            sin_corr_stmt = select(
                CalibrationExample.sin,
                func.avg(
                    func.abs(
                        CalibrationExample.validated_score
                        - CalibrationExample.gemini_raw_score
                    )
                ).label("avg_mag"),
                func.count().label("cnt"),
            ).where(
                CalibrationExample.review_status == "corrected"
            ).group_by(CalibrationExample.sin)

            sin_corr_result = await session.execute(sin_corr_stmt)
            correction_by_sin: dict[str, dict] = {}
            for sin_name, avg_mag, cnt in sin_corr_result.all():
                correction_by_sin[sin_name] = {
                    "avg_magnitude": round(float(avg_mag), 4) if avg_mag else 0.0,
                    "count": cnt,
                }

            # ── Score distributions by sin (approved + corrected) ─────
            dist_stmt = select(
                CalibrationExample.sin,
                func.avg(CalibrationExample.validated_score).label("mean"),
                func.min(CalibrationExample.validated_score).label("min_score"),
                func.max(CalibrationExample.validated_score).label("max_score"),
                func.count().label("cnt"),
            ).where(
                CalibrationExample.review_status.in_(["approved", "corrected"]),
                CalibrationExample.validated_score.isnot(None),
            ).group_by(CalibrationExample.sin)

            dist_result = await session.execute(dist_stmt)
            score_distributions: dict[str, dict] = {}
            for sin_name, mean_val, min_val, max_val, cnt in dist_result.all():
                score_distributions[sin_name] = {
                    "mean": round(float(mean_val), 4) if mean_val else 0.0,
                    "min": round(float(min_val), 4) if min_val else 0.0,
                    "max": round(float(max_val), 4) if max_val else 0.0,
                    "count": cnt,
                }

            # ── Drift report: early vs. recent corrections ────────────
            # Split corrections by median created_at date to compare
            # early-programme vs. recent correction magnitudes.
            drift_report = await self._compute_drift_report(session)

            return {
                "avg_correction_magnitude": avg_correction_magnitude,
                "total_corrections": total_corrections,
                "correction_magnitude_by_sin": correction_by_sin,
                "score_distributions_by_sin": score_distributions,
                "drift_report": drift_report,
            }

        if db_session is not None:
            metrics = await _execute(db_session)
        else:
            async with async_session_factory() as session:
                metrics = await _execute(session)

        log.info(
            "get_effectiveness_metrics_complete",
            avg_correction=metrics["avg_correction_magnitude"],
            total_corrections=metrics["total_corrections"],
        )
        return metrics

    # ══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _row_to_dict(row: CalibrationExample) -> dict:
        """Serialise a CalibrationExample ORM instance to a plain dict."""
        return {
            "id": str(row.id),
            "question_number": row.question_number,
            "response_text": row.response_text,
            "sin": row.sin,
            "gemini_raw_score": row.gemini_raw_score,
            "gemini_raw_confidence": row.gemini_raw_confidence,
            "gemini_raw_evidence": row.gemini_raw_evidence,
            "validated_score": row.validated_score,
            "validated_by": row.validated_by,
            "validated_at": (
                row.validated_at.isoformat() if row.validated_at else None
            ),
            "review_status": row.review_status,
            "review_notes": row.review_notes,
            "source_profile_id": (
                str(row.source_profile_id) if row.source_profile_id else None
            ),
            "created_at": (
                row.created_at.isoformat() if row.created_at else None
            ),
        }

    @staticmethod
    def _select_diverse_examples(
        rows: list[CalibrationExample],
        n: int,
    ) -> list[CalibrationExample]:
        """Select up to *n* examples that maximise score diversity.

        Strategy: from the pre-sorted list (corrected first, then by
        confidence), greedily pick examples whose ``validated_score`` is
        maximally distant from those already selected.
        """
        if len(rows) <= n:
            return list(rows)

        # Always include the first row (highest-priority: corrected or
        # highest-confidence approved).
        selected: list[CalibrationExample] = [rows[0]]
        remaining = list(rows[1:])

        while len(selected) < n and remaining:
            # Pick the candidate whose validated_score is most distant
            # from the nearest already-selected score.
            best_idx = 0
            best_min_dist = -1.0

            selected_scores = [
                r.validated_score if r.validated_score is not None
                else r.gemini_raw_score
                for r in selected
            ]

            for i, candidate in enumerate(remaining):
                c_score = (
                    candidate.validated_score
                    if candidate.validated_score is not None
                    else candidate.gemini_raw_score
                )
                min_dist = min(abs(c_score - s) for s in selected_scores)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    async def _compute_drift_report(session: AsyncSession) -> dict:
        """Compare correction magnitudes from the first half vs. the second
        half of corrected examples (by creation date) to detect temporal
        drift in Gemini scoring accuracy.

        Returns
        -------
        dict
            ``{"early_avg_magnitude", "recent_avg_magnitude",
              "drift_direction", "drift_magnitude"}``
        """
        # Fetch all corrected examples ordered by creation date
        stmt = (
            select(
                CalibrationExample.gemini_raw_score,
                CalibrationExample.validated_score,
                CalibrationExample.created_at,
            )
            .where(CalibrationExample.review_status == "corrected")
            .order_by(CalibrationExample.created_at.asc())
        )

        result = await session.execute(stmt)
        corrections = result.all()

        if len(corrections) < 4:
            return {
                "early_avg_magnitude": 0.0,
                "recent_avg_magnitude": 0.0,
                "drift_direction": "insufficient_data",
                "drift_magnitude": 0.0,
                "sample_size": len(corrections),
            }

        midpoint = len(corrections) // 2
        early = corrections[:midpoint]
        recent = corrections[midpoint:]

        early_magnitudes = [
            abs(float(row[1]) - float(row[0])) for row in early
            if row[0] is not None and row[1] is not None
        ]
        recent_magnitudes = [
            abs(float(row[1]) - float(row[0])) for row in recent
            if row[0] is not None and row[1] is not None
        ]

        early_avg = (
            sum(early_magnitudes) / len(early_magnitudes)
            if early_magnitudes
            else 0.0
        )
        recent_avg = (
            sum(recent_magnitudes) / len(recent_magnitudes)
            if recent_magnitudes
            else 0.0
        )

        drift_magnitude = recent_avg - early_avg

        if abs(drift_magnitude) < 0.1:
            drift_direction = "stable"
        elif drift_magnitude > 0:
            drift_direction = "worsening"
        else:
            drift_direction = "improving"

        return {
            "early_avg_magnitude": round(early_avg, 4),
            "recent_avg_magnitude": round(recent_avg, 4),
            "drift_direction": drift_direction,
            "drift_magnitude": round(drift_magnitude, 4),
            "sample_size": len(corrections),
        }
