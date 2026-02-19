"""
Harmonia V3 — GeminiService: Core AI Parsing Engine for the PIIP System

This is the central service responsible for converting free-text questionnaire
responses into 7-dimensional personality profiles anchored to the Seven Deadly
Sins framework.  It orchestrates:

- Per-sin concurrent trait extraction via the Gemini LLM
- Multi-model fallback chains with exponential-backoff retry
- Lightweight LIWC-inspired linguistic signal extraction
- Cross-response discrepancy detection
- Multi-observer consensus protocol for high-bias or ambiguous traits
- Fuzzy evidence location within source text
- Robust JSON response parsing with multiple fallback strategies

All scoring uses a bipolar [-5, +5] anchored scale with explicit negative,
neutral, and positive descriptors to reduce model anchoring bias.

Model fallback chain:
    gemini-3-pro-preview -> gemini-3-flash-preview -> gemini-2.5-flash

Multi-observer consensus for high-bias-risk traits (Wrath, Envy, Pride) when
initial confidence is below 0.70.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
import uuid
from typing import Any

import google.generativeai as genai
import structlog
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from jsonrepair import repair_json
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SIN_NAMES: list[str] = [
    "greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth",
]

# ────────────────────────────────────────────────────────────────────────────
# Word lists for lightweight LIWC-style linguistic analysis
# ────────────────────────────────────────────────────────────────────────────

_FIRST_PERSON_SINGULAR = re.compile(
    r"\b(i|me|my|mine|myself)\b", re.IGNORECASE
)
_FIRST_PERSON_PLURAL = re.compile(
    r"\b(we|us|our|ours|ourselves)\b", re.IGNORECASE
)
_NEGATIVE_EMOTION_WORDS = {
    "angry", "anger", "hate", "furious", "frustrated", "annoyed", "irritated",
    "resentful", "bitter", "hostile", "disgusted", "contempt", "sad", "miserable",
    "depressed", "anxious", "worried", "fearful", "afraid", "terrified",
    "jealous", "envious", "guilty", "ashamed", "hurt", "lonely", "desperate",
    "hopeless", "worthless", "terrible", "awful", "horrible", "dreadful",
    "painful", "suffering", "agony", "torment", "rage", "fury", "loathe",
}
_POSITIVE_EMOTION_WORDS = {
    "happy", "joy", "love", "excited", "grateful", "thankful", "blessed",
    "wonderful", "amazing", "fantastic", "excellent", "great", "good",
    "beautiful", "brilliant", "delighted", "pleased", "satisfied", "proud",
    "confident", "hopeful", "optimistic", "enthusiastic", "passionate",
    "peaceful", "calm", "content", "fulfilled", "inspired", "motivated",
    "thrilled", "elated", "cheerful", "glad", "kind", "generous", "caring",
    "compassionate", "empathetic", "warm", "supportive",
}
_CERTAINTY_WORDS = {
    "always", "never", "absolutely", "definitely", "certainly", "undoubtedly",
    "clearly", "obviously", "surely", "without doubt", "every time",
    "guaranteed", "100%", "completely", "entirely", "totally", "must",
    "impossible", "unquestionable",
}
_HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "somewhat", "sort of",
    "kind of", "probably", "likely", "tend to", "in a way", "sometimes",
    "occasionally", "I think", "I guess", "I suppose", "it seems",
    "not sure", "uncertain",
}
_FUTURE_ORIENTATION_WORDS = {
    "will", "going to", "plan", "planning", "hope to", "intend", "goal",
    "future", "tomorrow", "someday", "eventually", "aspire", "dream",
    "ambition", "aim", "target", "vision", "roadmap", "next",
}


def _is_retryable_api_error(exc: BaseException) -> bool:
    """Return True if the exception signals a retryable Gemini API error.

    We retry on HTTP 429 (rate limit) and 500/503 (server-side transient)
    errors.  The google-generativeai SDK wraps these as various exception
    types, so we inspect both the type name and string representation.
    """
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    # Rate limit
    if "429" in exc_str or "resource_exhausted" in exc_str:
        return True
    # Server errors
    if "500" in exc_str or "503" in exc_str or "internal" in exc_str:
        return True
    # Google API-specific classes
    if "resourceexhausted" in exc_type or "serviceunavailable" in exc_type:
        return True

    return False


class GeminiService:
    """Core AI parsing engine for the Personality Inference via
    Implicit Projection (PIIP) system.

    Converts free-text questionnaire answers into seven-dimensional sin
    profiles by orchestrating Gemini LLM calls with anchored prompts,
    multi-observer consensus, and robust post-processing.
    """

    # ── Class-level constants ─────────────────────────────────────────

    SIN_NAMES: list[str] = SIN_NAMES

    HIGH_BIAS_SINS: list[str] = ["wrath", "envy", "pride"]

    TRAIT_ANCHORS: dict[str, dict[str, str]] = {
        "greed": {
            "negative": "Extremely generous, prioritises others",
            "positive": "Highly materialistic, accumulates at others' expense",
        },
        "pride": {
            "negative": "Deeply humble, deflects credit",
            "positive": "Ego-driven, seeks status and validation constantly",
        },
        "lust": {
            "negative": "Very restrained, deliberate, avoids spontaneity",
            "positive": "Highly impulsive, novelty-seeking",
        },
        "wrath": {
            "negative": "Extreme conflict avoidance, never expresses anger",
            "positive": "Quick to anger, confrontational",
        },
        "gluttony": {
            "negative": "Highly moderate, strict self-control",
            "positive": "Strongly indulgent, struggles with restraint",
        },
        "envy": {
            "negative": "Deeply content, never compares",
            "positive": "Constantly competitive, resentful of success",
        },
        "sloth": {
            "negative": "Extremely proactive, takes initiative",
            "positive": "Avoidant, passive, procrastinates",
        },
    }

    # Multi-observer personas used for consensus protocol
    OBSERVER_PERSONAS: list[dict[str, str]] = [
        {
            "name": "Neutral",
            "instruction": (
                "You are a NEUTRAL psychological assessor. Evaluate the "
                "response strictly based on observable linguistic evidence. "
                "Do not infer intent beyond what is explicitly stated."
            ),
        },
        {
            "name": "Empathetic Therapist",
            "instruction": (
                "You are an EMPATHETIC THERAPIST psychological assessor. "
                "Consider the respondent's context and possible motivations "
                "charitably. Look for underlying emotional signals and give "
                "reasonable benefit of the doubt where the text is ambiguous."
            ),
        },
        {
            "name": "Skeptical Critic",
            "instruction": (
                "You are a SKEPTICAL CRITIC psychological assessor. Probe for "
                "social desirability bias and impression management. Discount "
                "self-flattering language and look for contradictions or "
                "deflections that may reveal the true trait level."
            ),
        },
    ]

    # ── Initialisation ────────────────────────────────────────────────

    def __init__(
        self,
        calibration_service: Any | None = None,
    ) -> None:
        """Configure the Gemini client, model fallback chain, and safety
        settings.

        Uses ``get_settings()`` to load the API key and model identifiers
        from application configuration.

        Parameters
        ----------
        calibration_service:
            Optional reference to the CalibrationService for fetching
            validated few-shot examples from the calibration DB.  When
            ``None``, prompts use only abstract scale anchors (zero-shot).
        """
        settings = get_settings()
        self._calibration_service = calibration_service

        # Configure the Gemini SDK with the API key
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Model fallback chain: primary -> fallback -> stable
        self._model_chain: list[str] = [
            settings.GEMINI_MODEL_PRIMARY,    # "gemini-3-pro-preview"
            settings.GEMINI_MODEL_FALLBACK,   # "gemini-3-flash-preview"
            settings.GEMINI_MODEL_STABLE,     # "gemini-2.5-flash"
        ]

        # Safety settings: disable all content blocking to allow authentic
        # personality analysis of potentially confrontational or sensitive
        # language in user responses.
        self._safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Generation config for deterministic, structured output
        self._generation_config = genai.GenerationConfig(
            max_output_tokens=1024,
            response_mime_type="application/json",
        )

        logger.info(
            "gemini_service_initialised",
            model_chain=self._model_chain,
            has_calibration_service=calibration_service is not None,
        )

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    async def parse_single_response(
        self,
        question: str,
        answer: str,
        question_number: int,
        user_id: str,
    ) -> dict:
        """Parse a single Q&A pair into 7 sin scores with evidence.

        Runs 7 concurrent Gemini calls (one per sin), extracts LIWC-style
        linguistic signals, detects cross-dimension discrepancies, and
        triggers multi-observer consensus for ambiguous or high-bias
        traits.

        After parsing, creates ``ParsingEvidence`` records with character
        offsets for full traceability.

        Parameters
        ----------
        question:
            The questionnaire question text.
        answer:
            The user's free-text response.
        question_number:
            The ordinal position of this question (1-6).
        user_id:
            UUID string of the user (for evidence storage).

        Returns
        -------
        dict
            Contains ``sins`` (per-sin score dicts), ``liwc_signals``,
            ``discrepancies``, ``question_number``, ``evidence_records``,
            and ``metadata``.
        """
        start_time = time.monotonic()

        logger.info(
            "parse_single_response_start",
            question_number=question_number,
            user_id=user_id,
            answer_length=len(answer),
        )

        # Run all 7 sin parsings concurrently
        tasks = [
            self._parse_single_trait(question, answer, sin, question_number)
            for sin in self.SIN_NAMES
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results — handle any per-sin failures gracefully
        sins: dict[str, dict] = {}
        failed_sins: list[str] = []

        for sin_name, result in zip(self.SIN_NAMES, raw_results):
            if isinstance(result, Exception):
                logger.error(
                    "sin_parsing_failed",
                    sin=sin_name,
                    question_number=question_number,
                    error=str(result),
                )
                failed_sins.append(sin_name)
                sins[sin_name] = {
                    "score": 0.0,
                    "confidence": 0.0,
                    "evidence": "",
                    "evidence_start": -1,
                    "evidence_end": -1,
                    "model_used": "none",
                    "error": str(result),
                }
            else:
                sins[sin_name] = result

        # ── Multi-observer consensus for high-bias / low-confidence ───
        sins = await self._apply_multi_observer_consensus(
            question, answer, question_number, sins
        )

        # ── Linguistic signals ────────────────────────────────────────
        liwc_signals = self._extract_liwc_signals(answer)

        # ── Discrepancy detection ─────────────────────────────────────
        discrepancies = self._detect_discrepancies(answer, sins)

        # ── Build ParsingEvidence records ─────────────────────────────
        evidence_records: list[dict] = []
        for sin_name, sin_data in sins.items():
            evidence_records.append({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "question_number": question_number,
                "sin": sin_name,
                "score": sin_data.get("score", 0.0),
                "confidence": sin_data.get("confidence", 0.0),
                "evidence_snippet": sin_data.get("evidence", ""),
                "snippet_start_index": sin_data.get("evidence_start", -1),
                "snippet_end_index": sin_data.get("evidence_end", -1),
                "interpretation": sin_data.get("interpretation", ""),
                "observer_persona": (
                    "multi_observer"
                    if sin_data.get("consensus_method") == "multi_observer"
                    else None
                ),
                "gemini_model_used": sin_data.get("model_used", "unknown"),
            })

        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

        logger.info(
            "parse_single_response_complete",
            question_number=question_number,
            user_id=user_id,
            elapsed_ms=elapsed_ms,
            failed_sins=failed_sins,
            discrepancies_count=len(discrepancies),
        )

        return {
            "question_number": question_number,
            "user_id": user_id,
            "response_text": answer,
            "word_count": len(answer.split()),
            "sins": sins,
            "scores": [
                {
                    "sin": sin_name,
                    "score": sins[sin_name]["score"],
                    "confidence": sins[sin_name]["confidence"],
                    "evidence": sins[sin_name].get("evidence", ""),
                    "evidence_start": sins[sin_name].get("evidence_start", -1),
                    "evidence_end": sins[sin_name].get("evidence_end", -1),
                }
                for sin_name in self.SIN_NAMES
            ],
            "liwc_signals": liwc_signals,
            "discrepancies": discrepancies,
            "evidence_records": evidence_records,
            "metadata": {
                "elapsed_ms": elapsed_ms,
                "failed_sins": failed_sins,
                "answer_word_count": len(answer.split()),
            },
        }

    async def parse_all_responses(
        self,
        responses: list[dict],
        user_id: str,
    ) -> dict:
        """Parse all 6 questionnaire Q&A pairs, return per-question results
        plus aggregated data.

        Parameters
        ----------
        responses:
            List of dicts, each with ``question_number``, ``question_text``
            (or ``question``), and ``response_text`` (or ``answer``) keys.
        user_id:
            UUID string of the user.

        Returns
        -------
        dict
            Contains ``per_question`` results, ``aggregated_sins``,
            ``overall_confidence``, ``evidence_records``, ``consistency_flags``,
            ``discrepancies``, ``flags``, and ``metadata``.
        """
        start_time = time.monotonic()

        logger.info(
            "parse_all_responses_start",
            user_id=user_id,
            response_count=len(responses),
        )

        # Parse each response (sequentially to respect rate limits, but
        # each individual parse runs 7 concurrent sin evaluations)
        per_question_results: list[dict] = []
        all_evidence_records: list[dict] = []

        for resp in responses:
            question_text = resp.get("question_text", resp.get("question", ""))
            answer_text = resp.get("response_text", resp.get("answer", ""))

            result = await self.parse_single_response(
                question=question_text,
                answer=answer_text,
                question_number=resp["question_number"],
                user_id=user_id,
            )
            per_question_results.append(result)
            all_evidence_records.extend(result.get("evidence_records", []))

        # ── Aggregate across all questions ────────────────────────────
        aggregated_sins: dict[str, dict] = {}
        all_flags: list[str] = []

        for sin in self.SIN_NAMES:
            scores = []
            confidences = []
            all_evidence: list[str] = []

            for qr in per_question_results:
                sin_data = qr["sins"].get(sin, {})
                score = sin_data.get("score", 0.0)
                conf = sin_data.get("confidence", 0.0)
                evidence = sin_data.get("evidence", "")

                scores.append(score)
                confidences.append(conf)
                if evidence:
                    all_evidence.append(evidence)

                # Collect any per-sin flags
                if sin_data.get("observer_disagreement"):
                    all_flags.append(
                        f"observer_disagreement_{sin}_q{qr['question_number']}"
                    )

            mean_score = statistics.mean(scores) if scores else 0.0
            mean_confidence = statistics.mean(confidences) if confidences else 0.0
            score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0

            aggregated_sins[sin] = {
                "score": round(mean_score, 3),
                "confidence": round(mean_confidence, 3),
                "score_std": round(score_std, 3),
                "per_question_scores": scores,
                "evidence_samples": all_evidence[:3],  # Top 3
            }

            # Flag high cross-question variance
            if score_std > 2.5:
                all_flags.append(f"high_variance_{sin}")

        # Collect discrepancies from all questions
        all_discrepancies: list[str] = []
        for qr in per_question_results:
            all_discrepancies.extend(qr.get("discrepancies", []))

        # Cross-response consistency check (max delta > 5 per sin)
        consistency_flags = self._check_cross_response_consistency(
            per_question_results
        )
        all_flags.extend(consistency_flags)

        # Overall confidence: mean of per-sin confidences
        overall_conf_values = [
            s["confidence"] for s in aggregated_sins.values()
        ]
        overall_confidence = (
            statistics.mean(overall_conf_values)
            if overall_conf_values
            else 0.0
        )

        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

        logger.info(
            "parse_all_responses_complete",
            user_id=user_id,
            elapsed_ms=elapsed_ms,
            overall_confidence=round(overall_confidence, 3),
            flag_count=len(all_flags),
        )

        return {
            "user_id": user_id,
            "per_question": per_question_results,
            "aggregated_sins": aggregated_sins,
            "overall_confidence": round(overall_confidence, 3),
            "discrepancies": all_discrepancies,
            "evidence_records": all_evidence_records,
            "consistency_flags": consistency_flags,
            "flags": list(set(all_flags)),
            "metadata": {
                "total_elapsed_ms": elapsed_ms,
                "questions_parsed": len(per_question_results),
                "model_chain": self._model_chain,
            },
        }

    # ══════════════════════════════════════════════════════════════════
    # Core trait parsing
    # ══════════════════════════════════════════════════════════════════

    async def _parse_single_trait(
        self,
        question: str,
        answer: str,
        sin: str,
        question_number: int,
    ) -> dict:
        """Parse a single trait (sin) from one Q&A pair using the Gemini
        model chain with retry logic.

        Builds an anchored prompt, injects few-shot examples from the
        calibration DB if available, calls Gemini with the fallback chain,
        and returns a structured result dict.

        Parameters
        ----------
        question:
            The questionnaire question text.
        answer:
            The user's response text.
        sin:
            One of the seven sin names to evaluate.
        question_number:
            Ordinal question position (for calibration lookup).

        Returns
        -------
        dict
            ``{"score", "confidence", "evidence", "evidence_start",
            "evidence_end", "interpretation", "model_used"}``.
        """
        # Fetch few-shot calibration examples if service is available
        few_shot_examples = await self._get_calibration_examples(
            sin, question_number
        )

        prompt = self._build_trait_prompt(
            question, answer, sin, few_shot_examples
        )

        # Try each model in the chain
        last_exception: Exception | None = None

        for model_name in self._model_chain:
            try:
                response_text = await self._call_gemini_with_retry(
                    model_name, prompt
                )
                parsed = self._parse_json_response(response_text)

                # Validate and normalise the parsed response
                score = float(parsed.get("score", 0.0))
                score = max(-5.0, min(5.0, score))  # Clamp to [-5, +5]

                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

                evidence = str(parsed.get("evidence", ""))
                interpretation = str(parsed.get("interpretation", ""))

                # Locate evidence in the original response
                ev_start, ev_end = self._locate_evidence_in_response(
                    answer, evidence
                )

                logger.debug(
                    "trait_parsed",
                    sin=sin,
                    score=score,
                    confidence=confidence,
                    model=model_name,
                    question_number=question_number,
                )

                return {
                    "score": round(score, 3),
                    "confidence": round(confidence, 3),
                    "evidence": evidence,
                    "evidence_start": ev_start,
                    "evidence_end": ev_end,
                    "interpretation": interpretation,
                    "model_used": model_name,
                }

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "model_fallback",
                    sin=sin,
                    failed_model=model_name,
                    error=str(exc),
                    question_number=question_number,
                )
                continue

        # All models in chain exhausted
        raise RuntimeError(
            f"All models in chain exhausted for sin={sin}, "
            f"question={question_number}. Last error: {last_exception}"
        )

    async def _call_gemini_with_retry(
        self,
        model_name: str,
        prompt: str,
    ) -> str:
        """Call a specific Gemini model with tenacity retry on transient
        errors.

        Uses exponential backoff: 1s initial wait, 2x multiplier, 60s
        max wait, up to 5 attempts.

        Parameters
        ----------
        model_name:
            The Gemini model identifier (e.g. ``gemini-3-pro-preview``).
        prompt:
            The fully constructed prompt string.

        Returns
        -------
        str
            The raw text content from the Gemini response.

        Raises
        ------
        Exception
            If all retry attempts are exhausted or a non-retryable error
            occurs.
        """
        model = genai.GenerativeModel(model_name)

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable_api_error),
                stop=stop_after_attempt(5),
                wait=wait_exponential(
                    multiplier=1,
                    min=1,
                    max=60,
                    exp_base=2,
                ),
                reraise=True,
            ):
                with attempt:
                    logger.debug(
                        "gemini_call_attempt",
                        model=model_name,
                        attempt_number=attempt.retry_state.attempt_number,
                    )
                    response = await model.generate_content_async(
                        prompt,
                        safety_settings=self._safety_settings,
                        generation_config=self._generation_config,
                    )

                    if not response.candidates:
                        raise ValueError(
                            f"Gemini returned no candidates for model "
                            f"{model_name}. Prompt feedback: "
                            f"{response.prompt_feedback}"
                        )

                    text = response.text
                    if not text or not text.strip():
                        raise ValueError(
                            f"Gemini returned empty text for model "
                            f"{model_name}"
                        )

                    return text

        except RetryError as retry_err:
            logger.error(
                "gemini_retry_exhausted",
                model=model_name,
                attempts=5,
                last_error=str(retry_err.last_attempt.exception()),
            )
            raise retry_err.last_attempt.exception() from retry_err

    # ══════════════════════════════════════════════════════════════════
    # Prompt construction
    # ══════════════════════════════════════════════════════════════════

    def _build_trait_prompt(
        self,
        question: str,
        answer: str,
        sin: str,
        few_shot_examples: list[dict],
    ) -> str:
        """Build a focused trait-evaluation prompt for a specific sin
        dimension.

        The prompt uses an anchored bipolar scale at -5, 0, and +5 to
        reduce central-tendency and acquiescence bias.  If calibration
        examples are available, they are included as few-shot references
        to improve scoring consistency.

        Parameters
        ----------
        question:
            The questionnaire question text.
        answer:
            The user's response text.
        sin:
            The sin dimension to evaluate.
        few_shot_examples:
            Validated examples from the calibration DB.

        Returns
        -------
        str
            The complete prompt string.
        """
        anchors = self.TRAIT_ANCHORS[sin]

        # Build the few-shot section
        few_shot_section = ""
        if few_shot_examples:
            examples_text = []
            for i, ex in enumerate(few_shot_examples, 1):
                validated_score = ex.get(
                    "validated_score",
                    ex.get("score", ex.get("gemini_raw_score", 0)),
                )
                evidence_snippet = ex.get(
                    "evidence_snippet",
                    ex.get("evidence", ex.get("gemini_raw_evidence", "")),
                )
                review_notes = ex.get("review_notes", "")

                entry_lines = [
                    f"  Example {i}:",
                    f"    Response: \"{ex.get('response_text', '')}\"",
                ]

                notes_suffix = ""
                if review_notes:
                    notes_suffix = f" ({review_notes})"

                entry_lines.append(
                    f"    {sin.capitalize()} Score: "
                    f"{validated_score}{notes_suffix}"
                )

                if evidence_snippet:
                    entry_lines.append(
                        f"    Key evidence: \"{evidence_snippet}\""
                    )

                examples_text.append("\n".join(entry_lines))

            few_shot_section = (
                "\n\n--- Reference Examples (validated by human reviewers) ---\n"
                + "\n\n".join(examples_text)
                + "\n--- End of Reference Examples ---\n"
                + f"\nNow analyze this NEW response for {sin.upper()} signals:\n"
            )

        prompt = f"""You are a psychological assessment expert specialising in personality trait extraction from open-ended text responses.

TASK: Evaluate the following questionnaire response for the trait dimension "{sin.upper()}".

TRAIT SCALE (Bipolar, anchored):
  -5 (Extreme Negative): {anchors['negative']}
   0 (Neutral): No clear signal for this trait in either direction
  +5 (Extreme Positive): {anchors['positive']}

IMPORTANT SCORING GUIDELINES:
- Use the FULL range of the scale [-5 to +5]. Do NOT cluster scores around 0.
- A score of 0 means genuinely NO signal, not "I'm unsure".
- Confidence reflects how CLEARLY the text reveals this trait, not your certainty about the person.
- Confidence range: 0.0 (no relevant content) to 1.0 (explicit, unambiguous signal).
- Extract a SPECIFIC quote from the response as evidence. The evidence MUST be a direct substring of the response text.
- Provide a one-line interpretation explaining what the evidence reveals about this trait.
- If the response contains no relevant signal for this trait, return score=0, confidence below 0.3, evidence="" (empty string), and interpretation="No clear signal".
{few_shot_section}
QUESTIONNAIRE QUESTION:
"{question}"

USER RESPONSE:
"{answer}"

Return a JSON object with exactly these fields:
{{
  "score": <float between -5.0 and 5.0>,
  "confidence": <float between 0.0 and 1.0>,
  "evidence": "<exact quote from the user response that supports your score>",
  "interpretation": "<one-line explanation of what the evidence reveals about this trait>"
}}

Respond with ONLY the JSON object. No additional text or explanation."""

        return prompt

    # ══════════════════════════════════════════════════════════════════
    # Multi-observer consensus protocol
    # ══════════════════════════════════════════════════════════════════

    async def _apply_multi_observer_consensus(
        self,
        question: str,
        answer: str,
        question_number: int,
        sins: dict[str, dict],
    ) -> dict[str, dict]:
        """Apply multi-observer consensus for high-bias sins with low
        confidence.

        When a high-bias sin (wrath, envy, pride) has confidence < 0.70,
        this method runs 3 parallel evaluations with different observer
        personas (Neutral, Empathetic Therapist, Skeptical Critic) and
        aggregates the results.

        The aggregated std_dev determines a confidence multiplier:
        - std_dev < 0.3  ->  multiplier 1.0 (strong agreement)
        - 0.3 <= std_dev <= 0.5  ->  multiplier 0.85
        - std_dev > 0.5  ->  multiplier 0.6 + flag "observer_disagreement"
        """
        updated_sins = dict(sins)
        consensus_tasks: list[tuple[str, asyncio.Task]] = []

        # Determine which sins need multi-observer consensus
        for sin_name, sin_data in sins.items():
            confidence = sin_data.get("confidence", 0.0)
            needs_consensus = False

            # High-bias sins with confidence < 0.70
            if sin_name in self.HIGH_BIAS_SINS and confidence < 0.70:
                needs_consensus = True
                logger.info(
                    "multi_observer_triggered",
                    sin=sin_name,
                    confidence=confidence,
                    question_number=question_number,
                )

            if needs_consensus and not sin_data.get("error"):
                task = asyncio.create_task(
                    self._run_multi_observer(
                        question, answer, sin_name, question_number
                    )
                )
                consensus_tasks.append((sin_name, task))

        # Await all consensus tasks
        if consensus_tasks:
            logger.info(
                "multi_observer_batch",
                sins=[s for s, _ in consensus_tasks],
                question_number=question_number,
            )

        for sin_name, task in consensus_tasks:
            try:
                consensus_result = await task
                updated_sins[sin_name] = consensus_result
            except Exception as exc:
                logger.error(
                    "multi_observer_failed",
                    sin=sin_name,
                    question_number=question_number,
                    error=str(exc),
                )
                # Keep the original single-observer result

        return updated_sins

    async def _run_multi_observer(
        self,
        question: str,
        answer: str,
        sin: str,
        question_number: int,
    ) -> dict:
        """Run 3 parallel evaluations with different observer personas
        and aggregate.

        Personas: Neutral, Empathetic Therapist, Skeptical Critic.

        Aggregation:
        - std_dev < 0.3 -> confidence multiplier 1.0
        - 0.3 <= std_dev <= 0.5 -> multiplier 0.85
        - std_dev > 0.5 -> multiplier 0.6 + flag "observer_disagreement"

        Returns an updated sin dict with consensus score, confidence,
        and potential observer_disagreement flag.
        """
        observer_tasks = [
            self._parse_trait_as_persona(
                question, answer, sin, question_number, persona
            )
            for persona in self.OBSERVER_PERSONAS
        ]
        results = await asyncio.gather(*observer_tasks, return_exceptions=True)

        # Collect successful results
        valid_results: list[dict] = []
        for persona, result in zip(self.OBSERVER_PERSONAS, results):
            if isinstance(result, Exception):
                logger.warning(
                    "observer_persona_failed",
                    persona=persona["name"],
                    sin=sin,
                    error=str(result),
                )
            else:
                valid_results.append(result)

        if not valid_results:
            raise RuntimeError(
                f"All observer personas failed for sin={sin}, "
                f"question={question_number}"
            )

        # Aggregate scores
        scores = [r["score"] for r in valid_results]
        confidences = [r["confidence"] for r in valid_results]

        mean_score = statistics.mean(scores)
        mean_confidence = statistics.mean(confidences)
        score_std = (
            statistics.stdev(scores) if len(scores) > 1 else 0.0
        )

        # Determine confidence multiplier based on observer agreement
        observer_disagreement = False
        if score_std < 0.3:
            confidence_multiplier = 1.0
        elif score_std <= 0.5:
            confidence_multiplier = 0.85
        else:
            confidence_multiplier = 0.6
            observer_disagreement = True

        adjusted_confidence = min(
            1.0, mean_confidence * confidence_multiplier
        )

        # Select best evidence from the highest-confidence observer
        best_result = max(valid_results, key=lambda r: r["confidence"])

        logger.info(
            "multi_observer_aggregated",
            sin=sin,
            question_number=question_number,
            mean_score=round(mean_score, 3),
            score_std=round(score_std, 3),
            confidence_multiplier=confidence_multiplier,
            observer_disagreement=observer_disagreement,
            observer_count=len(valid_results),
        )

        return {
            "score": round(mean_score, 3),
            "confidence": round(adjusted_confidence, 3),
            "evidence": best_result.get("evidence", ""),
            "evidence_start": best_result.get("evidence_start", -1),
            "evidence_end": best_result.get("evidence_end", -1),
            "interpretation": best_result.get("interpretation", ""),
            "model_used": best_result.get("model_used", "unknown"),
            "observer_disagreement": observer_disagreement,
            "observer_scores": scores,
            "observer_std": round(score_std, 3),
            "consensus_method": "multi_observer",
            "observer_details": {
                r.get("persona", "unknown"): r["score"]
                for r in valid_results
            },
        }

    async def _parse_trait_as_persona(
        self,
        question: str,
        answer: str,
        sin: str,
        question_number: int,
        persona: dict[str, str],
    ) -> dict:
        """Parse a trait from a specific observer persona's perspective.

        Each persona has a distinct evaluation stance:
        - Neutral: balanced, evidence-focused
        - Empathetic Therapist: gives benefit of the doubt, attentive to context
        - Skeptical Critic: probes for inconsistencies, discounts social desirability

        Parameters
        ----------
        question:
            The questionnaire question text.
        answer:
            The user's response text.
        sin:
            The sin dimension to evaluate.
        question_number:
            Ordinal question position.
        persona:
            Dict with ``"name"`` and ``"instruction"`` keys.

        Returns
        -------
        dict
            Score result dict with ``persona`` field.
        """
        anchors = self.TRAIT_ANCHORS[sin]

        prompt = f"""{persona['instruction']}

TASK: Evaluate the following questionnaire response for the trait dimension "{sin.upper()}".

TRAIT SCALE (Bipolar, anchored):
  -5 (Extreme Negative): {anchors['negative']}
   0 (Neutral): No clear signal for this trait in either direction
  +5 (Extreme Positive): {anchors['positive']}

SCORING GUIDELINES:
- Use the FULL range [-5 to +5]. Score 0 means genuinely NO signal.
- Confidence: 0.0 (no relevant content) to 1.0 (explicit, unambiguous signal).
- Evidence must be a DIRECT quote from the response text.
- Provide a one-line interpretation explaining what the evidence reveals.

QUESTIONNAIRE QUESTION:
"{question}"

USER RESPONSE:
"{answer}"

Return a JSON object with exactly these fields:
{{
  "score": <float between -5.0 and 5.0>,
  "confidence": <float between 0.0 and 1.0>,
  "evidence": "<exact quote from the user response>",
  "interpretation": "<one-line explanation of what the evidence reveals>"
}}

Respond with ONLY the JSON object."""

        # Use the model chain with retry
        last_exception: Exception | None = None
        for model_name in self._model_chain:
            try:
                response_text = await self._call_gemini_with_retry(
                    model_name, prompt
                )
                parsed = self._parse_json_response(response_text)

                score = float(parsed.get("score", 0.0))
                score = max(-5.0, min(5.0, score))

                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                evidence = str(parsed.get("evidence", ""))
                interpretation = str(parsed.get("interpretation", ""))
                ev_start, ev_end = self._locate_evidence_in_response(
                    answer, evidence
                )

                return {
                    "score": round(score, 3),
                    "confidence": round(confidence, 3),
                    "evidence": evidence,
                    "evidence_start": ev_start,
                    "evidence_end": ev_end,
                    "interpretation": interpretation,
                    "model_used": model_name,
                    "persona": persona["name"],
                }

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "persona_model_fallback",
                    persona=persona["name"],
                    sin=sin,
                    failed_model=model_name,
                    error=str(exc),
                )
                continue

        raise RuntimeError(
            f"All models exhausted for persona={persona['name']}, sin={sin}. "
            f"Last error: {last_exception}"
        )

    # ══════════════════════════════════════════════════════════════════
    # LIWC-style linguistic extraction
    # ══════════════════════════════════════════════════════════════════

    def _extract_liwc_signals(self, text: str) -> dict:
        """Perform lightweight LIWC-inspired linguistic analysis on the
        response text.

        Extracts frequency counts (normalised by word count) for:
        - First person singular (I, me, my, mine, myself)
        - First person plural (we, us, our, ours, ourselves)
        - Negative emotion words
        - Positive emotion words
        - Certainty markers
        - Hedging words
        - Future orientation words

        Parameters
        ----------
        text:
            The user's response text.

        Returns
        -------
        dict
            Counts and ratios for each linguistic category.
        """
        words = text.lower().split()
        word_count = len(words) if words else 1  # Avoid division by zero

        # Strip punctuation from individual words for matching
        clean_words = [
            re.sub(r"[^\w']", "", w) for w in words
        ]
        clean_word_set = set(clean_words)

        # First person singular (regex-based for accuracy)
        fps_count = len(_FIRST_PERSON_SINGULAR.findall(text))

        # First person plural
        fpp_count = len(_FIRST_PERSON_PLURAL.findall(text))

        # Emotion words (set intersection for speed)
        neg_emotion_count = len(clean_word_set & _NEGATIVE_EMOTION_WORDS)
        pos_emotion_count = len(clean_word_set & _POSITIVE_EMOTION_WORDS)

        # Certainty words (some are multi-word, check full text)
        certainty_count = 0
        for word in _CERTAINTY_WORDS:
            if " " in word:
                certainty_count += text.lower().count(word)
            elif word.lower() in clean_word_set:
                certainty_count += 1

        # Hedging words
        hedging_count = 0
        for word in _HEDGING_WORDS:
            if " " in word:
                hedging_count += text.lower().count(word)
            elif word.lower() in clean_word_set:
                hedging_count += 1

        # Future orientation words
        future_count = 0
        for word in _FUTURE_ORIENTATION_WORDS:
            if " " in word:
                future_count += text.lower().count(word)
            elif word.lower() in clean_word_set:
                future_count += 1

        return {
            "first_person_singular": fps_count,
            "first_person_plural": fpp_count,
            "negative_emotion_words": neg_emotion_count,
            "positive_emotion_words": pos_emotion_count,
            "certainty_words": certainty_count,
            "hedging_words": hedging_count,
            "future_orientation_words": future_count,
            "word_count": len(words),
            # Normalised ratios (per word, for comparability)
            "fps_ratio": round(fps_count / word_count, 4),
            "fpp_ratio": round(fpp_count / word_count, 4),
            "neg_emotion_ratio": round(neg_emotion_count / word_count, 4),
            "pos_emotion_ratio": round(pos_emotion_count / word_count, 4),
            "certainty_ratio": round(certainty_count / word_count, 4),
            "hedging_ratio": round(hedging_count / word_count, 4),
            "future_ratio": round(future_count / word_count, 4),
        }

    # ══════════════════════════════════════════════════════════════════
    # Discrepancy detection (social desirability bias)
    # ══════════════════════════════════════════════════════════════════

    def _detect_discrepancies(
        self, text: str, sins: dict[str, dict]
    ) -> list[str]:
        """Detect mismatches between linguistic content and sin scores.

        Implements social desirability bias detection:
        - Wrath reverse coding: claims low wrath but uses anger language
        - Pride discrepancy: claims humility but uses self-promotion language
        - Envy discrepancy: claims contentment but uses comparative language
        - Sloth discrepancy: claims proactivity but uses passive language
        - Cross-sin divergence: related sins with large deltas

        Parameters
        ----------
        text:
            The user's response text.
        sins:
            Dict mapping sin names to their score dicts.

        Returns
        -------
        list[str]
            Human-readable discrepancy descriptions.
        """
        discrepancies: list[str] = []
        text_lower = text.lower()

        # ── Wrath: claims low wrath but uses anger words ──────────────
        wrath_data = sins.get("wrath", {})
        wrath_score = wrath_data.get("score", 0.0)

        anger_indicators = [
            "angry", "furious", "hate", "rage", "infuriated", "livid",
            "outraged", "pissed", "annoyed", "irritated", "frustrated",
        ]
        anger_word_count = sum(
            1 for w in anger_indicators if w in text_lower
        )

        if wrath_score < -1.0 and anger_word_count >= 2:
            discrepancies.append(
                f"WRATH_LANGUAGE_MISMATCH: Score is {wrath_score:.1f} (low "
                f"wrath) but response contains {anger_word_count} anger-"
                f"related words"
            )

        # ── Pride: claims humility but self-promotes ──────────────────
        pride_data = sins.get("pride", {})
        pride_score = pride_data.get("score", 0.0)

        self_promotion_indicators = [
            "i'm the best", "better than", "i always",
            "everyone knows", "obviously i", "i'm great",
            "my achievement", "i accomplished", "i excelled",
            "top of my", "number one", "i won",
        ]
        self_promotion_count = sum(
            1 for phrase in self_promotion_indicators if phrase in text_lower
        )

        if pride_score < -1.0 and self_promotion_count >= 1:
            discrepancies.append(
                f"PRIDE_LANGUAGE_MISMATCH: Score is {pride_score:.1f} (low "
                f"pride / humble) but response contains self-promotional "
                f"language ({self_promotion_count} indicators)"
            )

        # ── Envy: claims contentment but uses comparative language ────
        envy_data = sins.get("envy", {})
        envy_score = envy_data.get("score", 0.0)

        comparison_indicators = [
            "wish i had", "they have", "not fair", "why them",
            "lucky them", "must be nice", "i deserve", "should be me",
            "better than me", "ahead of me",
        ]
        comparison_count = sum(
            1 for phrase in comparison_indicators if phrase in text_lower
        )

        if envy_score < -1.0 and comparison_count >= 1:
            discrepancies.append(
                f"ENVY_LANGUAGE_MISMATCH: Score is {envy_score:.1f} (low "
                f"envy / content) but response contains {comparison_count} "
                f"comparative/envious phrases"
            )

        # ── Sloth: claims proactivity but uses passive language ───────
        sloth_data = sins.get("sloth", {})
        sloth_score = sloth_data.get("score", 0.0)

        passive_indicators = [
            "i'll get to it", "eventually", "haven't gotten around",
            "too lazy", "can't be bothered", "maybe later",
            "i should but", "i keep putting off", "procrastinat",
        ]
        passive_count = sum(
            1 for phrase in passive_indicators if phrase in text_lower
        )

        if sloth_score < -1.0 and passive_count >= 1:
            discrepancies.append(
                f"SLOTH_LANGUAGE_MISMATCH: Score is {sloth_score:.1f} (low "
                f"sloth / proactive) but response contains {passive_count} "
                f"passive/avoidant phrases"
            )

        # ── Cross-response consistency: large delta between related sins
        related_pairs = [
            ("greed", "envy"),
            ("wrath", "pride"),
            ("lust", "gluttony"),
        ]
        for sin_a, sin_b in related_pairs:
            score_a = sins.get(sin_a, {}).get("score", 0.0)
            score_b = sins.get(sin_b, {}).get("score", 0.0)
            delta = abs(score_a - score_b)

            if delta > 5.0:
                discrepancies.append(
                    f"CROSS_SIN_DIVERGENCE: {sin_a} ({score_a:.1f}) and "
                    f"{sin_b} ({score_b:.1f}) differ by {delta:.1f} — "
                    f"unusual for typically correlated dimensions"
                )

        return discrepancies

    # ══════════════════════════════════════════════════════════════════
    # Evidence location
    # ══════════════════════════════════════════════════════════════════

    def _locate_evidence_in_response(
        self,
        response_text: str,
        evidence_quote: str,
    ) -> tuple[int, int]:
        """Find the character offsets of an evidence snippet within the
        original response text.

        Attempts three matching strategies in order:
        1. Exact substring match
        2. Case-insensitive match
        3. Longest common substring match (fuzzy fallback via subsequence)

        Parameters
        ----------
        response_text:
            The full user response.
        evidence_quote:
            The evidence snippet extracted by the LLM.

        Returns
        -------
        tuple[int, int]
            (start_index, end_index) character offsets, or (-1, -1) if
            no match is found.
        """
        if not evidence_quote or not response_text:
            return (-1, -1)

        # Strategy 1: Exact substring match
        idx = response_text.find(evidence_quote)
        if idx >= 0:
            return (idx, idx + len(evidence_quote))

        # Strategy 2: Case-insensitive match
        response_lower = response_text.lower()
        evidence_lower = evidence_quote.lower()
        idx = response_lower.find(evidence_lower)
        if idx >= 0:
            return (idx, idx + len(evidence_quote))

        # Strategy 3: Try progressively shorter substrings of the evidence
        # (sliding window from largest to smallest meaningful chunk)
        min_chunk_len = min(20, len(evidence_quote) // 2)
        if len(evidence_quote) >= min_chunk_len:
            for chunk_len in range(
                len(evidence_quote) - 1, min_chunk_len - 1, -1
            ):
                for start in range(len(evidence_quote) - chunk_len + 1):
                    chunk = evidence_lower[start : start + chunk_len]
                    idx = response_lower.find(chunk)
                    if idx >= 0:
                        return (idx, idx + chunk_len)

        logger.debug(
            "evidence_location_failed",
            evidence_preview=evidence_quote[:50],
            response_length=len(response_text),
        )
        return (-1, -1)

    # ══════════════════════════════════════════════════════════════════
    # Cross-response consistency checking
    # ══════════════════════════════════════════════════════════════════

    def _check_cross_response_consistency(
        self,
        per_question_results: list[dict],
    ) -> list[str]:
        """Check for inconsistencies across all question responses.

        Flags when the max delta for any sin across questions exceeds 5
        on the 11-point scale from -5 to +5.

        Parameters
        ----------
        per_question_results:
            List of parsed results from ``parse_single_response``.

        Returns
        -------
        list[str]
            Consistency flag strings.
        """
        flags: list[str] = []

        # Collect per-sin scores across all questions
        sin_scores: dict[str, list[float]] = {
            sin: [] for sin in self.SIN_NAMES
        }

        for result in per_question_results:
            for sin_name, sin_data in result.get("sins", {}).items():
                if sin_name in sin_scores:
                    sin_scores[sin_name].append(
                        sin_data.get("score", 0.0)
                    )

        # Check max delta for each sin
        for sin, scores in sin_scores.items():
            if len(scores) < 2:
                continue
            max_delta = max(scores) - min(scores)
            if max_delta > 5.0:
                flags.append(
                    f"consistency_delta:{sin}:max_delta={max_delta:.1f}"
                )

        return flags

    # ══════════════════════════════════════════════════════════════════
    # JSON response parsing
    # ══════════════════════════════════════════════════════════════════

    def _parse_json_response(self, text: str) -> dict:
        """Parse a JSON response from Gemini using multiple fallback
        strategies.

        Pipeline:
        1. Direct ``json.loads`` on the raw text
        2. Markdown code-fence extraction (triple-backtick json ... triple-backtick)
        3. Prefix/suffix stripping (remove leading/trailing non-JSON)
        4. ``jsonrepair`` library as a last resort

        Parameters
        ----------
        text:
            Raw text from the Gemini response.

        Returns
        -------
        dict
            The parsed JSON object.

        Raises
        ------
        ValueError
            If no strategy can extract valid JSON.
        """
        if not text or not text.strip():
            raise ValueError("Empty response text — cannot parse JSON")

        cleaned = text.strip()

        # Strategy 1: Direct parse
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        # Strategy 2: Markdown code-fence extraction
        md_pattern = re.compile(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
        )
        md_match = md_pattern.search(cleaned)
        if md_match:
            try:
                result = json.loads(md_match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 3: Prefix/suffix removal — find the first '{' and
        # last '}' to extract the JSON object
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            json_candidate = cleaned[first_brace : last_brace + 1]
            try:
                result = json.loads(json_candidate)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 4: jsonrepair library (best-effort)
        try:
            repaired = repair_json(cleaned)
            result = json.loads(repaired)
            if isinstance(result, dict):
                logger.info(
                    "json_parsed_via_jsonrepair",
                    original_preview=cleaned[:80],
                )
                return result
        except (json.JSONDecodeError, TypeError, Exception) as exc:
            logger.debug(
                "jsonrepair_failed",
                error=str(exc),
            )

        # Also try jsonrepair on the brace-extracted candidate
        if first_brace >= 0 and last_brace > first_brace:
            json_candidate = cleaned[first_brace : last_brace + 1]
            try:
                repaired = repair_json(json_candidate)
                result = json.loads(repaired)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, TypeError, Exception):
                pass

        raise ValueError(
            f"Failed to parse JSON from Gemini response. "
            f"Preview: {cleaned[:200]}"
        )

    # ══════════════════════════════════════════════════════════════════
    # Calibration example retrieval
    # ══════════════════════════════════════════════════════════════════

    async def _get_calibration_examples(
        self,
        sin: str,
        question_number: int,
    ) -> list[dict]:
        """Fetch validated few-shot calibration examples for a given sin
        and question.

        If no calibration service is configured, returns an empty list
        (the system degrades gracefully to zero-shot).

        Parameters
        ----------
        sin:
            The sin dimension.
        question_number:
            The ordinal question number (1-6).

        Returns
        -------
        list[dict]
            Up to 3 validated calibration examples.
        """
        if self._calibration_service is None:
            return []

        try:
            # Support multiple method signatures on the calibration service
            get_fn = getattr(
                self._calibration_service,
                "get_examples",
                getattr(
                    self._calibration_service,
                    "get_approved_examples",
                    None,
                ),
            )
            if get_fn is None:
                return []

            result = get_fn(
                sin=sin,
                question_number=question_number,
                limit=3,
                n=3,
            )

            # Handle both sync and async returns
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                result = await result

            return result if isinstance(result, list) else []

        except TypeError:
            # Retry with fewer kwargs if signature doesn't accept all
            try:
                get_fn = getattr(
                    self._calibration_service,
                    "get_examples",
                    getattr(
                        self._calibration_service,
                        "get_approved_examples",
                        None,
                    ),
                )
                if get_fn is None:
                    return []

                result = get_fn(
                    question_number=question_number,
                    sin=sin,
                )

                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    result = await result

                return result if isinstance(result, list) else []

            except Exception as inner_exc:
                logger.warning(
                    "calibration_fetch_failed",
                    sin=sin,
                    question_number=question_number,
                    error=str(inner_exc),
                )
                return []

        except Exception as exc:
            logger.warning(
                "calibration_fetch_failed",
                sin=sin,
                question_number=question_number,
                error=str(exc),
            )
            return []

    # ══════════════════════════════════════════════════════════════════
    # Evidence storage
    # ══════════════════════════════════════════════════════════════════

    async def store_parsing_evidence(
        self,
        db_session: Any,
        user_id: str,
        question_number: int,
        sins: dict[str, dict],
    ) -> list[Any]:
        """Persist per-sin parsing evidence to the database.

        Creates one ``ParsingEvidence`` row per sin for the given
        question, storing the score, confidence, evidence snippet,
        character offsets, and interpretation.

        Parameters
        ----------
        db_session:
            An active async SQLAlchemy session.
        user_id:
            The user's UUID string.
        question_number:
            The ordinal question number.
        sins:
            The per-sin result dict from ``parse_single_response``.

        Returns
        -------
        list
            The created ``ParsingEvidence`` ORM instances.
        """
        from app.models.evidence import ParsingEvidence

        evidence_records = []

        for sin_name, sin_data in sins.items():
            record = ParsingEvidence(
                user_id=uuid.UUID(user_id) if isinstance(user_id, str) else user_id,
                question_number=question_number,
                sin=sin_name,
                score=sin_data.get("score", 0.0),
                confidence=sin_data.get("confidence", 0.0),
                evidence_snippet=sin_data.get("evidence", ""),
                snippet_start_index=sin_data.get("evidence_start", -1),
                snippet_end_index=sin_data.get("evidence_end", -1),
                interpretation=sin_data.get("interpretation", ""),
                observer_persona=(
                    "multi_observer"
                    if sin_data.get("consensus_method") == "multi_observer"
                    else None
                ),
                gemini_model_used=sin_data.get("model_used", "unknown"),
            )
            db_session.add(record)
            evidence_records.append(record)

        try:
            await db_session.flush()
            logger.info(
                "parsing_evidence_stored",
                user_id=user_id,
                question_number=question_number,
                record_count=len(evidence_records),
            )
        except Exception as exc:
            logger.error(
                "parsing_evidence_storage_failed",
                user_id=user_id,
                question_number=question_number,
                error=str(exc),
            )
            raise

        return evidence_records
