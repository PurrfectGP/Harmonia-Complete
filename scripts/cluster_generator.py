#!/usr/bin/env python3
"""
Harmonia V3 — Cluster Generator: Synthetic Profile Generation via Claude Haiku

Generates synthetic personality profiles by instructing Claude Haiku to adopt
random personality archetypes and answer the six Felix questionnaire questions.
Generated responses are submitted to the PIIP pipeline for parsing, profiling,
and calibration-example ingestion.

Usage examples
--------------
  # Generate 10 profiles interactively:
  python scripts/cluster_generator.py --count 10

  # Generate 100 profiles using the Batch API:
  python scripts/cluster_generator.py --count 100 --batch

  # Verbose output with custom backend URL:
  python scripts/cluster_generator.py --count 5 --api-url http://localhost:8000 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from typing import Any

import anthropic
import httpx

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

HAIKU_MODEL = "claude-haiku-4-5"

FELIX_QUESTIONS: list[dict[str, Any]] = [
    {
        "question_number": 1,
        "question_text": (
            "You're at a group dinner with friends. The bill arrives and it's "
            "not split evenly — some people ordered much more than others. "
            "How do you handle it?"
        ),
        "category": "resource_conflict",
    },
    {
        "question_number": 2,
        "question_text": (
            "You receive an unexpected expense notification — your car needs "
            "repairs or your laptop dies. It's going to cost more than you'd "
            "like. How do you deal with it?"
        ),
        "category": "financial_stress",
    },
    {
        "question_number": 3,
        "question_text": (
            "You get a surprise day off this weekend with no obligations. "
            "How do you spend it?"
        ),
        "category": "leisure_autonomy",
    },
    {
        "question_number": 4,
        "question_text": (
            "You and a friend worked equally on a project, but they received "
            "more credit publicly. How do you react?"
        ),
        "category": "recognition_fairness",
    },
    {
        "question_number": 5,
        "question_text": (
            "A close friend calls you at midnight in a crisis. You have an "
            "important meeting at 8am tomorrow. What do you do?"
        ),
        "category": "loyalty_sacrifice",
    },
    {
        "question_number": 6,
        "question_text": (
            "Your manager gives you mixed feedback — praise for one thing, "
            "criticism for another. How do you process it?"
        ),
        "category": "authority_feedback",
    },
]

MIN_WORD_COUNT = 25
MAX_WORD_COUNT = 150

DEFAULT_API_URL = "http://localhost:8000"
SUBMIT_ENDPOINT = "/api/v1/questionnaire/submit"

MAX_RETRIES = 3
RETRY_DELAY_S = 2.0

# ──────────────────────────────────────────────────────────────────────────────
# System prompt for Claude Haiku
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a creative personality simulator. Your task is to generate realistic questionnaire responses as a fictional person.

INSTRUCTIONS:
1. Silently generate a random personality archetype for yourself. Do NOT reveal or label the archetype. Choose from a wide range: a cautious accountant, an impulsive artist, a resentful middle manager, a generous grandmother, a competitive athlete, a lazy student, an anxious perfectionist, etc.

2. Answer each of the 6 questions below as that person would, in 50-100 words each.

3. Show realistic MIXED motivations. Real people are contradictory — they can be generous in one area and selfish in another. Avoid one-dimensional responses.

4. Do NOT include any trait labels, personality type names, or meta-commentary about the character. Just answer naturally as that person would.

5. Vary your writing style across responses: some might be more formal, others casual; some reflective, others reactive.

6. Return ONLY a JSON object with keys "q1" through "q6", where each value is the response text. No other text or explanation.

QUESTIONS:

Q1: You're at a group dinner with friends. The bill arrives and it's not split evenly — some people ordered much more than others. How do you handle it?

Q2: You receive an unexpected expense notification — your car needs repairs or your laptop dies. It's going to cost more than you'd like. How do you deal with it?

Q3: You get a surprise day off this weekend with no obligations. How do you spend it?

Q4: You and a friend worked equally on a project, but they received more credit publicly. How do you react?

Q5: A close friend calls you at midnight in a crisis. You have an important meeting at 8am tomorrow. What do you do?

Q6: Your manager gives you mixed feedback — praise for one thing, criticism for another. How do you process it?

Return ONLY valid JSON in this exact format:
{
  "q1": "your 50-100 word response here",
  "q2": "your 50-100 word response here",
  "q3": "your 50-100 word response here",
  "q4": "your 50-100 word response here",
  "q5": "your 50-100 word response here",
  "q6": "your 50-100 word response here"
}"""

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("harmonia.cluster_generator")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_haiku_response(raw_text: str) -> dict[str, str]:
    """Parse Claude Haiku's JSON response with multiple fallback strategies.

    Returns a dict with keys q1-q6 mapping to response strings.

    Raises
    ------
    ValueError
        If the response cannot be parsed as valid JSON containing q1-q6.
    """
    text = raw_text.strip()

    # Strategy 1: direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            _validate_response_keys(data)
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: extract from markdown code fence
    import re
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        try:
            data = json.loads(md_match.group(1).strip())
            if isinstance(data, dict):
                _validate_response_keys(data)
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: brace extraction
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                _validate_response_keys(data)
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    raise ValueError(f"Failed to parse Haiku response as JSON: {text[:200]}")


def _validate_response_keys(data: dict) -> None:
    """Ensure the parsed dict has all q1-q6 keys."""
    required = {f"q{i}" for i in range(1, 7)}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing question keys: {sorted(missing)}")


# ──────────────────────────────────────────────────────────────────────────────
# Quality controls
# ──────────────────────────────────────────────────────────────────────────────

def _validate_word_counts(responses: dict[str, str]) -> list[str]:
    """Return a list of warnings for responses outside the word count range.

    Returns an empty list if all responses are within bounds.
    """
    warnings: list[str] = []
    for key in [f"q{i}" for i in range(1, 7)]:
        text = responses.get(key, "")
        word_count = len(text.split())
        if word_count < MIN_WORD_COUNT:
            warnings.append(
                f"{key}: too short ({word_count} words, min {MIN_WORD_COUNT})"
            )
        elif word_count > MAX_WORD_COUNT:
            warnings.append(
                f"{key}: too long ({word_count} words, max {MAX_WORD_COUNT})"
            )
    return warnings


def _check_diversity(
    responses: dict[str, str],
    previous_responses: list[dict[str, str]],
) -> bool:
    """Basic diversity check: ensure this response set is not too similar
    to any previous one.

    Uses a simple Jaccard similarity on word sets.  Returns True if the
    responses are sufficiently diverse.
    """
    if not previous_responses:
        return True

    # Build word set from the new responses
    new_words = set()
    for text in responses.values():
        new_words.update(text.lower().split())

    for prev in previous_responses[-20:]:  # Compare against last 20
        prev_words = set()
        for text in prev.values():
            prev_words.update(text.lower().split())

        if not new_words or not prev_words:
            continue

        intersection = len(new_words & prev_words)
        union = len(new_words | prev_words)
        similarity = intersection / union if union else 0.0

        if similarity > 0.7:
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Single profile generation (interactive API)
# ──────────────────────────────────────────────────────────────────────────────

async def generate_single_profile(
    client: anthropic.AsyncAnthropic,
    http_client: httpx.AsyncClient,
    api_url: str,
    previous_responses: list[dict[str, str]],
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate a single synthetic profile via Claude Haiku and submit it
    to the PIIP pipeline.

    Parameters
    ----------
    client:
        The Anthropic async client.
    http_client:
        An httpx async client for backend API calls.
    api_url:
        Base URL of the Harmonia backend.
    previous_responses:
        List of previously generated response sets (for diversity checking).
    verbose:
        If True, print detailed output.

    Returns
    -------
    dict
        Result containing the synthetic user_id, responses, word counts,
        and pipeline submission result.
    """
    synthetic_user_id = str(uuid.uuid4())
    start_time = time.monotonic()

    # ── Call Claude Haiku ─────────────────────────────────────────────
    responses = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = await client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Generate a complete set of questionnaire responses "
                            "as described in your instructions. Return ONLY "
                            "the JSON object."
                        ),
                    }
                ],
            )

            raw_text = message.content[0].text
            responses = _parse_haiku_response(raw_text)

            # Validate word counts
            word_warnings = _validate_word_counts(responses)
            if word_warnings:
                logger.warning(
                    "Word count warnings (attempt %d/%d): %s",
                    attempt, MAX_RETRIES, "; ".join(word_warnings),
                )
                # Retry if too many responses are out of range
                out_of_range = len(word_warnings)
                if out_of_range >= 3 and attempt < MAX_RETRIES:
                    logger.info("Too many out-of-range responses, retrying...")
                    responses = None
                    continue

            # Diversity check
            if not _check_diversity(responses, previous_responses):
                logger.warning(
                    "Diversity check failed (attempt %d/%d), retrying...",
                    attempt, MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
                    responses = None
                    continue

            break  # Success

        except (anthropic.APIError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Generation attempt %d/%d failed: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_S * attempt)
            continue

    if responses is None:
        raise RuntimeError(
            f"Failed to generate valid profile after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ── Compute word counts ───────────────────────────────────────────
    word_counts = {
        key: len(responses[key].split())
        for key in [f"q{i}" for i in range(1, 7)]
    }

    # ── Submit to PIIP pipeline ───────────────────────────────────────
    submission_payload = {
        "user_id": synthetic_user_id,
        "source": "claude_agent",
        "responses": [
            {
                "question_number": q["question_number"],
                "question_text": q["question_text"],
                "response_text": responses[f"q{q['question_number']}"],
            }
            for q in FELIX_QUESTIONS
        ],
    }

    pipeline_result = None
    try:
        resp = await http_client.post(
            f"{api_url}{SUBMIT_ENDPOINT}",
            json=submission_payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        pipeline_result = resp.json()
    except httpx.HTTPError as exc:
        logger.error(
            "Pipeline submission failed for %s: %s",
            synthetic_user_id, exc,
        )
        pipeline_result = {"error": str(exc)}

    elapsed_s = round(time.monotonic() - start_time, 2)

    result = {
        "user_id": synthetic_user_id,
        "source": "claude_agent",
        "responses": responses,
        "word_counts": word_counts,
        "pipeline_result": pipeline_result,
        "elapsed_s": elapsed_s,
        "model": HAIKU_MODEL,
    }

    logger.info(
        "Profile generated: user_id=%s, elapsed=%.2fs, words=%s",
        synthetic_user_id, elapsed_s,
        "/".join(str(word_counts[f"q{i}"]) for i in range(1, 7)),
    )

    if verbose:
        for key in [f"q{i}" for i in range(1, 7)]:
            print(f"  {key} ({word_counts[key]} words): {responses[key][:80]}...")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Batch API support
# ──────────────────────────────────────────────────────────────────────────────

def build_batch_requests(count: int) -> list[dict[str, Any]]:
    """Build a list of Anthropic Message Batch request dicts for high-volume
    generation.

    Each request generates one synthetic profile.  The caller should submit
    these to the Anthropic Message Batches API endpoint.

    Parameters
    ----------
    count:
        Number of profiles to generate.

    Returns
    -------
    list[dict]
        Each dict is a valid request body for the Message Batches API,
        containing a unique ``custom_id`` for tracking.
    """
    requests = []
    for i in range(count):
        custom_id = f"harmonia-cluster-{uuid.uuid4().hex[:12]}-{i:04d}"
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": HAIKU_MODEL,
                "max_tokens": 2048,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Generate a complete set of questionnaire responses "
                            "as described in your instructions. Return ONLY "
                            "the JSON object."
                        ),
                    }
                ],
            },
        })
    return requests


async def submit_batch(
    client: anthropic.AsyncAnthropic,
    count: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Submit a batch of generation requests to the Anthropic Message
    Batches API.

    Parameters
    ----------
    client:
        The Anthropic async client.
    count:
        Number of profiles to generate.
    verbose:
        If True, print detailed output.

    Returns
    -------
    dict
        The batch creation response from the API, including the batch ID
        for polling status.
    """
    requests = build_batch_requests(count)

    logger.info("Submitting batch of %d generation requests...", count)

    batch = await client.messages.batches.create(requests=requests)

    result = {
        "batch_id": batch.id,
        "request_count": count,
        "status": batch.processing_status,
        "created_at": batch.created_at.isoformat() if hasattr(batch, "created_at") else None,
    }

    logger.info(
        "Batch submitted: id=%s, count=%d, status=%s",
        batch.id, count, batch.processing_status,
    )

    if verbose:
        print(f"\nBatch submitted successfully:")
        print(f"  Batch ID: {batch.id}")
        print(f"  Requests: {count}")
        print(f"  Status:   {batch.processing_status}")
        print(f"\nPoll status with:")
        print(f"  python -c \"import anthropic; "
              f"print(anthropic.Anthropic().messages.batches.retrieve('{batch.id}'))\"")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Generation loop
# ──────────────────────────────────────────────────────────────────────────────

async def run_generation_loop(
    count: int,
    api_url: str = DEFAULT_API_URL,
    verbose: bool = False,
) -> dict[str, Any]:
    """Generate *count* synthetic profiles sequentially, submitting each to
    the PIIP pipeline.

    Parameters
    ----------
    count:
        Number of profiles to generate.
    api_url:
        Base URL of the Harmonia backend.
    verbose:
        If True, print detailed per-profile output.

    Returns
    -------
    dict
        Summary including total generated, succeeded, failed, elapsed time,
        and estimated cost.
    """
    client = anthropic.AsyncAnthropic()
    previous_responses: list[dict[str, str]] = []
    results: list[dict] = []
    succeeded = 0
    failed = 0

    start_time = time.monotonic()

    async with httpx.AsyncClient() as http_client:
        for i in range(count):
            logger.info("Generating profile %d/%d...", i + 1, count)

            try:
                result = await generate_single_profile(
                    client=client,
                    http_client=http_client,
                    api_url=api_url,
                    previous_responses=previous_responses,
                    verbose=verbose,
                )
                results.append(result)
                previous_responses.append(result["responses"])
                succeeded += 1

            except Exception as exc:
                logger.error(
                    "Profile %d/%d generation failed: %s",
                    i + 1, count, exc,
                )
                results.append({"error": str(exc), "index": i})
                failed += 1

            # Brief pause between generations to respect rate limits
            if i < count - 1:
                await asyncio.sleep(0.5)

    total_elapsed = round(time.monotonic() - start_time, 2)

    # Estimate cost: Claude Haiku ~$0.25/MTok input, ~$1.25/MTok output
    # Rough estimate: ~1500 tokens input, ~800 tokens output per profile
    estimated_input_tokens = count * 1500
    estimated_output_tokens = count * 800
    estimated_cost = (
        (estimated_input_tokens / 1_000_000) * 0.25
        + (estimated_output_tokens / 1_000_000) * 1.25
    )

    summary = {
        "total_requested": count,
        "succeeded": succeeded,
        "failed": failed,
        "total_elapsed_s": total_elapsed,
        "avg_per_profile_s": round(total_elapsed / count, 2) if count else 0,
        "estimated_cost_usd": round(estimated_cost, 4),
        "model": HAIKU_MODEL,
    }

    logger.info(
        "Generation complete: %d/%d succeeded in %.1fs (est. $%.4f)",
        succeeded, count, total_elapsed, estimated_cost,
    )

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonia Cluster Generator — generate synthetic personality "
            "profiles via Claude Haiku for calibration data."
        ),
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of profiles to generate (default: 10).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help=(
            "Use the Anthropic Message Batches API for high-volume "
            "generation instead of sequential calls."
        ),
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Backend API URL (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output with per-profile details.",
    )

    args = parser.parse_args()

    _configure_logging(args.verbose)

    if args.count < 1:
        parser.error("--count must be at least 1")

    print(f"\n{'=' * 60}")
    print(f"  Harmonia Cluster Generator")
    print(f"  Model: {HAIKU_MODEL}")
    print(f"  Profiles: {args.count}")
    print(f"  Mode: {'Batch API' if args.batch else 'Sequential'}")
    print(f"  Backend: {args.api_url}")
    print(f"{'=' * 60}\n")

    if args.batch:
        result = asyncio.run(
            submit_batch(
                client=anthropic.AsyncAnthropic(),
                count=args.count,
                verbose=args.verbose,
            )
        )
    else:
        result = asyncio.run(
            run_generation_loop(
                count=args.count,
                api_url=args.api_url,
                verbose=args.verbose,
            )
        )

    print(f"\n{'=' * 60}")
    print(f"  Results:")
    for key, value in result.items():
        print(f"    {key}: {value}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
