#!/usr/bin/env python3
"""
Harmonia V3 — Calibration Manager: Statistics, Coverage, and Generation CLI

Management script for the calibration programme.  Provides three subcommands:

  stats     — Report calibration-example counts and review statistics.
  coverage  — Display the 6x7 calibration coverage map and identify gaps.
  generate  — Invoke the cluster generator to fill coverage gaps.

Usage examples
--------------
  # Show overall calibration stats
  python scripts/calibration_manager.py stats

  # Show the coverage map with gap identification
  python scripts/calibration_manager.py coverage

  # Generate 20 profiles to fill gaps
  python scripts/calibration_manager.py generate --count 20

  # Generate with verbose output
  python scripts/calibration_manager.py generate --count 5 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

# Ensure the project root is importable
sys.path.insert(0, ".")

from app.database import async_session_factory
from app.services.calibration_service import (
    CalibrationService,
    QUESTION_NUMBERS,
    SIN_NAMES,
    TARGET_EXAMPLES_PER_CELL,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Estimated cost per profile (Claude Haiku generation + Gemini parsing)
# Haiku: ~$0.25/MTok in + $1.25/MTok out, ~1500 in + ~800 out per profile
# Gemini: ~$0.075/1K chars, ~6 questions x 7 sins x ~500 chars per call
ESTIMATED_HAIKU_COST_PER_PROFILE = (1500 / 1_000_000) * 0.25 + (800 / 1_000_000) * 1.25
ESTIMATED_GEMINI_COST_PER_PROFILE = 6 * 7 * 0.500 * 0.000075  # 6 questions x 7 sins x 500 chars
ESTIMATED_TOTAL_COST_PER_PROFILE = (
    ESTIMATED_HAIKU_COST_PER_PROFILE + ESTIMATED_GEMINI_COST_PER_PROFILE
)


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand: stats
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_stats(args: argparse.Namespace) -> None:
    """Report calibration-example counts, review statistics, and
    effectiveness metrics."""
    service = CalibrationService()

    async with async_session_factory() as session:
        stats = await service.get_calibration_stats(db_session=session)
        metrics = await service.get_effectiveness_metrics(db_session=session)

    # ── Overall counts ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Calibration Statistics")
    print(f"{'=' * 60}")
    print(f"  Total examples:    {stats['total']}")
    print(f"  Pending:           {stats['pending']}")
    print(f"  Approved:          {stats['approved']}")
    print(f"  Corrected:         {stats['corrected']}")
    print(f"  Rejected:          {stats['rejected']}")

    # ── Coverage summary ──────────────────────────────────────────────
    cov = stats["coverage_summary"]
    print(f"\n  Coverage:")
    print(f"    Cells with data:  {cov['covered_cells']}/{cov['total_cells']} ({cov['coverage_pct']}%)")
    print(f"    Fully covered:    {cov['fully_covered_cells']}/{cov['total_cells']} ({cov['full_coverage_pct']}%)")
    print(f"    Target per cell:  {TARGET_EXAMPLES_PER_CELL}")

    # ── Correction metrics ────────────────────────────────────────────
    print(f"\n  Correction Metrics:")
    print(f"    Avg correction magnitude: {metrics['avg_correction_magnitude']}")
    print(f"    Total corrections:        {metrics['total_corrections']}")

    if metrics.get("correction_magnitude_by_sin"):
        print(f"\n    Per-sin correction magnitude:")
        for sin_name, data in sorted(metrics["correction_magnitude_by_sin"].items()):
            print(f"      {sin_name:<10} avg={data['avg_magnitude']:.3f} (n={data['count']})")

    # ── Drift report ──────────────────────────────────────────────────
    drift = metrics.get("drift_report", {})
    if drift.get("drift_direction") != "insufficient_data":
        print(f"\n  Drift Report:")
        print(f"    Early avg magnitude:  {drift['early_avg_magnitude']}")
        print(f"    Recent avg magnitude: {drift['recent_avg_magnitude']}")
        print(f"    Direction:            {drift['drift_direction']}")
        print(f"    Drift magnitude:      {drift['drift_magnitude']}")
    else:
        print(f"\n  Drift Report: insufficient data (need 4+ corrections)")

    # ── By-question breakdown ─────────────────────────────────────────
    if stats.get("by_question"):
        print(f"\n  By Question:")
        for qn in sorted(stats["by_question"].keys()):
            breakdown = stats["by_question"][qn]
            parts = [
                f"{status}={count}"
                for status, count in sorted(breakdown.items())
            ]
            print(f"    Q{qn}: {', '.join(parts)}")

    # ── By-sin breakdown ──────────────────────────────────────────────
    if stats.get("by_sin"):
        print(f"\n  By Sin:")
        for sin_name in SIN_NAMES:
            if sin_name in stats["by_sin"]:
                breakdown = stats["by_sin"][sin_name]
                parts = [
                    f"{status}={count}"
                    for status, count in sorted(breakdown.items())
                ]
                print(f"    {sin_name:<10} {', '.join(parts)}")

    # ── Cost estimate ─────────────────────────────────────────────────
    target_total = cov["total_cells"] * TARGET_EXAMPLES_PER_CELL
    remaining = max(0, target_total - (stats["approved"] + stats["corrected"]))
    estimated_cost = remaining * ESTIMATED_TOTAL_COST_PER_PROFILE

    print(f"\n  Cost Estimate:")
    print(f"    Target total examples:   {target_total}")
    print(f"    Validated so far:        {stats['approved'] + stats['corrected']}")
    print(f"    Remaining to generate:   {remaining}")
    print(f"    Est. cost per profile:   ${ESTIMATED_TOTAL_COST_PER_PROFILE:.4f}")
    print(f"    Est. remaining cost:     ${estimated_cost:.2f}")

    print(f"{'=' * 60}\n")

    if args.json:
        combined = {"stats": stats, "metrics": metrics}
        print(json.dumps(combined, indent=2, default=str))


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand: coverage
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_coverage(args: argparse.Namespace) -> None:
    """Display the 6x7 calibration coverage map and identify gaps."""
    service = CalibrationService()

    async with async_session_factory() as session:
        stats = await service.get_calibration_stats(db_session=session)

    coverage_map = stats["coverage_map"]

    # ── Render coverage grid ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Calibration Coverage Map  (target: {TARGET_EXAMPLES_PER_CELL} per cell)")
    print(f"{'=' * 70}")

    # Header row
    header = "         "
    for sin_name in SIN_NAMES:
        header += f" {sin_name[:7]:>7}"
    print(header)
    print("  " + "-" * (9 + 8 * len(SIN_NAMES)))

    # Data rows
    gaps: list[tuple[int, str, int]] = []

    for qn in QUESTION_NUMBERS:
        q_key = f"q{qn}"
        row = f"    Q{qn}  |"

        for sin_name in SIN_NAMES:
            cell = coverage_map.get(q_key, {}).get(sin_name, {})
            count = cell.get("count", 0)
            target = cell.get("target", TARGET_EXAMPLES_PER_CELL)

            if count >= target:
                indicator = f"  {count:>2} ok "
            elif count > 0:
                indicator = f"  {count:>2}/{target} "
                gaps.append((qn, sin_name, target - count))
            else:
                indicator = f"  --/{target} "
                gaps.append((qn, sin_name, target))

            row += indicator

        print(row)

    print("  " + "-" * (9 + 8 * len(SIN_NAMES)))

    # ── Gap report ────────────────────────────────────────────────────
    if gaps:
        print(f"\n  Gaps ({len(gaps)} cells below target):")
        total_needed = 0
        for qn, sin_name, needed in sorted(gaps, key=lambda x: (-x[2], x[0], x[1])):
            print(f"    Q{qn} x {sin_name:<10} needs {needed} more example(s)")
            total_needed += needed

        print(f"\n  Total examples needed to fill all gaps: {total_needed}")
        # Each profile generates 1 example per (question, sin) when parsed,
        # but only if it contains signal for that sin.  Rough estimate: each
        # profile covers ~60-80% of cells on average.
        estimated_profiles = max(1, int(total_needed / (len(SIN_NAMES) * 0.7)))
        estimated_cost = estimated_profiles * ESTIMATED_TOTAL_COST_PER_PROFILE
        print(f"  Estimated profiles to generate: ~{estimated_profiles}")
        print(f"  Estimated cost: ~${estimated_cost:.2f}")
    else:
        print(f"\n  All cells meet the target of {TARGET_EXAMPLES_PER_CELL} examples!")

    # ── Summary ───────────────────────────────────────────────────────
    cov = stats["coverage_summary"]
    print(f"\n  Summary:")
    print(f"    Cells with data:  {cov['covered_cells']}/{cov['total_cells']} ({cov['coverage_pct']}%)")
    print(f"    Fully covered:    {cov['fully_covered_cells']}/{cov['total_cells']} ({cov['full_coverage_pct']}%)")
    print(f"{'=' * 70}\n")

    if args.json:
        print(json.dumps(stats["coverage_map"], indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand: generate
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_generate(args: argparse.Namespace) -> None:
    """Invoke the cluster generator to create synthetic profiles."""
    # Import here to keep the dependency optional for stats/coverage commands
    from scripts.cluster_generator import (
        run_generation_loop,
        submit_batch,
        HAIKU_MODEL,
    )
    import anthropic as anthropic_mod

    print(f"\n{'=' * 60}")
    print(f"  Cluster Generator")
    print(f"  Model: {HAIKU_MODEL}")
    print(f"  Profiles: {args.count}")
    print(f"  Mode: {'Batch API' if args.batch else 'Sequential'}")
    print(f"  Backend: {args.api_url}")
    print(f"{'=' * 60}\n")

    if args.batch:
        client = anthropic_mod.AsyncAnthropic()
        result = await submit_batch(
            client=client,
            count=args.count,
            verbose=args.verbose,
        )
    else:
        result = await run_generation_loop(
            count=args.count,
            api_url=args.api_url,
            verbose=args.verbose,
        )

    print(f"\n{'=' * 60}")
    print(f"  Generation Results:")
    for key, value in result.items():
        print(f"    {key}: {value}")
    print(f"{'=' * 60}\n")

    # Show updated stats if not in batch mode
    if not args.batch:
        print("Fetching updated calibration stats...\n")
        try:
            service = CalibrationService()
            async with async_session_factory() as session:
                stats = await service.get_calibration_stats(db_session=session)

            cov = stats["coverage_summary"]
            print(f"  Updated Coverage:")
            print(f"    Cells with data:  {cov['covered_cells']}/{cov['total_cells']} ({cov['coverage_pct']}%)")
            print(f"    Fully covered:    {cov['fully_covered_cells']}/{cov['total_cells']} ({cov['full_coverage_pct']}%)")
            print()
        except Exception as exc:
            print(f"  (Could not fetch updated stats: {exc})\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonia Calibration Manager — statistics, coverage maps, "
            "and synthetic profile generation."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available subcommands",
    )

    # ── stats ─────────────────────────────────────────────────────────
    stats_parser = subparsers.add_parser(
        "stats",
        help="Report calibration-example counts and review statistics.",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Also output raw JSON data.",
    )

    # ── coverage ──────────────────────────────────────────────────────
    coverage_parser = subparsers.add_parser(
        "coverage",
        help="Display the 6x7 calibration coverage map and identify gaps.",
    )
    coverage_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Also output raw JSON coverage map.",
    )

    # ── generate ──────────────────────────────────────────────────────
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic profiles to fill calibration gaps.",
    )
    generate_parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of profiles to generate (default: 10).",
    )
    generate_parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Use the Anthropic Message Batches API for high-volume generation.",
    )
    generate_parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000).",
    )
    generate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output with per-profile details.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "stats":
        asyncio.run(cmd_stats(args))
    elif args.command == "coverage":
        asyncio.run(cmd_coverage(args))
    elif args.command == "generate":
        asyncio.run(cmd_generate(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
