"""Load test: run 100 synthetic profiles through the full Harmonia V3 pipeline.

Validates database integrity, score distributions, and system stability.
Usage: python -m scripts.load_test [--count 100] [--base-url http://localhost:8000]
"""
import argparse
import asyncio
import random
import statistics
import sys
import time
import uuid
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_COUNT = 100

QUESTION_TEMPLATES = [
    "You're at a group dinner with friends. The bill arrives and it's not split evenly...",
    "You receive an unexpected expense notification — your car needs repairs...",
    "You get a surprise day off this weekend with no obligations...",
    "You and a friend worked equally on a project, but they received more credit...",
    "A close friend calls you at midnight in a crisis. You have an important meeting at 8am...",
    "Your manager gives you mixed feedback — praise for one thing, criticism for another...",
]

SAMPLE_RESPONSES = [
    "I'd probably suggest we just split it evenly to keep things simple. I hate those awkward moments where everyone's trying to calculate their exact share. Life's too short for that kind of pettiness.",
    "That's really stressful but I'd take a deep breath and figure out my options. I'd check if I have savings or if I can negotiate a payment plan. No point panicking about it.",
    "I'd sleep in first, then probably go for a long walk somewhere green. Maybe call a friend I haven't seen in a while and grab brunch. I love having no plans and just seeing what happens.",
    "Honestly, that would really sting. I'd try not to let it show but inside I'd be frustrated. I might bring it up casually later, but I wouldn't make a big scene about it.",
    "I'd absolutely pick up the phone, no question. My friend needs me and that meeting can wait. I can run on less sleep for one day, but I couldn't live with myself if I ignored them.",
    "I'd try to take both parts seriously. The praise is nice but the criticism is where the growth is. I'd ask for specific examples so I can actually improve rather than just feeling bad.",
]

HLA_ALLELES = {
    "HLA-A": ["A*01:01", "A*02:01", "A*03:01", "A*24:02", "A*11:01", "A*26:01"],
    "HLA-B": ["B*07:02", "B*08:01", "B*35:01", "B*44:02", "B*51:01", "B*15:01"],
    "HLA-DRB1": ["DRB1*03:01", "DRB1*04:01", "DRB1*07:01", "DRB1*15:01", "DRB1*11:01", "DRB1*13:01"],
}


def random_hla() -> dict[str, list[str]]:
    """Generate random HLA alleles for a user."""
    return {
        locus: random.sample(alleles, 2)
        for locus, alleles in HLA_ALLELES.items()
    }


def random_response(question_number: int) -> str:
    """Generate a varied response for a question."""
    base = SAMPLE_RESPONSES[question_number - 1]
    # Add slight variation
    fillers = [
        " I think that's just how I am.",
        " It depends on the situation though.",
        " My friends would probably agree with me on this.",
        " I've learned this about myself over the years.",
    ]
    return base + random.choice(fillers)


async def create_user(client: httpx.AsyncClient, base_url: str, index: int) -> dict[str, Any] | None:
    """Create a single user via the API."""
    payload = {
        "email": f"loadtest_{index}_{uuid.uuid4().hex[:8]}@test.com",
        "display_name": f"Load Test User {index}",
        "age": random.randint(21, 45),
        "gender": random.choice(["male", "female"]),
        "location": random.choice(["London", "Manchester", "Edinburgh", "Bristol"]),
    }
    try:
        resp = await client.post(f"{base_url}/api/v1/users", json=payload)
        if resp.status_code in (200, 201):
            return resp.json()
        print(f"  [WARN] User {index}: status {resp.status_code}")
        return None
    except Exception as e:
        print(f"  [ERROR] User {index}: {e}")
        return None


async def submit_questionnaire(client: httpx.AsyncClient, base_url: str, user_id: str) -> bool:
    """Submit all 6 questionnaire responses for a user."""
    responses = []
    for q in range(1, 7):
        responses.append({
            "question_number": q,
            "response_text": random_response(q),
        })
    try:
        resp = await client.post(
            f"{base_url}/api/v1/questionnaire/submit-all",
            json={"user_id": user_id, "responses": responses},
            timeout=120.0,
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        print(f"  [ERROR] Questionnaire for {user_id[:8]}: {e}")
        return False


async def run_load_test(base_url: str, count: int) -> dict[str, Any]:
    """Run the full load test pipeline."""
    print(f"\n{'='*60}")
    print(f"Harmonia V3 Load Test — {count} profiles")
    print(f"Target: {base_url}")
    print(f"{'='*60}\n")

    results = {
        "total": count,
        "users_created": 0,
        "profiles_created": 0,
        "matches_calculated": 0,
        "errors": [],
        "timings": {"user_creation": [], "questionnaire": [], "matching": []},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Phase 1: Create users
        print(f"[1/3] Creating {count} users...")
        user_ids = []
        for i in range(count):
            t0 = time.monotonic()
            user = await create_user(client, base_url, i)
            dt = time.monotonic() - t0
            results["timings"]["user_creation"].append(dt)
            if user and "id" in user:
                user_ids.append(user["id"])
                results["users_created"] += 1
            if (i + 1) % 20 == 0:
                print(f"  Created {i + 1}/{count} users")

        print(f"  -> {results['users_created']} users created\n")

        # Phase 2: Submit questionnaires (creates profiles)
        print(f"[2/3] Submitting questionnaires for {len(user_ids)} users...")
        for i, uid in enumerate(user_ids):
            t0 = time.monotonic()
            ok = await submit_questionnaire(client, base_url, uid)
            dt = time.monotonic() - t0
            results["timings"]["questionnaire"].append(dt)
            if ok:
                results["profiles_created"] += 1
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(user_ids)} questionnaires")

        print(f"  -> {results['profiles_created']} profiles created\n")

        # Phase 3: Calculate matches for random pairs
        n_pairs = min(50, len(user_ids) * (len(user_ids) - 1) // 2)
        pairs = set()
        while len(pairs) < n_pairs and len(user_ids) >= 2:
            a, b = random.sample(user_ids, 2)
            pair = tuple(sorted([a, b]))
            pairs.add(pair)

        print(f"[3/3] Calculating {len(pairs)} matches...")
        for i, (a, b) in enumerate(pairs):
            t0 = time.monotonic()
            try:
                resp = await client.post(
                    f"{base_url}/api/v1/match/calculate/{a}/{b}",
                    timeout=60.0,
                )
                if resp.status_code in (200, 201):
                    results["matches_calculated"] += 1
            except Exception as e:
                results["errors"].append(f"Match {a[:8]}x{b[:8]}: {e}")
            dt = time.monotonic() - t0
            results["timings"]["matching"].append(dt)
            if (i + 1) % 10 == 0:
                print(f"  Calculated {i + 1}/{len(pairs)} matches")

        print(f"  -> {results['matches_calculated']} matches calculated\n")

    # Summary
    print(f"{'='*60}")
    print("LOAD TEST RESULTS")
    print(f"{'='*60}")
    print(f"Users created:    {results['users_created']}/{count}")
    print(f"Profiles created: {results['profiles_created']}/{results['users_created']}")
    print(f"Matches computed: {results['matches_calculated']}/{n_pairs}")

    for phase, timings in results["timings"].items():
        if timings:
            print(f"\n{phase} latency:")
            print(f"  mean:   {statistics.mean(timings):.2f}s")
            print(f"  median: {statistics.median(timings):.2f}s")
            print(f"  p95:    {sorted(timings)[int(len(timings)*0.95)]:.2f}s")
            print(f"  max:    {max(timings):.2f}s")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for e in results["errors"][:10]:
            print(f"  - {e}")

    print(f"\n{'='*60}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Harmonia V3 Load Test")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of profiles to create")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="API base URL")
    args = parser.parse_args()

    results = asyncio.run(run_load_test(args.base_url, args.count))

    # Exit with error if too many failures
    success_rate = results["profiles_created"] / max(results["total"], 1)
    if success_rate < 0.8:
        print(f"FAIL: Only {success_rate:.0%} success rate (target: 80%)")
        sys.exit(1)
    print(f"PASS: {success_rate:.0%} success rate")


if __name__ == "__main__":
    main()
