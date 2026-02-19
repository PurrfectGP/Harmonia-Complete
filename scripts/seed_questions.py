"""Seed the 6 Felix questions into the questions reference table."""
import asyncio
import sys
sys.path.insert(0, ".")

from sqlalchemy import select
from app.database import async_session_factory, engine
from app.models.questionnaire import Question


FELIX_QUESTIONS = [
    {
        "question_number": 1,
        "question_text": (
            "You're at a group dinner with friends. The bill arrives and it's not split evenly — "
            "some people ordered much more than others. How do you handle it?"
        ),
        "category": "resource_conflict",
    },
    {
        "question_number": 2,
        "question_text": (
            "You receive an unexpected expense notification — your car needs repairs or your laptop dies. "
            "It's going to cost more than you'd like. How do you deal with it?"
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
            "You and a friend worked equally on a project, but they received more credit publicly. "
            "How do you react?"
        ),
        "category": "recognition_fairness",
    },
    {
        "question_number": 5,
        "question_text": (
            "A close friend calls you at midnight in a crisis. "
            "You have an important meeting at 8am tomorrow. What do you do?"
        ),
        "category": "loyalty_sacrifice",
    },
    {
        "question_number": 6,
        "question_text": (
            "Your manager gives you mixed feedback — praise for one thing, criticism for another. "
            "How do you process it?"
        ),
        "category": "authority_feedback",
    },
]


async def seed():
    async with async_session_factory() as session:
        for q in FELIX_QUESTIONS:
            existing = await session.execute(
                select(Question).where(Question.question_number == q["question_number"])
            )
            if existing.scalar_one_or_none() is None:
                session.add(Question(**q))
                print(f"  Seeded question {q['question_number']}: {q['category']}")
            else:
                print(f"  Question {q['question_number']} already exists, skipping.")
        await session.commit()
    print("Done seeding questions.")


if __name__ == "__main__":
    asyncio.run(seed())
