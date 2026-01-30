import json
import logging
from typing import Tuple

from config import DEBUG_RAG, llm
from helpers import extract_json_object, safe_print
from models import QAItem
from rag_store import rag_context_for


def is_context_good(ctx: str) -> bool:
    if not ctx:
        return False
    return ("[Source:" in ctx) and (len(ctx) >= 220)


def is_grounded_answer(expected: str, key_points: list, ctx: str) -> bool:
    if not expected or not ctx:
        return False
    if expected not in ctx:
        return False
    for kp in (key_points or []):
        if kp and kp not in ctx:
            return False
    return True


def generate_question_from_context(tech: str, difficulty: int, rag_ctx: str) -> QAItem:
    prompt = f"""
Ты технический интервьюер. Сгенерируй ОДИН вопрос по технологии: {tech}.
Сложность: {difficulty}/5.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1) Используй ТОЛЬКО факты из CONTEXT ниже.
2) expected_answer ДОЛЖЕН быть ТОЧНОЙ подстрокой (exact substring) из CONTEXT (включая token., go/parser и т.д.).
3) key_points: каждый пункт тоже должен быть подстрокой из CONTEXT.
4) Если не получается — верни expected_answer пустым и сформулируй более простой вопрос по CONTEXT.

Верни ТОЛЬКО JSON:
{{
  "question": "...",
  "expected_answer": "...",
  "key_points": ["...", "..."],
  "topic": "..."
}}


CONTEXT:
{rag_ctx}
"""
    resp = llm.invoke([("human", prompt)])
    obj = extract_json_object(resp.content)
    try:
        data = json.loads(obj)
    except Exception:
        data = {
            "question": f"Расскажи базово про {tech}.",
            "expected_answer": "",
            "key_points": [],
            "topic": tech,
        }
    try:
        return QAItem(**data)
    except Exception:
        return QAItem(question=f"Расскажи базово про {tech}.", topic=tech)


def build_expected_for_question(tech: str, question: str, rag_ctx: str) -> QAItem:
    prompt = f"""
Ты технический ассистент. Твоя задача — выписать эталонный ответ из CONTEXT для заданного QUESTION.

ЖЁСТКИЕ ПРАВИЛА:
1) expected_answer ДОЛЖЕН быть ТОЧНОЙ подстрокой (exact substring) из CONTEXT (с теми же символами).
2) key_points: каждый пункт тоже должен быть ТОЧНОЙ подстрокой из CONTEXT.
3) Если в CONTEXT нет прямого ответа — верни expected_answer пустым и key_points [].
4) Верни ТОЛЬКО JSON.

TECH: {tech}
QUESTION: {question}

CONTEXT:
{rag_ctx}

Верни JSON:
{{
  "question": "{question}",
  "expected_answer": "...",
  "key_points": ["...", "..."],
  "topic": "..."
}}
"""
    resp = llm.invoke([("human", prompt)])
    obj = extract_json_object(resp.content)

    try:
        data = json.loads(obj)
    except Exception:
        return QAItem(question=question, expected_answer="", key_points=[], topic=tech)

    try:
        qa = QAItem(**data)
    except Exception:
        qa = QAItem(question=question, expected_answer="", key_points=[], topic=tech)

    if not is_grounded_answer(qa.expected_answer, qa.key_points, rag_ctx):
        return QAItem(question=question, expected_answer="", key_points=[], topic=qa.topic or tech)

    qa.question = question
    return qa


def ensure_expected_from_rag(tech: str, question: str, topic_seed: str) -> Tuple[QAItem, str]:
    """
    1) Ретривим rag_ctx по topic_seed/вопросу
    2) Пытаемся построить expected_answer/key_points как exact substring
    3) Если не вышло — пробуем альтернативный ретрив
    """
    rag_ctx = rag_context_for(topic_seed, tech=tech, k=4)
    if not is_context_good(rag_ctx):
        rag_ctx = rag_context_for(question, tech=tech, k=4)

    if not is_context_good(rag_ctx):
        return QAItem(question=question, expected_answer="", key_points=[], topic=tech), rag_ctx or ""

    qa = build_expected_for_question(tech, question, rag_ctx)
    if qa.expected_answer:
        return qa, rag_ctx

    rag_ctx2 = rag_context_for(f"{tech} {topic_seed} reference", tech=tech, k=4)
    if is_context_good(rag_ctx2):
        qa2 = build_expected_for_question(tech, question, rag_ctx2)
        if qa2.expected_answer:
            return qa2, rag_ctx2

    return qa, rag_ctx


def make_answerable_question(
    tech: str,
    difficulty: int,
    max_tries: int = 3,
    focus_topic: str = "",
) -> Tuple[QAItem, str]:
    """
    Generate a question that is answerable from current RAG context.
    """
    difficulty = max(1, min(5, difficulty))

    seed_queries = [
        f"{tech} introduction overview",
        f"{tech} basic concepts",
        f"{tech} api reference",
        f"{tech} examples tutorial",
        f"{tech} edge cases best practices",
    ]

    seed_query = (f"{tech} {focus_topic}".strip() if focus_topic else seed_queries[difficulty - 1])

    for attempt in range(1, max_tries + 1):
        rag_ctx = rag_context_for(seed_query, tech=tech, k=4)
        if not is_context_good(rag_ctx):
            rag_ctx = rag_context_for(tech, tech=tech, k=4)

        if not is_context_good(rag_ctx):
            seed_query = f"{seed_query} reference"
            continue

        qa = generate_question_from_context(tech, difficulty, rag_ctx)

        if not is_grounded_answer(qa.expected_answer, qa.key_points, rag_ctx):
            if qa.topic:
                seed_query = f"{tech} {qa.topic} reference"
            else:
                seed_query = f"{seed_query} reference"
            continue

        check_ctx = rag_context_for(qa.question, tech=tech, k=2)

        if is_context_good(check_ctx):
            return qa, check_ctx

        if qa.topic:
            seed_query = f"{tech} {qa.topic} reference"
        else:
            seed_query = f"{seed_query} reference"

    return (
        QAItem(
            question=f"Расскажи, что такое {tech}, где используется и какие основные понятия ты знаешь?",
            expected_answer="",
            key_points=[],
            topic=tech,
        ),
        "",
    )
