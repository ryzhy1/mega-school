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


def debug_block(title: str, payload: str, limit: int = 900):
    if not DEBUG_RAG:
        return
    safe_print("\n" + "═" * 80)
    safe_print(f"🔎 DEBUG: {title}")
    safe_print("─" * 80)
    if payload is None:
        safe_print("(None)")
    else:
        payload = str(payload)
        if len(payload) > limit:
            safe_print(payload[:limit] + f"\n...[trimmed {len(payload)-limit} chars]")
        else:
            safe_print(payload)
    safe_print("═" * 80 + "\n")


def generate_question_from_context(tech: str, difficulty: int, rag_ctx: str) -> QAItem:
    prompt = f"""
You are a technical interviewer. Generate ONE question about the technology: {tech}.
Difficulty: {difficulty}/5.

MANDATORY RULES:
1) Use ONLY facts from the CONTEXT below.
2) expected_answer MUST be an exact substring of CONTEXT (including token., go/parser, etc.).
3) key_points: each item must also be a substring of CONTEXT.
4) If this is not possible — return an empty expected_answer and ask a simpler question based on CONTEXT.

Return ONLY JSON:
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
You are a technical assistant. Your task is to extract the expected answer from CONTEXT for the given QUESTION.

STRICT RULES:
1) expected_answer MUST be an exact substring of CONTEXT (with the same characters).
2) key_points: each item must also be an exact substring of CONTEXT.
3) If CONTEXT does not contain a direct answer — return an empty expected_answer and key_points [].
4) Return ONLY JSON.

TECH: {tech}
QUESTION: {question}

CONTEXT:
{rag_ctx}

Return JSON:
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

        debug_block(
            f"QuestionGen attempt={attempt} tech={tech} difficulty={difficulty} seed_query='{seed_query}' "
            f"(RAG ctx used for generation)",
            rag_ctx,
        )

        if not is_context_good(rag_ctx):
            debug_block("QuestionGen: RAG ctx is NOT good (will retry)", f"len={len(rag_ctx)}")
            seed_query = f"{seed_query} reference"
            continue

        qa = generate_question_from_context(tech, difficulty, rag_ctx)

        if not is_grounded_answer(qa.expected_answer, qa.key_points, rag_ctx):
            debug_block(
                "❌ NOT GROUNDED: expected_answer/key_points not in generation ctx",
                f"expected={qa.expected_answer}\nkey_points={qa.key_points}",
            )
            if qa.topic:
                seed_query = f"{tech} {qa.topic} reference"
            else:
                seed_query = f"{seed_query} reference"
            continue

        check_ctx = rag_context_for(qa.question, tech=tech, k=2)
        debug_block("Validation RAG ctx (retrieved by the question)", check_ctx)
        debug_block("Is answerable by RAG?", str(is_context_good(check_ctx)))

        if is_context_good(check_ctx):
            return qa, check_ctx

        if qa.topic:
            seed_query = f"{tech} {qa.topic} reference"
        else:
            seed_query = f"{seed_query} reference"

    debug_block("QuestionGen fallback", f"tech={tech} difficulty={difficulty} -> generic question (no RAG)")
    return (
        QAItem(
            question=f"Расскажи, что такое {tech}, где используется и какие основные понятия ты знаешь?",
            expected_answer="",
            key_points=[],
            topic=tech,
        ),
        "",
    )
