import json

from config import llm
from helpers import extract_json_object
from models import ObserverResult


def build_interviewer_visible_message(
    profile: dict,
    tech: str,
    next_question: str,
    observer: ObserverResult,
    user_message: str,
    last_question: str,
    history_summary: str = "",
) -> str:
    flags = observer.flags or {}
    assessment = observer.assessment or {}
    allowed_techs = profile.get("technologies", [])

    observer_hint = {
        "correctness": (assessment or {}).get("correctness", ""),
        "off_topic": bool(flags.get("off_topic")),
        "hallucination": bool(flags.get("hallucination_detected")),
        "candidate_question": bool(flags.get("candidate_question")),
        "missing_points": (assessment or {}).get("missing_points", [])[:3],
        "correct_answer_short": (assessment or {}).get("correct_answer_short", ""),
    }

    prompt = f"""
Ты — Interviewer (технический интервьюер) в IT. Позиция и уровень кандидата определяются профилем ниже.

Твоя задача — вести интервью и при необходимости ПЕРЕКЛЮЧАТЬ ТЕХНОЛОГИЮ, но только в рамках списка ALLOWED_TECHS.

АКТУАЛЬНЫЕ ВХОДНЫЕ ДАННЫЕ:
- ALLOWED_TECHS: {json.dumps(allowed_techs, ensure_ascii=False)}
- CURRENT_TECH: {tech}
- USER_MESSAGE: {user_message}
- LAST_QUESTION: {last_question}
- OBSERVER_HINT: {json.dumps(observer_hint, ensure_ascii=False)}
- HISTORY_SUMMARY: {history_summary}

ПРАВИЛА ПЕРЕКЛЮЧЕНИЯ (важно):
1) Если ALLOWED_TECHS пустой или CURRENT_TECH пустой — НЕ предлагай и НЕ делай переключений, next_tech должен быть "".
2) Если кандидат просит сменить технологию на конкретную (например "давай питон", "переключись на python") —
   установи next_tech в эту технологию (строго из ALLOWED_TECHS).
3) Если кандидат просит "другую технологию" без уточнения —
   установи next_tech в следующую технологию из ALLOWED_TECHS (циклически).
4) Если кандидат НЕ просит переключение — next_tech должен быть пустой строкой "".

АДАПТАЦИЯ ПОВЕДЕНИЯ (важно):
- Если кандидат резко отрицает современные практики (тесты, CI/CD, code review), уточни про риски, альтернативы и опыт инцидентов.
- Если ответ перегружен терминами без ясного смысла, попроси практический пример или объяснение простыми словами.
- Если ответ выглядит как вставка из источника/AI, попроси пересказать своими словами и кратко.
- Если ответ слишком длинный или уходит в сторону, вежливо попроси короткий ответ по сути.
- Если технологий нет (ALLOWED_TECHS пустой), фокусируйся на роли, процессах, ответственности, метриках и кейсах.


ФОРМАТ: верни ТОЛЬКО JSON:
{{
  "reaction": "1-2 предложения реакции",
  "next_tech": "python|go|...|\"\""
}}

ОГРАНИЧЕНИЕ:
- reaction не должен содержать несколько вопросов.
- Если next_tech не пустой — в reaction сначала подтвердить переключение.
"""

    try:
        resp = llm.invoke([("human", prompt)])
        raw = resp.content or ""
        obj_text = extract_json_object(raw)
        parsed = json.loads(obj_text) if obj_text else {}

        reaction = ""
        next_tech = ""
        if isinstance(parsed, dict):
            reaction = str(parsed.get("reaction", "")).strip()
            next_tech = str(parsed.get("next_tech", "")).strip().lower()

        marker = f"\n__NEXT_TECH__={next_tech}" if next_tech else "\n__NEXT_TECH__="
        if reaction:
            return reaction + marker + "\n\n" + next_question
        return marker + "\n\n" + next_question
    except Exception:
        return "\n__NEXT_TECH__=\n\n" + next_question
