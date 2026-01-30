import json

from config import llm
from helpers import extract_json_object, safe_print
from models import ObserverResult, QAItem

def observer_analyze(
    tech: str,
    last_agent_question: str,
    user_answer: str,
    qa: QAItem,
    rag_ctx: str,
) -> ObserverResult:
    prompt = f"""
Ты Observer/Critic. Оцени ответ кандидата на вопрос, опираясь на документацию (CONTEXT) и эталон (EXPECTED).

TECH: {tech}
QUESTION: {last_agent_question}

CANDIDATE ANSWER:
{user_answer}

EXPECTED (ground truth):
{qa.expected_answer}

KEY POINTS:
{qa.key_points}

CONTEXT (docs):
{rag_ctx}

Требования:
- Отметь, если кандидат уходит в сторону (off_topic).
- Отметь, если кандидат уверенно несет бред/ложные факты (hallucination_detected).
- Отметь, если кандидат вместо ответа задал вопрос интервьюеру (candidate_question).
- Если ответ кандидата НЕ отвечает на заданный вопрос (даже если по теме Go в целом) — поставь off_topic=true.
- Если EXPECTED и CONTEXT пустые (нет фактической базы), оценивай только релевантность вопросу и связность ответа,
  и не ставь hallucination_detected без явных противоречий здравому смыслу.

Верни ТОЛЬКО JSON:
{{
  "internal_thoughts": "...",
  "instruction_to_interviewer": "...",
  "topic_status": "continue | change | wrap_up",
  "difficulty_adjustment": "increase | decrease | maintain",
  "flags": {{
    "hallucination_detected": true/false,
    "off_topic": true/false,
    "candidate_question": true/false
  }},
  "assessment": {{
    "topic": "{qa.topic}",
    "correctness": "high | medium | low",
    "missing_points": ["..."],
    "correct_answer_short": "..."
  }}
}}

Рубрика оценки (очень важно):
- Если ответ кандидата СЕМАНТИЧЕСКИ совпадает с EXPECTED, но написан сокращённо/без префикса пакета — это считать correctness="high" (для Junior допускается "high"), missing_points может быть пустым или ["желательно указывать полный путь пакета go/parser"].
- off_topic=true ставь ТОЛЬКО если ответ явно не про вопрос (пример: "давай другую технологию", "не знаю", "поговорим о жизни").
  Сокращенный/неполный ответ по теме НЕ является off_topic.
- candidate_question=true ставь, если кандидат задаёт вопрос/просит сменить технологию вместо ответа.
- hallucination_detected=true ставь только если кандидат уверенно утверждает факт, противоречащий CONTEXT/EXPECTED.

Правила correctness:
- high: ответ по сути верный (включая допустимые сокращения), ключевой факт совпал
- medium: ответ частично верный/неточен, но направление правильное
- low: неверно или нет ответа по сути

"""
    resp = llm.invoke([("human", prompt)])
    raw = resp.content
    obj = extract_json_object(raw)

    try:
        data = json.loads(obj)
    except Exception:
        data = {
            "internal_thoughts": "Failed to parse analysis.",
            "instruction_to_interviewer": "Продолжай интервью и задай уточняющий вопрос.",
            "topic_status": "continue",
            "difficulty_adjustment": "maintain",
            "flags": {
                "hallucination_detected": False,
                "off_topic": False,
                "candidate_question": False,
            },
            "assessment": {
                "topic": qa.topic,
                "correctness": "medium",
                "missing_points": [],
                "correct_answer_short": "",
            },
        }

    try:
        return ObserverResult(**data)
    except Exception:
        return ObserverResult(
            internal_thoughts="Fallback analysis.",
            instruction_to_interviewer="Продолжай интервью.",
            topic_status="continue",
            difficulty_adjustment="maintain",
            flags={
                "hallucination_detected": False,
                "off_topic": False,
                "candidate_question": False,
            },
            assessment={
                "topic": qa.topic,
                "correctness": "medium",
                "missing_points": [],
                "correct_answer_short": "",
            },
        )
