import json
from typing import Any, Dict, List, Optional

from config import llm
from helpers import extract_json_array, extract_json_object
from mcp_client import MCPServerClient


def generate_topics_for_tech(
    grade: str,
    position: str,
    tech: str,
    max_topics: int = 3,
) -> List[str]:
    prompt = f"""
Верни СТРОГО JSON-массив коротких keyword-тем для технологии "{tech}".
Темы должны соответствовать реальным заголовкам/терминам/пакетам в официальной документации DevDocs
и быть пригодными для использования как keyword в инструменте search_devdocs.
Количество тем не больше {max_topics}. Не добавляй пояснений — только JSON-массив строк.
Контекст: grade={grade}, position={position}, tech={tech}
Пример: ["slices","maps","fmt"]
"""
    try:
        resp = llm.invoke([("human", prompt)])
        arr_text = extract_json_array(resp.content)
        parsed = json.loads(arr_text)
        if isinstance(parsed, list):
            out: List[str] = []
            for item in parsed:
                if isinstance(item, str):
                    entry = item.strip()
                    if entry:
                        out.append(entry)
                        if len(out) >= max_topics:
                            break
            return out
    except Exception:
        pass
    basics = {
        "python": ["asyncio", "typing", "decorators"],
        "go": ["slices", "maps", "goroutines"],
        "javascript": ["promises", "async-await", "dom"],
    }
    return basics.get(tech, [tech, "introduction"])[:max_topics]


def generate_question_for_tech_topic(grade: str, position: str, tech: str, topic: str) -> str:
    prompt = f"""
Ты технический интервьюер. Сформулируй ОДИН конкретный вопрос для кандидата уровня {grade} на позицию {position}.
TECH: {tech}
TOPIC (keyword/area): {topic}
Требование: верни только JSON-объект {{"question": "..."}}. Вопрос должен быть кратким и прямо относиться к TOPIC.

Адаптивность (важно):
- Если кандидат уходит в пустые общие слова/бuzzword'ы, задай уточняющий вопрос о реальном опыте.
- Если ответ слишком общий или не по теме, попроси конкретику/пример.
- Вопрос всегда должен оставаться в сфере IT и релевантен позиции.
"""
    try:
        resp = llm.invoke([("human", prompt)])
        obj = extract_json_object(resp.content)
        parsed = json.loads(obj)
        question = parsed.get("question") if isinstance(parsed, dict) else None
        if isinstance(question, str) and question.strip():
            return question.strip()
    except Exception:
        pass
    return f"Объясните, что такое {topic} в контексте {tech} и где это обычно используется."


def generate_role_question(
    grade: str,
    position: str,
    last_question: str,
    last_answer: str,
) -> str:
    prompt = f"""
Ты HR/технический интервьюер в сфере IT. Сформулируй ОДИН следующий вопрос для кандидата.

Контекст:
- Должность: {position}
- Уровень: {grade}
- Последний вопрос: {last_question}
- Ответ кандидата: {last_answer}

Требования:
- Если кандидат не называет технологии, не требуй их и не проси перечислять стек.
  Спроси про роль, процессы, зоны ответственности, типичные задачи, кейсы, метрики
  или архитектурные решения в зависимости от позиции.
- Если кандидат уходит в общие слова/шум или отвечает слишком длинно, задай уточняющий и более конкретный вопрос.
- Если кандидат уверенно несет абсурд/путает термины, попроси объяснить на практике или привести пример.
- Вопрос должен быть кратким, один на выходе, и всегда оставаться в IT-сфере.

Верни только JSON:
{{"question": "..."}}
"""
    try:
        resp = llm.invoke([("human", prompt)])
        obj = extract_json_object(resp.content)
        parsed = json.loads(obj)
        question = parsed.get("question") if isinstance(parsed, dict) else None
        if isinstance(question, str) and question.strip():
            return question.strip()
    except Exception:
        pass
    return f"Расскажите про вашу роль {position} и ключевые задачи на последних проектах."


def generate_interview_plan(
    mcp: MCPServerClient,
    grade: str,
    position: str,
    techs: List[str],
    per_tech: int = 3,
) -> Dict[str, Any]:
    topics_map: Dict[str, List[str]] = {}
    questions_queue: List[Dict[str, str]] = []

    for tech in (techs or []):
        topics = generate_topics_for_tech(grade, position, tech, max_topics=per_tech)
        validated: List[str] = []
        for topic in topics:
            try:
                res = mcp.call_tool("search_devdocs", {"doc_name": tech, "keyword": topic})
            except Exception:
                res = None
            if res and isinstance(res, list) and len(res) > 0:
                validated.append(topic)
        if not validated:
            validated = [tech]
        topics_map[tech] = validated

        for topic in validated:
            question = generate_question_for_tech_topic(grade, position, tech, topic)
            questions_queue.append({"tech": tech, "topic": topic, "question": question})

    return {"topics_map": topics_map, "questions_queue": questions_queue}


def pop_next_question_for_tech(queue: List[Dict[str, str]], tech: str) -> Optional[Dict[str, str]]:
    if not queue:
        return None

    wanted = (tech or "").strip().lower()
    for i, item in enumerate(queue):
        item_tech = (item.get("tech") or "").strip().lower()
        if item_tech == wanted:
            return queue.pop(i)
    return None


def normalize_topic_seed(topic: str, tech: str) -> str:
    topic = (topic or "").strip()
    return topic if topic else tech
