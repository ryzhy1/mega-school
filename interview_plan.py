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
Return a STRICT JSON array of short keyword topics for the technology "{tech}".
Topics must match real headings/terms/packages in official DevDocs documentation
and be suitable as keywords for the search_devdocs tool.
The number of topics must be at most {max_topics}. Do not add explanations — only a JSON array of strings.
Context: grade={grade}, position={position}, tech={tech}
Example: ["slices","maps","fmt"]
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
You are a technical interviewer. Ask ONE specific question for a {grade}-level candidate applying for {position}.
TECH: {tech}
TOPIC (keyword/area): {topic}
Requirement: return only a JSON object {{"question": "..."}}. The question must be concise and directly related to TOPIC.

Adaptation (important):
- If the candidate drifts into empty generalities/buzzword salad, ask a clarifying question about real experience.
- If the answer is too generic or off-topic, ask for specifics/examples.
- The question must stay within IT and be relevant to the role.
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
You are an HR/technical interviewer in the IT domain. Ask ONE next question for the candidate.
The question must be in Russian.

Context:
- Role: {position}
- Level: {grade}
- Last question: {last_question}
- Candidate answer: {last_answer}

Requirements:
- If the candidate does not mention technologies, do not require them. Ask about the role, processes,
  responsibilities, typical tasks, cases, metrics, or architecture depending on the role.
- If the candidate drifts into generic noise or answers too long, ask a shorter, more specific follow-up.
- If the candidate confidently says nonsense or mixes terms, ask to explain in practice or give an example.
- The question must be concise, only one, and always stay within the IT domain.

Return only JSON:
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
