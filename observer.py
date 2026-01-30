import json

from config import llm
from helpers import extract_json_object, safe_print
from models import ObserverResult, QAItem
from question_generation import debug_block


def observer_analyze(
    tech: str,
    last_agent_question: str,
    user_answer: str,
    qa: QAItem,
    rag_ctx: str,
) -> ObserverResult:
    prompt = f"""
You are an Observer/Critic. Evaluate the candidate's answer based on the documentation (CONTEXT) and the expected ground truth (EXPECTED).

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

Requirements:
- Mark if the candidate goes off-topic (off_topic).
- Mark if the candidate confidently states nonsense/false facts (hallucination_detected).
- Mark if the candidate asks a question instead of answering (candidate_question).
- If the candidate does NOT answer the asked question (even if it's about Go in general) — set off_topic=true.
- If EXPECTED and CONTEXT are empty (no factual base), evaluate only relevance and coherence,
  and do NOT set hallucination_detected without obvious contradictions to common sense.

Return ONLY JSON:
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

Scoring rubric (important):
- If the answer SEMANTICALLY matches EXPECTED but is shortened/without full package prefix, treat as correctness="high"
  (for Junior, "high" is acceptable); missing_points may be empty or ["prefer full path like go/parser"].
- Set off_topic=true ONLY if the answer is clearly not about the question (e.g. "switch tech", "I don't know", "let's talk life").
  A short/incomplete but on-topic answer is NOT off_topic.
- Set candidate_question=true if the candidate asks a question/requests a tech switch instead of answering.
- Set hallucination_detected=true only if the candidate confidently states a fact contradicting CONTEXT/EXPECTED.

Correctness rules:
- high: essentially correct (including acceptable abbreviations), key fact matches
- medium: partially correct/inexact but direction is right
- low: incorrect or no substantive answer

"""
    resp = llm.invoke([("human", prompt)])
    raw = resp.content
    obj = extract_json_object(raw)

    debug_block("Observer input: EXPECTED ANSWER", qa.expected_answer)
    debug_block("Observer input: KEY POINTS", json.dumps(qa.key_points, ensure_ascii=False))
    debug_block("Observer input: RAG CONTEXT", rag_ctx)

    try:
        data = json.loads(obj)
    except Exception:
        debug_block("❌ Observer JSON parse failed. Raw model output", raw)
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
        debug_block("Observer result JSON", json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass

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
