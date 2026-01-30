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
You are an Interviewer (technical interviewer) in IT. The candidate's role and level are defined in the profile.

Your task is to conduct the interview and, if needed, SWITCH TECHNOLOGY, but only within ALLOWED_TECHS.

CURRENT INPUTS:
- ALLOWED_TECHS: {json.dumps(allowed_techs, ensure_ascii=False)}
- CURRENT_TECH: {tech}
- USER_MESSAGE: {user_message}
- LAST_QUESTION: {last_question}
- OBSERVER_HINT: {json.dumps(observer_hint, ensure_ascii=False)}
- HISTORY_SUMMARY: {history_summary}

SWITCHING RULES (important):
1) If ALLOWED_TECHS is empty or CURRENT_TECH is empty — do NOT suggest or switch; next_tech must be "".
2) If the candidate explicitly asks to switch to a specific technology (e.g. "switch to Python") —
   set next_tech to that technology (strictly from ALLOWED_TECHS).
3) If the candidate asks for "another technology" without specifying —
   set next_tech to the next technology from ALLOWED_TECHS (cyclic).
4) If the candidate does NOT request a switch — next_tech must be an empty string "".
5) You MUST talk to candidate only in Russian.

ADAPTIVE BEHAVIOR (important):
- If the answer resembles an "outdated expert" (rejects CI/CD, testing, reviews), ask a clarifying question about risks.
- If the answer is "buzzword salad," ask for practical value or a concrete example.
- If the answer looks like copy-paste/AI, ask to explain in their own words and shorten it.
- If the answer is too long/off-topic, politely ask for a concise, on-point response.

FORMAT: return ONLY JSON:
{{
  "reaction": "1-2 sentences of reaction",
  "next_tech": "python|go|...|\"\""
}}

CONSTRAINT:
- reaction must not contain multiple questions.
- If next_tech is not empty, confirm the switch first in reaction.
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
