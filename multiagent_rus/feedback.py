import json
from typing import Any, Dict

from config import llm
from helpers import extract_json_object


def generate_final_feedback(profile: Dict[str, Any], evaluation: Dict[str, Any]) -> dict:
    """
    Use LLM to produce structured feedback, but grounded in collected evaluation.
    """
    prompt = f"""
Ты Hiring Manager + Mentor. Сгенерируй структурированный отчет по итогам интервью.

Профиль:
{profile}

Накопленные результаты интервью (ground truth):
{json.dumps(evaluation, ensure_ascii=False, indent=2)}

Требуемый формат (ТОЛЬКО JSON):
{{
  "decision": {{
    "grade": "Junior | Middle | Senior",
    "hiring_recommendation": "Hire | No Hire | Strong Hire",
    "confidence_score": 0-100
  }},
  "technical_review": {{
    "topics": ["..."],
    "confirmed_skills": ["..."],
    "knowledge_gaps": [
      {{
        "topic": "...",
        "issue": "...",
        "correct_answer": "..."
      }}
    ]
  }},
  "soft_skills": {{
    "clarity": "low|medium|high",
    "honesty": "low|medium|high",
    "engagement": "low|medium|high",
    "notes": ["..."]
  }},
  "roadmap": [
    {{
      "topic": "...",
      "next_steps": ["...","..."],
      "links": ["optional"]
    }}
  ]
}}

Правила:
- В knowledge_gaps обязательно дай правильный ответ.
- Roadmap должен следовать из knowledge_gaps.
- Никакого текста кроме JSON.
"""
    resp = llm.invoke([("human", prompt)])
    obj = extract_json_object(resp.content)
    try:
        return json.loads(obj)
    except Exception:
        gaps = evaluation.get("gaps", [])
        confirmed = evaluation.get("confirmed", [])
        topics = sorted(set(evaluation.get("topics", [])))
        return {
            "decision": {
                "grade": profile.get("grade", "Junior"),
                "hiring_recommendation": "No Hire",
                "confidence_score": 55,
            },
            "technical_review": {
                "topics": topics,
                "confirmed_skills": confirmed,
                "knowledge_gaps": [
                    {
                        "topic": g.get("topic", "unknown"),
                        "issue": g.get("issue", ""),
                        "correct_answer": g.get("correct_answer", ""),
                    }
                    for g in gaps
                ],
            },
            "soft_skills": {
                "clarity": "medium",
                "honesty": "medium",
                "engagement": "medium",
                "notes": ["Fallback feedback."],
            },
            "roadmap": [],
        }
