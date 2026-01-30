from typing import Any, Dict, List

from pydantic import BaseModel, Field


class QAItem(BaseModel):
    question: str
    expected_answer: str = ""
    key_points: List[str] = Field(default_factory=list)
    topic: str = ""


class ObserverResult(BaseModel):
    internal_thoughts: str
    instruction_to_interviewer: str
    topic_status: str
    difficulty_adjustment: str
    flags: Dict[str, Any] = Field(default_factory=dict)
    assessment: Dict[str, Any] = Field(default_factory=dict)
