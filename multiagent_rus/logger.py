import json

from helpers import now_iso


class InterviewLogger:
    def __init__(self, team_name: str = "Скирляк Ярослав Юрьевич", filename: str = "interview_log_1.json"):
        self.filename = filename
        self.log_data = {
            "team_name": team_name,
            "created_at": now_iso(),
            "turns": [],
            "final_feedback": None,
        }
        self.turn_counter = 1
        self._save()

    def log_turn(self, user_msg: str, agent_msg: str, thoughts: str):
        self.log_data["turns"].append(
            {
                "turn_id": self.turn_counter,
                "agent_visible_message": agent_msg,
                "user_message": user_msg,
                "internal_thoughts": thoughts,
            }
        )
        self.turn_counter += 1
        self._save()

    def log_feedback(self, feedback: dict):
        self.log_data["final_feedback"] = feedback
        self._save()

    def _save(self):
        with open(self.filename, "w", encoding="utf-8") as file:
            json.dump(self.log_data, file, ensure_ascii=False, indent=2)
