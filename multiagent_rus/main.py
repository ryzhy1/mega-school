import asyncio
import json
import logging
import re
from typing import Any, Dict

from devdocs_loader import (
    background_load_other_techs,
    load_devdocs_for_tech,
    load_devdocs_for_tech_with_topics,
)
from feedback import generate_final_feedback
from helpers import ainput, safe_print
from interview_plan import (
    generate_interview_plan,
    generate_role_question,
    normalize_topic_seed,
    pop_next_question_for_tech,
)
from interviewer import build_interviewer_visible_message
from logger import InterviewLogger
from mcp_client import MCPServerClient
from models import QAItem
from observer import observer_analyze
from question_generation import ensure_expected_from_rag, make_answerable_question
from tech_extraction import extract_tech_slugs_from_user_text


async def run_interview():
    safe_print("=== MULTI-AGENT INTERVIEW COACH (Primary-first + Parallel RAG) ===\n")

    logger = InterviewLogger(team_name="Скирляк Ярослав Юрьевич", filename="logs/interview_log.json")
    mcp = MCPServerClient(server_script="server.py")

    try:
        name = (await ainput("👤 Имя кандидата (Alex): ")).strip() or "Alex"
        position = (await ainput("💼 Позиция (Backend Developer): ")).strip() or "Backend Developer"
        grade = (await ainput("📊 Уровень (Junior): ")).strip() or "Junior"

        profile: Dict[str, Any] = {"name": name, "position": position, "grade": grade}

        stack_text = (await ainput("\n🔧 Опиши свой стек (можно по-русски): ")).strip()

        techs = extract_tech_slugs_from_user_text(stack_text)
        domain_mode = False
        if not techs:
            domain_mode = True
            safe_print("ℹ️ Технологии не указаны — перейду к вопросам по роли и опыту в IT.")

        primary = techs[0] if techs else ""
        pending = techs[1:] if techs else []
        current_tech = primary

        profile["technologies"] = techs
        if techs:
            safe_print(f"\n🎯 Распознанные технологии: {techs}")
            safe_print(f"⭐ Primary tech: {primary}")
            if pending:
                safe_print(f"⏳ Pending (background): {pending}")

        topics_map = {}
        questions_queue = []
        if not domain_mode:
            safe_print("\n🧭 Генерирую план интервью (темы и начальные вопросы)...")
            plan = generate_interview_plan(mcp, grade, position, techs, per_tech=3)
            topics_map = plan.get("topics_map", {})
            questions_queue = plan.get("questions_queue", [])
            safe_print(f"🔎 План по темам: {topics_map}")
            safe_print(f"🗂️ Вопросов в очереди: {len(questions_queue)}")

        loaded_techs = set()
        if not domain_mode:
            safe_print(f"\n⏳ Загружаю документацию по первичной технологии: {primary}")
            primary_topics = topics_map.get(primary, [])
            if primary_topics:
                ok_primary = await asyncio.to_thread(
                    load_devdocs_for_tech_with_topics,
                    mcp,
                    primary,
                    primary_topics,
                    3,
                )
            else:
                ok_primary = await asyncio.to_thread(load_devdocs_for_tech, mcp, primary, 3)

            if ok_primary:
                loaded_techs.add(primary)
                safe_print(f"✅ Primary RAG ready: {primary}")
            else:
                safe_print(f"⚠️ Не удалось загрузить DevDocs по {primary}. Продолжу без гарантий RAG.")

        bg_task = None
        if pending and not domain_mode:
            bg_task = asyncio.create_task(
                background_load_other_techs(mcp, pending, loaded_techs, topics_map)
            )

        evaluation: Dict[str, Any] = {
            "topics": [],
            "confirmed": [],
            "gaps": [],
            "turns": [],
            "soft_notes": [],
            "signals": {"hallucination": 0, "off_topic": 0, "candidate_question": 0},
        }

        difficulty = 1
        last_agent_message = ""

        if domain_mode:
            opening_question = await asyncio.to_thread(generate_role_question, grade, position, "", "")
            qa = QAItem(question=opening_question, expected_answer="", key_points=[], topic=position)
            rag_ctx = ""
            last_agent_message = (
                f"Привет, {name}! Давай начнем интервью по роли {position}.\n\n{qa.question}"
            )
        else:
            first = pop_next_question_for_tech(questions_queue, primary)
            if first:
                q_text = first.get("question", f"Расскажи базово про {primary}.")
                seed = normalize_topic_seed(first.get("topic", ""), primary)

                qa, rag_ctx = await asyncio.to_thread(ensure_expected_from_rag, primary, q_text, seed)

                if not qa.expected_answer:
                    qa, rag_ctx = await asyncio.to_thread(
                        make_answerable_question,
                        primary,
                        difficulty,
                        3,
                        focus_topic=first.get("topic", ""),
                    )
            else:
                qa, rag_ctx = await asyncio.to_thread(make_answerable_question, primary, difficulty, 3)
            last_agent_message = f"Привет, {name}! Давай начнем техническое интервью по {primary}.\n\n{qa.question}"
        safe_print(f"\n🤖 Interviewer:\n{last_agent_message}")

        safe_print("\n(Введи 'стоп' чтобы закончить и получить фидбэк)\n")

        while True:
            user_input = (await ainput("👤 Ты: ")).strip()

            if user_input.lower() in ["стоп", "stop", "exit", "выход"]:
                internal_thoughts = "[Observer]: stop requested. [Interviewer]: generate final feedback."
                logger.log_turn(user_msg=user_input, agent_msg=last_agent_message, thoughts=internal_thoughts)
                break

            if not user_input:
                safe_print("🤖 Interviewer: Пожалуйста, ответь на вопрос (или напиши 'стоп').")
                continue

            try:
                safe_print(f"👤 {user_input}")
            except Exception:
                logging.info(f"Candidate answer: {user_input}")

            obs = await asyncio.to_thread(
                observer_analyze,
                current_tech or position,
                qa.question,
                user_input,
                qa,
                rag_ctx,
            )

            if obs.flags.get("hallucination_detected"):
                evaluation["signals"]["hallucination"] += 1
            if obs.flags.get("off_topic"):
                evaluation["signals"]["off_topic"] += 1
            if obs.flags.get("candidate_question"):
                evaluation["signals"]["candidate_question"] += 1

            topic = (obs.assessment or {}).get("topic") or qa.topic or primary or position
            if topic and topic not in evaluation["topics"]:
                evaluation["topics"].append(topic)

            correctness = (obs.assessment or {}).get("correctness", "medium")
            missing_points = (obs.assessment or {}).get("missing_points", []) or []
            correct_short = (obs.assessment or {}).get("correct_answer_short", "") or ""

            evaluation["turns"].append(
                {
                    "tech": primary,
                    "topic": topic,
                    "question": qa.question,
                    "user_answer": user_input,
                    "correctness": correctness,
                    "missing_points": missing_points,
                    "correct_answer_short": correct_short,
                }
            )

            if correctness == "high":
                tag = f"{primary}:{topic}"
                if tag not in evaluation["confirmed"]:
                    evaluation["confirmed"].append(tag)
            elif correctness == "low":
                evaluation["gaps"].append(
                    {
                        "topic": f"{primary}:{topic}",
                        "issue": "Ответ содержит ошибки/пробелы.",
                        "correct_answer": correct_short
                        or qa.expected_answer
                        or "См. документацию DevDocs по теме.",
                    }
                )

            if obs.difficulty_adjustment == "increase":
                difficulty = min(5, difficulty + 1)
            elif obs.difficulty_adjustment == "decrease":
                difficulty = max(1, difficulty - 1)

            thoughts = (
                f"[Observer]: {obs.internal_thoughts} [Interviewer]: will_follow='{obs.instruction_to_interviewer}'"
            )
            logger.log_turn(user_msg=user_input, agent_msg=last_agent_message, thoughts=thoughts)

            if obs.topic_status == "wrap_up":
                break

            prev_question = qa.question

            if domain_mode:
                next_question = await asyncio.to_thread(
                    generate_role_question,
                    grade,
                    position,
                    qa.question,
                    user_input,
                )
                qa = QAItem(question=next_question, expected_answer="", key_points=[], topic=position)
                rag_ctx = ""
            else:
                if questions_queue:
                    nxt = pop_next_question_for_tech(questions_queue, current_tech) or questions_queue.pop(0)
                    q_text = nxt.get("question", f"Расскажи про {current_tech} базово.")
                    nxt_topic = nxt.get("topic", current_tech)
                    seed = normalize_topic_seed(nxt_topic, current_tech)

                    qa, rag_ctx = await asyncio.to_thread(ensure_expected_from_rag, current_tech, q_text, seed)

                    if not qa.expected_answer:
                        qa, rag_ctx = await asyncio.to_thread(
                            make_answerable_question,
                            current_tech,
                            difficulty,
                            3,
                            focus_topic=nxt_topic,
                        )
                else:
                    qa, rag_ctx = await asyncio.to_thread(make_answerable_question, current_tech, difficulty, 3)
            history_summary = (
                f"Текущая технология: {primary or 'роль'}. "
                f"Последняя оценка: {(obs.assessment or {}).get('correctness', '')}."
            )

            reaction = await asyncio.to_thread(
                build_interviewer_visible_message,
                profile,
                current_tech,
                qa.question,
                obs,
                user_input,
                prev_question,
                history_summary,
            )

            reaction = (reaction or "").strip()

            next_tech = ""
            match = re.search(r"__NEXT_TECH__=([a-z0-9_+-]*)", reaction)
            if match:
                next_tech = (match.group(1) or "").strip().lower()

            reaction = re.sub(r"\n?__NEXT_TECH__=.*(\n)?", "\n", reaction).strip()

            if next_tech:
                allowed = (profile.get("technologies") or [])
                if next_tech in allowed:
                    current_tech = next_tech

            last_agent_message = (reaction + "\n\n" if reaction else "")
            safe_print(f"\n🤖 Interviewer:\n{last_agent_message}\n")

        if bg_task:
            try:
                await asyncio.wait_for(bg_task, timeout=3.0)
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

        feedback = await asyncio.to_thread(generate_final_feedback, profile, evaluation)
        logger.log_feedback(feedback)

        safe_print("\n📊 FINAL FEEDBACK (saved to interview_log_1.json):\n")
        safe_print(json.dumps(feedback, ensure_ascii=False, indent=2))

        safe_print("\n🏁 Done. Log file: interview_log_1.json")

    finally:
        try:
            mcp.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_interview())
