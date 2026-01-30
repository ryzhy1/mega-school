import asyncio
import json
import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone

from bs4 import BeautifulSoup

CONSOLE_LOCK = threading.Lock()
INPUT_ACTIVE = threading.Event()
PRINT_BUFFER = deque()

logging.basicConfig(
    filename="../interview.log",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s %(levelname)s %(message)s",
)

CLEAN_CONSOLE_LOGS = True
SHOW_RAG_IN_CONSOLE = True


def clean_html(html_content: str) -> str:
    """Clean HTML -> plain text for embedding."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 3500:
        text = text[:3500] + "..."
    return text


def extract_json_array(text: str) -> str:
    """Extract best-effort JSON array from text."""
    text = text.strip()
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            json_part = parts[1].split("```")[0].strip()
            if json_part.startswith("[") and json_part.endswith("]"):
                return json_part

    starts = [i for i, c in enumerate(text) if c == "["]
    ends = [i for i, c in enumerate(text) if c == "]"]
    best = None
    best_len = 0
    for start in starts:
        for end in ends:
            if end > start:
                cand = text[start:end + 1]
                if len(cand) <= best_len:
                    continue
                try:
                    json.loads(cand)
                    best = cand
                    best_len = len(cand)
                except Exception:
                    pass
    return best if best else "[]"


def extract_json_object(text: str) -> str:
    """Extract best-effort JSON object from text."""
    text = text.strip()
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            json_part = parts[1].split("```")[0].strip()
            if json_part.startswith("{") and json_part.endswith("}"):
                return json_part
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= 0:
        return "{}"
    return text[start:end]


def safe_print(*args, **kwargs):
    """Thread-safe console output with buffering during input."""
    msg = " ".join(str(a) for a in args)
    try:
        logging.info(msg)
    except Exception:
        pass

    should_print = True

    with CONSOLE_LOCK:
        if INPUT_ACTIVE.is_set() and should_print:
            PRINT_BUFFER.append(msg)
        elif INPUT_ACTIVE.is_set() and not should_print:
            PRINT_BUFFER.append(msg)
        else:
            if should_print:
                print(msg, **{k: v for k, v in kwargs.items() if k != "file"})


def flush_buffered_prints():
    """Flush buffered console messages."""
    with CONSOLE_LOCK:
        while PRINT_BUFFER:
            msg = PRINT_BUFFER.popleft()
            print(msg)


async def ainput(prompt: str) -> str:
    """Async-friendly input() that marks input-active period to avoid interleaving prints."""
    loop = asyncio.get_running_loop()

    def blocking_input():
        INPUT_ACTIVE.set()
        try:
            return input(prompt)
        finally:
            INPUT_ACTIVE.clear()
            try:
                flush_buffered_prints()
            except Exception:
                pass

    return await loop.run_in_executor(None, blocking_input)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
