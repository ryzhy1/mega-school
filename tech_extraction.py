import json
import re
from typing import List

from config import DEV_DOCS_SLUGS, TECH_SYNONYMS, llm
from helpers import extract_json_array


def extract_tech_slugs_from_user_text(user_text: str) -> List[str]:
    slugs_text = ", ".join(DEV_DOCS_SLUGS)
    prompt = f"""
Return ONLY a JSON array of strings (no other text), where each string is a DevDocs slug from the allowed list below.
If nothing is found, return an empty array [] (and no other text).
Example format: ["python", "go"]

Allowed slugs:
[{slugs_text}]

User text:
{user_text}
"""

    resp = llm.invoke([("human", prompt)])
    arr = extract_json_array(resp.content)

    llm_slugs: List[str] = []
    try:
        parsed = json.loads(arr)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, str):
                    continue
                slug = item.strip().lower()
                if slug in TECH_SYNONYMS:
                    slug = TECH_SYNONYMS[slug]
                if slug in DEV_DOCS_SLUGS:
                    llm_slugs.append(slug)
            llm_slugs = list(dict.fromkeys(llm_slugs))
    except Exception:
        llm_slugs = []

    user_low = user_text.lower()
    tokens = re.findall(r"[a-zA-Zа-яА-Я0-9_+#.-]+", user_low)

    pre: List[str] = []
    for token in tokens:
        if token in TECH_SYNONYMS:
            pre.append(TECH_SYNONYMS[token])
        if token in DEV_DOCS_SLUGS:
            pre.append(token)

    pre = list(dict.fromkeys(pre))

    if llm_slugs:
        merged = llm_slugs + [p for p in pre if p not in llm_slugs]
    else:
        merged = pre

    return merged[:6]
