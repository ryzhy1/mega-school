import re
from typing import List

from config import DEV_DOCS_SLUGS, TECH_SYNONYMS, llm


def extract_tech_slugs_from_user_text(user_text: str) -> List[str]:
    user_low = user_text.lower()
    tokens = re.findall(r"[a-zA-Zа-яА-Я0-9_+#.-]+", user_low)

    pre: List[str] = []
    for token in tokens:
        if token in TECH_SYNONYMS:
            pre.append(TECH_SYNONYMS[token])
        if token in DEV_DOCS_SLUGS:
            pre.append(token)

    pre = list(dict.fromkeys(pre))
    return pre[:6]
