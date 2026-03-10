from typing import Dict, List

from .config import NO_ANSWER_TOKEN


def build_messages(question: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def normalize_prediction(text: str) -> str:
    # Keep output deterministic: first non-empty line, trimmed.
    candidate = (text or "").strip().splitlines()
    value = candidate[0].strip() if candidate else ""
    if not value:
        return NO_ANSWER_TOKEN
    if NO_ANSWER_TOKEN in value:
        return NO_ANSWER_TOKEN
    return value

