import json
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import Dataset

from .config import NO_ANSWER_TOKEN


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_fields(rows: Iterable[Dict], required_fields: Iterable[str]) -> None:
    required = set(required_fields)
    for idx, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {idx} missing fields: {sorted(missing)}")


def normalize_gold_output(value: str) -> str:
    value = (value or "").strip()
    return NO_ANSWER_TOKEN if not value else value


def to_hf_dataset(rows: List[Dict]) -> Dataset:
    return Dataset.from_list(rows)

