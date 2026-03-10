from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NO_ANSWER_TOKEN = "<NO-ANSWER>"

# Short and explicit: answer when grounded, abstain when not.
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful question answering assistant. "
    "If the question cannot be answered reliably, output exactly <NO-ANSWER>. "
    "If it can be answered, output only the short answer with no extra words."
)

DATASET_PATHS = {
    "dataset1": "data/dataset1.jsonl",
    "dataset2": "data/dataset2.jsonl",
    "dataset3": "data/dataset3.jsonl",
    "dataset3b": "data/dataset3b.jsonl",
    "dataset3c": "data/dataset3c.jsonl",
    "dataset3d": "data/dataset3d.jsonl",
    "dataset3e": "data/dataset3e.jsonl",
    "dataset4": "data/dataset4.jsonl",
}


@dataclass
class QuantizationConfig:
    mode: str = "none"  # none | 8bit | 4bit

    def validate(self) -> None:
        allowed = {"none", "8bit", "4bit"}
        if self.mode not in allowed:
            raise ValueError(f"Unsupported quantization mode '{self.mode}'. Use one of {allowed}.")


def resolve_dataset_path(repo_root: Path, dataset_name: str) -> Path:
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid options: {sorted(DATASET_PATHS.keys())}")
    return repo_root / DATASET_PATHS[dataset_name]

