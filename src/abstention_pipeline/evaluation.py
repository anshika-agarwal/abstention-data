import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm

from .config import NO_ANSWER_TOKEN
from .data import ensure_fields, load_jsonl, normalize_gold_output
from .metrics import compute_metrics
from .prompting import build_messages, normalize_prediction


def _batched(items: List[Dict], batch_size: int) -> List[List[Dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@torch.inference_mode()
def run_benchmark(
    model,
    tokenizer,
    dataset_path: Path,
    system_prompt: str,
    max_new_tokens: int = 32,
    batch_size: int = 8,
    max_examples: Optional[int] = None,
    max_input_tokens: Optional[int] = None,
) -> Dict:
    rows = load_jsonl(dataset_path)
    ensure_fields(rows, ["id", "input", "output"])
    if max_examples is not None:
        rows = rows[:max_examples]

    records: List[Dict] = []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    model_max = getattr(tokenizer, "model_max_length", 4096)
    # Some tokenizers report very large sentinels for "unbounded"; clamp to practical limits.
    if not isinstance(model_max, int) or model_max > 100000:
        model_max = 4096
    effective_max_input_tokens = max_input_tokens or model_max
    effective_max_input_tokens = min(effective_max_input_tokens, model_max)

    # Decoder-only generation with batched padding should use left padding.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # We use deterministic greedy decode; clear sampling-only knobs to avoid warnings.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        for key in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, key):
                setattr(generation_config, key, None)

    try:
        for chunk in tqdm(list(_batched(rows, batch_size)), desc=f"Eval {dataset_path.name}"):
            prompts = []
            for row in chunk:
                messages = build_messages(row["input"], system_prompt)
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt_text)

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=effective_max_input_tokens,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

            prompt_length = encoded["input_ids"].shape[1]
            for i, row in enumerate(chunk):
                output_ids = generated[i, prompt_length:]
                raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                pred = normalize_prediction(raw_text)
                gold = normalize_gold_output(row["output"])
                if not gold:
                    gold = NO_ANSWER_TOKEN

                records.append(
                    {
                        "id": row["id"],
                        "input": row["input"],
                        "prediction": pred,
                        "gold": gold,
                        "raw_prediction": raw_text.strip(),
                        "unanswerable_type": row.get("unanswerable_type"),
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side

    metrics = compute_metrics(records)
    return {"dataset": str(dataset_path), "metrics": metrics, "predictions": records}


def save_report(report: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def save_predictions(records: List[Dict], output_path: Path) -> None:
    """Save just the predictions list as JSON (decoupled from full report)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def load_predictions(path: Path) -> List[Dict]:
    """Load predictions from JSON. Handles both formats:
    - Old full-report format with a 'predictions' key
    - New predictions-only format (bare list)
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    raise ValueError(f"Cannot extract predictions from {path}: expected list or dict with 'predictions' key")

