import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments

from .data import ensure_fields, load_jsonl, normalize_gold_output
from .modeling import load_base_model, load_tokenizer
from .prompting import build_messages


def _build_train_examples(rows: List[Dict], tokenizer, system_prompt: str, max_length: int) -> Dataset:
    features = {"input_ids": [], "attention_mask": [], "labels": []}

    for row in rows:
        prompt_messages = build_messages(row["input"], system_prompt)
        full_messages = prompt_messages + [{"role": "assistant", "content": normalize_gold_output(row["output"])}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        features["input_ids"].append(input_ids)
        features["attention_mask"].append(attention_mask)
        features["labels"].append(labels)

    return Dataset.from_dict(features)


def train_lora(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    system_prompt: str,
    quantization_mode: str = "4bit",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 768,
    max_train_examples: Optional[int] = None,
    logging_steps: int = 20,
    save_steps: int = 200,
    resume_from_checkpoint: Optional[str] = None,
    # New parameters for improved training
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    lr_scheduler_type: str = "linear",
    eval_dataset_path: Optional[Path] = None,
    eval_steps: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
) -> Dict:
    rows = load_jsonl(dataset_path)
    ensure_fields(rows, ["id", "input", "output"])
    if max_train_examples is not None:
        rows = rows[:max_train_examples]

    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, quantization_mode=quantization_mode, for_training=True)

    if quantization_mode in {"4bit", "8bit"}:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = _build_train_examples(rows, tokenizer=tokenizer, system_prompt=system_prompt, max_length=max_length)

    # Build eval dataset if provided
    eval_dataset = None
    if eval_dataset_path is not None:
        eval_rows = load_jsonl(eval_dataset_path)
        ensure_fields(eval_rows, ["id", "input", "output"])
        eval_dataset = _build_train_examples(eval_rows, tokenizer=tokenizer, system_prompt=system_prompt, max_length=max_length)

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if quantization_mode in {"4bit", "8bit"} else "adamw_torch",
    )

    if eval_dataset is not None:
        training_args_kwargs["eval_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = eval_steps or save_steps
        training_args_kwargs["load_best_model_at_end"] = early_stopping_patience is not None
        training_args_kwargs["metric_for_best_model"] = "eval_loss"
        training_args_kwargs["greater_is_better"] = False

    args = TrainingArguments(**training_args_kwargs)

    callbacks = []
    if early_stopping_patience is not None and eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=callbacks if callbacks else None,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_adapter_dir = output_dir / "final_adapter"
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # Copy trainer_state.json to output dir root for easy access
    trainer_state_src = Path(trainer.state.best_model_checkpoint or str(output_dir)) / "trainer_state.json"
    if not trainer_state_src.exists():
        # Search checkpoints for trainer_state.json
        checkpoints = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
        if checkpoints:
            trainer_state_src = checkpoints[-1]
    if trainer_state_src.exists():
        shutil.copy2(trainer_state_src, output_dir / "trainer_state.json")

    return {"adapter_dir": final_adapter_dir, "output_dir": output_dir}
