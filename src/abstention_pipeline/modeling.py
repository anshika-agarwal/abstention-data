from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _build_quant_config(mode: str) -> Optional[BitsAndBytesConfig]:
    if mode == "none":
        return None
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    raise ValueError(f"Unsupported quantization mode: {mode}")


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str,
    quantization_mode: str = "none",
    for_training: bool = False,
):
    quant_config = _build_quant_config(quantization_mode)
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if for_training:
        model.config.use_cache = False
    return model


def load_model_with_adapter(
    model_name: str,
    adapter_path: str,
    quantization_mode: str = "none",
) -> Tuple[AutoModelForCausalLM, object]:
    tokenizer = load_tokenizer(model_name)
    base_model = load_base_model(model_name, quantization_mode=quantization_mode, for_training=False)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

