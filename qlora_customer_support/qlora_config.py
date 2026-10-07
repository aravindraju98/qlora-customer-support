"""4-bit QLoRA settings. Importing this does not download or train a model."""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def bitsandbytes_config() -> BitsAndBytesConfig:
    """NF4 4-bit load with double quantization. This is the QLoRA quantization step."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def lora_config() -> LoraConfig:
    """LoRA adapters on the attention projections of a LLaMA-architecture model."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=list(TARGET_MODULES),
        bias="none",
        inference_mode=False,
    )
