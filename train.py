"""Fine-tune a LLaMA-architecture chat model with QLoRA on support replies.

QLoRA here means: load the base weights in 4-bit NF4 (bitsandbytes, double
quantization) and train LoRA adapters with PEFT. The run refuses to continue
on a machine without CUDA, so it cannot silently fall back to full-precision LoRA.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from qlora_customer_support.data import format_example, load_jsonl, mask_prompt_labels, train_eval_split
from qlora_customer_support.qlora_config import DEFAULT_MODEL, bitsandbytes_config, lora_config

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "support.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune on Northline support dialogues.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "northline-qlora")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the quantization and LoRA settings, then exit without training.",
    )
    return parser.parse_args()


def require_cuda() -> None:
    if torch.cuda.is_available():
        return
    raise SystemExit(
        "QLoRA 4-bit training needs an NVIDIA GPU with CUDA. "
        "This machine has no CUDA device, so training stopped instead of "
        "falling back to full-precision LoRA. Use a CUDA machine or Colab, "
        "or run `python train.py --check` to validate the config on CPU."
    )


def tokenize_row(tokenizer, row: dict[str, str], max_length: int) -> dict[str, list[int]]:
    prompt, full = format_example(row["instruction"], row["response"])
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full + eos,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    labels = mask_prompt_labels(full_ids, len(prompt_ids))
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def collate(features: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(feature["input_ids"]) for feature in features)
    input_ids, labels, attention = [], [], []
    for feature in features:
        pad = width - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_id] * pad)
        labels.append(feature["labels"] + [-100] * pad)
        attention.append(feature["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


class EncodedDialogues(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, list[int]]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


def build_trainer(args: argparse.Namespace) -> tuple[Trainer, AutoTokenizer]:
    rows = load_jsonl(args.data)
    train_rows, eval_rows = train_eval_split(rows, eval_size=args.eval_size, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bitsandbytes_config(),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config())
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_tokens = EncodedDialogues([tokenize_row(tokenizer, row, args.max_length) for row in train_rows])
    eval_tokens = EncodedDialogues([tokenize_row(tokenizer, row, args.max_length) for row in eval_rows])
    argument_kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        logging_steps=5,
        eval_steps=max(args.max_steps // 2, 1),
        save_strategy="no",
        fp16=True,
        report_to=[],
        seed=args.seed,
    )
    # Transformers renamed evaluation_strategy to eval_strategy.
    parameters = TrainingArguments.__init__.__code__.co_varnames
    if "eval_strategy" in parameters:
        argument_kwargs["eval_strategy"] = "steps"
    else:
        argument_kwargs["evaluation_strategy"] = "steps"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**argument_kwargs),
        train_dataset=train_tokens,
        eval_dataset=eval_tokens,
        data_collator=partial(collate, pad_id=tokenizer.pad_token_id),
    )
    return trainer, tokenizer


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.data)
    quant = bitsandbytes_config()
    adapters = lora_config()
    print(f"dialogues: {len(rows)}")
    print(
        "quantization: "
        f"load_in_4bit={quant.load_in_4bit} "
        f"type={quant.bnb_4bit_quant_type} "
        f"double_quant={quant.bnb_4bit_use_double_quant} "
        f"compute={quant.bnb_4bit_compute_dtype}"
    )
    print(
        "lora: "
        f"r={adapters.r} alpha={adapters.lora_alpha} "
        f"dropout={adapters.lora_dropout} targets={list(adapters.target_modules)}"
    )
    if args.check:
        print("check ok")
        return

    require_cuda()
    trainer, tokenizer = build_trainer(args)
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = {"log_history": trainer.state.log_history}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved adapter to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
