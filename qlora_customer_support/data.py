"""Load support dialogues and mask the prompt so loss is on the reply."""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            instruction = str(row.get("instruction", "")).strip()
            response = str(row.get("response", "")).strip()
            if not instruction or not response:
                raise ValueError(f"Line {line_number} needs instruction and response.")
            rows.append({"instruction": instruction, "response": response})
    if not rows:
        raise ValueError(f"No dialogues in {path}.")
    return rows


def format_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def format_example(instruction: str, response: str) -> tuple[str, str]:
    prompt = format_prompt(instruction)
    return prompt, prompt + response.strip()


def mask_prompt_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    """Label only the reply. Prompt tokens, and any padding, stay at -100."""
    if not input_ids:
        raise ValueError("input_ids is empty.")
    masked = min(max(prompt_len, 0), len(input_ids) - 1)
    labels = list(input_ids)
    for index in range(masked):
        labels[index] = -100
    return labels


def train_eval_split(
    rows: list[dict[str, str]],
    eval_size: int,
    seed: int = 7,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if eval_size <= 0 or eval_size >= len(rows):
        raise ValueError("eval_size must leave at least one training row.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[eval_size:], shuffled[:eval_size]
