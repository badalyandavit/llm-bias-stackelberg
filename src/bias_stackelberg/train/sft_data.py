from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from bias_stackelberg.core.jsonl import iter_jsonl


@dataclass(frozen=True)
class SftDatasetConfig:
    max_length: int = 512


def load_sft_records(path: str | Path) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        prompt = row.get("prompt")
        completion = row.get("completion")
        ex_id = row.get("id")

        if not isinstance(ex_id, str) or not ex_id:
            raise ValueError("SFT row missing valid id")
        if not isinstance(prompt, str):
            raise ValueError(f"SFT row {ex_id} missing prompt")
        if not isinstance(completion, str):
            raise ValueError(f"SFT row {ex_id} missing completion")

        recs.append(
            {
                "id": ex_id,
                "prompt": prompt,
                "completion": completion,
                "meta": row.get("meta", {}),
            }
        )
    if not recs:
        raise ValueError("No SFT records found")
    return recs


def format_prompt(prompt: str) -> str:
    if not prompt:
        return prompt
    if prompt[-1].isspace():
        return prompt
    return prompt + "\n"


class SftCausalLMDataset(Dataset):
    def __init__(
        self, records: list[dict[str, Any]], tokenizer: Any, cfg: SftDatasetConfig
    ) -> None:
        self.records = records
        self.tok = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, text: str) -> list[int]:
        out = self.tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        ids = out["input_ids"]
        if not isinstance(ids, list):
            raise TypeError("tokenizer must return list input_ids")
        return ids

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        p = format_prompt(r["prompt"])
        c = r["completion"]

        prompt_ids = self._encode(p)
        full_ids = self._encode(p + c)

        labels = full_ids.copy()
        cut = min(len(prompt_ids), len(labels))
        for i in range(cut):
            labels[i] = -100

        attn = [1] * len(full_ids)

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def pad_batch(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(x["input_ids"].shape[0]) for x in batch)

    def _pad(x: torch.Tensor, value: int) -> torch.Tensor:
        n = int(x.shape[0])
        if n == max_len:
            return x
        pad = torch.full((max_len - n,), value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([_pad(b["input_ids"], pad_id) for b in batch])
    attention_mask = torch.stack([_pad(b["attention_mask"], 0) for b in batch])
    labels = torch.stack([_pad(b["labels"], -100) for b in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
