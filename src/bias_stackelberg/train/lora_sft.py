from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bias_stackelberg.train.sft_data import (
    SftCausalLMDataset,
    SftDatasetConfig,
    load_sft_records,
    pad_batch,
)


@dataclass(frozen=True)
class TrainLoRAConfig:
    sft_jsonl: str
    out_dir: str
    model_name: str = "distilgpt2"
    max_length: int = 512

    seed: int = 0
    max_steps: int = 50
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 1

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("c_attn",)


def train_lora_sft(*, cfg: TrainLoRAConfig) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as e:
        raise RuntimeError('Training deps missing. Install with: pip install -e ".[train]"') from e

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)

    records = load_sft_records(cfg.sft_jsonl)

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.config.pad_token_id = tok.pad_token_id

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(model, peft_cfg)

    ds_cfg = SftDatasetConfig(max_length=cfg.max_length)
    train_ds = SftCausalLMDataset(records, tok, ds_cfg)

    def collate(batch):
        return pad_batch(batch, pad_id=int(tok.pad_token_id))

    args = TrainingArguments(
        output_dir=str(out_dir / "hf_trainer"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        logging_steps=max(1, cfg.max_steps // 10),
        save_steps=cfg.max_steps,
        report_to=[],
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
    trainer.train()

    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    summary = {
        "train_records": len(records),
        "adapter_dir": str(adapter_dir),
        "config": asdict(cfg),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
