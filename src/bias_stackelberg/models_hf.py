from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bias_stackelberg.core.types import Generation
from bias_stackelberg.models import GenConfig


def _stable_seed_from_prompt(seed: int, prompt: str) -> int:
    s = f"{seed}|{prompt}".encode()
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:4], byteorder="big", signed=False)


def _pick_device(device: str) -> str:
    d = device.strip().lower()
    if d != "auto":
        return d
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class HfCausalLM:
    model_name: str
    adapter_dir: str | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        self._device = _pick_device(self.device)
        torch_device = torch.device(self._device)

        dtype: torch.dtype
        if self._device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.float16

        tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )
        model.to(torch_device)
        model.eval()

        if self.adapter_dir:
            try:
                from peft import PeftModel
            except Exception as e:
                raise RuntimeError("Install extras: pip install -e '.[train]'") from e
            model = PeftModel.from_pretrained(model, self.adapter_dir)
            model.to(torch_device)
            model.eval()

        self._tok = tok
        self._model = model

    def generate(self, prompt: str, config: GenConfig, **extra_kwargs: Any) -> Generation:
        torch_device = torch.device(self._device)

        seed = _stable_seed_from_prompt(config.seed, prompt)
        torch.manual_seed(seed)
        if self._device == "cuda":
            torch.cuda.manual_seed_all(seed)

        inputs = self._tok(prompt, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}

        do_sample = bool(config.temperature and config.temperature > 0.0)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(config.max_tokens),
            "do_sample": do_sample,
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
            "pad_token_id": self._tok.pad_token_id,
            "eos_token_id": self._tok.eos_token_id,
        }
        gen_kwargs.update(extra_kwargs)

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = int(inputs["input_ids"].shape[1])
        gen_ids = out[0][prompt_len:]
        text = self._tok.decode(gen_ids, skip_special_tokens=True).strip()

        if config.stop:
            for s in config.stop:
                if not s:
                    continue
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    break

        config_dict = {
            "seed": int(config.seed),
            "max_tokens": int(config.max_tokens),
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
            "stop": list(config.stop) if config.stop else None,
        }

        meta = {
            "backend": "hf",
            "model_name": self.model_name,
            "device": self._device,
            "adapter_dir": self.adapter_dir,
            "config": config_dict,
            "extra_kwargs": extra_kwargs,
        }
        return Generation(text=text, meta=meta)
