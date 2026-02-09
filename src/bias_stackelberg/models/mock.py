from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from bias_stackelberg.core.rng import stable_int_seed
from bias_stackelberg.core.types import Generation
from bias_stackelberg.models.base import GenConfig


class MockLLM:
    """
    Deterministic mock generator.
    Output is a function of (prompt, seed, config fields).
    """

    def generate(
        self, prompt: str, *, config: GenConfig | None = None, **kwargs: Any
    ) -> Generation:
        cfg = config or GenConfig()
        rng = random.Random(stable_int_seed(cfg.seed, prompt))

        words = prompt.strip().split()
        base = " ".join(words[: min(8, len(words))]) if words else "EMPTY_PROMPT"

        # Make output depend on config in a simple, stable way.
        t_bucket = int(cfg.temperature * 1000)
        p_bucket = int(cfg.top_p * 1000)
        tok = max(1, int(cfg.max_tokens))

        noise = "".join(rng.choice("abcdef0123456789") for _ in range(16))
        text = f"[mock t={t_bucket} p={p_bucket} mt={tok}] {base} :: {noise}"

        meta = {
            "backend": "mock",
            "config": asdict(cfg),
            "extra_kwargs": dict(kwargs),
        }
        return Generation(text=text, meta=meta)
