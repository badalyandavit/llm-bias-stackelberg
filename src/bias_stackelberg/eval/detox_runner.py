from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bias_stackelberg.core.jsonl import JsonlWriter
from bias_stackelberg.core.prompts import detox_rewrite_prompt
from bias_stackelberg.core.types import Example, Generation
from bias_stackelberg.eval.detox_metrics import (
    DetoxMetrics,
    compute_detox_metrics,
    constraint_stats,
)
from bias_stackelberg.models import GenConfig, MockLLM


@dataclass(frozen=True)
class EvalDetoxConfig:
    out_dir: str
    gen: GenConfig
    min_new_tokens: int = 8
    use_meta_y0: bool = True


def _source_text(ex: Example, *, use_meta_y0: bool) -> str:
    if use_meta_y0:
        y0 = ex.meta.get("y0_text")
        if isinstance(y0, str) and y0.strip():
            return y0
    return ex.prompt


def _leader_score(leader_obj: Any, prompt: str, text: str) -> Any:
    try:
        return leader_obj.score(prompt, text)
    except TypeError:
        return leader_obj.score(text)


def run_detox_eval(
    examples: list[Example],
    *,
    cfg: EvalDetoxConfig,
    leader: Any,
    llm: Any | None = None,
) -> DetoxMetrics:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_obj = llm if llm is not None else MockLLM()

    rows: list[dict[str, Any]] = []

    with JsonlWriter(out_dir / "predictions.jsonl") as w:
        for ex in examples:
            raw_text = _source_text(ex, use_meta_y0=cfg.use_meta_y0)
            model_prompt = detox_rewrite_prompt(raw_text)

            before = _leader_score(leader, raw_text, raw_text)

            gen: Generation = llm_obj.generate(
                model_prompt,
                config=cfg.gen,
                min_new_tokens=int(cfg.min_new_tokens),
            )
            gtxt = gen.text

            after = _leader_score(leader, raw_text, gtxt)

            constraints = constraint_stats(raw_text, gtxt)

            row = {
                "id": ex.id,
                "prompt_text": raw_text,
                "model_prompt": model_prompt,
                "meta": ex.meta,
                "y_gen": {"text": gtxt, "meta": gen.meta},
                "before": before.to_dict(),
                "after": after.to_dict(),
                "constraints": constraints,
            }
            w.write(row)
            rows.append(row)

    metrics = compute_detox_metrics(rows)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2),
        encoding="utf-8",
    )
    return metrics
