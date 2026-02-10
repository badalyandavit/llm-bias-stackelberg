from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bias_stackelberg.core.jsonl import JsonlWriter
from bias_stackelberg.core.types import Example, Generation
from bias_stackelberg.eval.metrics import RunMetrics, compute_metrics
from bias_stackelberg.follower.option_a import OptionAConfig, OptionAFollower
from bias_stackelberg.leader.rule_based import RuleBasedLeader, default_policy
from bias_stackelberg.models import GenConfig, MockLLM


@dataclass(frozen=True)
class EvalAConfig:
    out_dir: str
    gen: GenConfig
    follower: OptionAConfig
    reference_rewrite: bool = False


def _forced_y0(ex: Example) -> Generation | None:
    y0_text = ex.meta.get("y0_text")
    if isinstance(y0_text, str) and y0_text.strip():
        return Generation(text=y0_text, meta={"backend": "provided"})

    if ex.meta.get("force_biased"):
        return Generation(text="Women are bad at math.", meta={"backend": "forced"})

    return None


def _leader_score(leader_obj: Any, prompt: str, text: str) -> Any:
    try:
        return leader_obj.score(prompt, text)
    except TypeError:
        return leader_obj.score(text)


def run_option_a(
    examples: list[Example],
    *,
    cfg: EvalAConfig,
    leader: Any | None = None,
    llm: Any | None = None,
) -> RunMetrics:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    leader_obj = leader if leader is not None else RuleBasedLeader(policy=default_policy())
    llm_obj = llm if llm is not None else MockLLM()
    follower = OptionAFollower(leader=leader_obj, llm=llm_obj, config=cfg.follower)

    rows: list[dict[str, Any]] = []

    with JsonlWriter(out_dir / "predictions.jsonl") as w:
        for ex in examples:
            forced = _forced_y0(ex)
            gen0 = forced if forced is not None else llm_obj.generate(ex.prompt, config=cfg.gen)

            before = _leader_score(leader_obj, ex.prompt, gen0.text)

            ref = ex.meta.get("reference_text")
            if cfg.reference_rewrite and isinstance(ref, str) and ref.strip():
                gen1 = Generation(
                    text=ref, meta={"follower": "reference", "action": "rewrite_reference"}
                )
                after = _leader_score(leader_obj, ex.prompt, gen1.text)
                decision = {
                    "action": "rewrite",
                    "reason": "reference_rewrite",
                    "before_score": before.score,
                    "after_score": after.score,
                }
            else:
                gen1, dec = follower.intervene(ex.prompt, gen0.text, before)
                after = _leader_score(leader_obj, ex.prompt, gen1.text)
                decision = {
                    "action": dec.action,
                    "reason": dec.reason,
                    "before_score": dec.before_score,
                    "after_score": dec.after_score,
                }

            row = {
                "id": ex.id,
                "prompt": ex.prompt,
                "meta": ex.meta,
                "y0": {"text": gen0.text, "meta": gen0.meta},
                "y1": {"text": gen1.text, "meta": gen1.meta},
                "before": before.to_dict(),
                "after": after.to_dict(),
                "decision": decision,
            }
            w.write(row)
            rows.append(row)

    metrics = compute_metrics(rows)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2),
        encoding="utf-8",
    )
    return metrics
