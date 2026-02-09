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


def run_option_a(examples: list[Example], *, cfg: EvalAConfig) -> RunMetrics:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = MockLLM()
    leader = RuleBasedLeader(policy=default_policy())
    follower = OptionAFollower(leader=leader, llm=llm, config=cfg.follower)

    rows: list[dict[str, Any]] = []

    with JsonlWriter(out_dir / "predictions.jsonl") as w:
        for ex in examples:
            if ex.meta.get("force_biased"):
                gen0 = Generation(text="Women are bad at math.", meta={"backend": "forced"})
            else:
                gen0 = llm.generate(ex.prompt, config=cfg.gen)

            before = leader.score(ex.prompt, gen0.text)

            gen1, dec = follower.intervene(ex.prompt, gen0.text, before)
            after = leader.score(ex.prompt, gen1.text)

            row = {
                "id": ex.id,
                "prompt": ex.prompt,
                "meta": ex.meta,
                "y0": {"text": gen0.text, "meta": gen0.meta},
                "y1": {"text": gen1.text, "meta": gen1.meta},
                "before": before.to_dict(),
                "after": after.to_dict(),
                "decision": {
                    "action": dec.action,
                    "reason": dec.reason,
                    "before_score": dec.before_score,
                    "after_score": dec.after_score,
                },
            }
            w.write(row)
            rows.append(row)

    metrics = compute_metrics(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    return metrics
