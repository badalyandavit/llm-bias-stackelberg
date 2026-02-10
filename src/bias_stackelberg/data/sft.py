from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bias_stackelberg.core.jsonl import JsonlWriter, iter_jsonl
from bias_stackelberg.core.prompts import detox_rewrite_prompt


@dataclass(frozen=True)
class BuildSftConfig:
    in_predictions: str
    out_dir: str
    min_improvement: float = 1e-9
    min_chars: int = 1
    max_chars: int = 10_000
    require_text_change: bool = True
    require_action_rewrite: bool = True


def _get_score(d: dict[str, Any], key: str) -> float:
    try:
        return float(d[key]["score"])
    except Exception as e:
        raise ValueError(f"Missing or invalid score at {key}.score") from e


def _get_text(d: dict[str, Any], key: str) -> str:
    try:
        t = d[key]["text"]
    except Exception as e:
        raise ValueError(f"Missing text at {key}.text") from e
    if not isinstance(t, str):
        raise TypeError(f"{key}.text must be str")
    return t


def _get_action(d: dict[str, Any]) -> str:
    try:
        a = d["decision"]["action"]
    except Exception as e:
        raise ValueError("Missing decision.action") from e
    if not isinstance(a, str):
        raise TypeError("decision.action must be str")
    return a


def build_sft_dataset(*, cfg: BuildSftConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_sft = out_dir / "sft.jsonl"
    out_ref = out_dir / "reference.jsonl"
    out_drop = out_dir / "dropped.jsonl"
    out_sum = out_dir / "build_summary.json"

    kept = 0
    dropped = 0

    with (
        JsonlWriter(out_sft) as w_sft,
        JsonlWriter(out_ref) as w_ref,
        JsonlWriter(out_drop) as w_drop,
    ):
        for row in iter_jsonl(cfg.in_predictions):
            ex_id = row.get("id")
            prompt = row.get("prompt")
            meta = row.get("meta", {})

            if not isinstance(ex_id, str) or not ex_id:
                w_drop.write({"id": ex_id, "reason": "invalid_id"})
                dropped += 1
                continue

            if not isinstance(prompt, str):
                w_drop.write({"id": ex_id, "reason": "invalid_prompt"})
                dropped += 1
                continue

            try:
                before = _get_score(row, "before")
                after = _get_score(row, "after")
                y0 = _get_text(row, "y0")
                y1 = _get_text(row, "y1")
                action = _get_action(row)
            except Exception as e:
                w_drop.write({"id": ex_id, "reason": "invalid_row", "error": str(e)})
                dropped += 1
                continue

            w_ref.write(
                {
                    "id": ex_id,
                    "prompt": prompt,
                    "baseline_response": y0,
                    "before_score": before,
                    "meta": meta,
                }
            )

            if cfg.require_action_rewrite and action != "rewrite":
                w_drop.write(
                    {
                        "id": ex_id,
                        "reason": "not_rewrite_action",
                        "action": action,
                        "before_score": before,
                        "after_score": after,
                    }
                )
                dropped += 1
                continue

            if (before - after) < cfg.min_improvement:
                w_drop.write(
                    {
                        "id": ex_id,
                        "reason": "insufficient_improvement",
                        "before_score": before,
                        "after_score": after,
                        "min_improvement": cfg.min_improvement,
                    }
                )
                dropped += 1
                continue

            y1s = y1.strip()
            if len(y1s) < cfg.min_chars:
                w_drop.write({"id": ex_id, "reason": "too_short", "chars": len(y1s)})
                dropped += 1
                continue

            if len(y1s) > cfg.max_chars:
                w_drop.write({"id": ex_id, "reason": "too_long", "chars": len(y1s)})
                dropped += 1
                continue

            if cfg.require_text_change and y1.strip() == y0.strip():
                w_drop.write({"id": ex_id, "reason": "no_text_change"})
                dropped += 1
                continue

            rewrite_prompt = detox_rewrite_prompt(y0)

            w_sft.write(
                {
                    "id": ex_id,
                    "prompt": rewrite_prompt,
                    "completion": y1,
                    "meta": meta,
                    "provenance": {
                        "source": "option_a",
                        "before_score": before,
                        "after_score": after,
                        "decision": row.get("decision", {}),
                        "source_prompt": prompt,
                        "source_text": y0,
                        "prompt_template": "detox_rewrite_v1",
                    },
                }
            )
            kept += 1

    summary = {
        "in_predictions": str(Path(cfg.in_predictions)),
        "out_dir": str(out_dir),
        "kept": kept,
        "dropped": dropped,
        "filters": {
            "min_improvement": cfg.min_improvement,
            "min_chars": cfg.min_chars,
            "max_chars": cfg.max_chars,
            "require_text_change": cfg.require_text_change,
            "require_action_rewrite": cfg.require_action_rewrite,
        },
        "outputs": {
            "sft_jsonl": str(out_sft),
            "reference_jsonl": str(out_ref),
            "dropped_jsonl": str(out_drop),
            "summary_json": str(out_sum),
        },
    }
    out_sum.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
