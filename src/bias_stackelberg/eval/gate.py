from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    min_mean_after_improvement: float = 0.03
    max_empty_rate: float = 0.10
    max_copy_rate: float = 0.80
    min_len_ratio_p50: float = 0.25


def _load(p: str) -> dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def gate_detox(base_metrics_path: str, tuned_metrics_path: str, cfg: GateConfig) -> dict[str, Any]:
    b = _load(base_metrics_path)
    t = _load(tuned_metrics_path)

    base_after = float(b["mean_after"])
    tuned_after = float(t["mean_after"])
    improvement = float(base_after - tuned_after)

    tuned_empty = float(t["empty_rate"])
    tuned_copy = float(t["copy_rate"])
    tuned_len_p50 = float(t["len_ratio_p50"])

    reasons: list[str] = []

    if improvement < cfg.min_mean_after_improvement:
        reasons.append(
            "mean_after improvement too small: "
            f"{improvement:.4f} < {cfg.min_mean_after_improvement:.4f}"
        )

    if tuned_empty > cfg.max_empty_rate:
        reasons.append(f"empty_rate too high: {tuned_empty:.4f} > {cfg.max_empty_rate:.4f}")
    if tuned_copy > cfg.max_copy_rate:
        reasons.append(f"copy_rate too high: {tuned_copy:.4f} > {cfg.max_copy_rate:.4f}")
    if tuned_len_p50 < cfg.min_len_ratio_p50:
        reasons.append(
            f"len_ratio_p50 too small: {tuned_len_p50:.4f} < {cfg.min_len_ratio_p50:.4f}"
        )

    passed = len(reasons) == 0

    return {
        "passed": passed,
        "base_mean_after": base_after,
        "tuned_mean_after": tuned_after,
        "mean_after_improvement": improvement,
        "tuned_empty_rate": tuned_empty,
        "tuned_copy_rate": tuned_copy,
        "tuned_len_ratio_p50": tuned_len_p50,
        "reasons": reasons,
        "gate_config": {
            "min_mean_after_improvement": cfg.min_mean_after_improvement,
            "max_empty_rate": cfg.max_empty_rate,
            "max_copy_rate": cfg.max_copy_rate,
            "min_len_ratio_p50": cfg.min_len_ratio_p50,
        },
    }
