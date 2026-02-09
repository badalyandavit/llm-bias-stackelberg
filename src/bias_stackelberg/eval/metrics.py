from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunMetrics:
    n: int
    mean_before: float
    mean_after: float
    mean_delta: float
    rewrite_rate: float
    abstain_rate: float
    accept_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "mean_delta": self.mean_delta,
            "rewrite_rate": self.rewrite_rate,
            "abstain_rate": self.abstain_rate,
            "accept_rate": self.accept_rate,
        }


def compute_metrics(rows: list[dict[str, Any]]) -> RunMetrics:
    if not rows:
        raise ValueError("rows must be non-empty")

    n = len(rows)
    before = [float(r["before"]["score"]) for r in rows]
    after = [float(r["after"]["score"]) for r in rows]
    delta = [a - b for a, b in zip(after, before, strict=True)]

    rewrites = sum(1 for r in rows if r["decision"]["action"] == "rewrite")
    abstains = sum(1 for r in rows if r["decision"]["action"] == "abstain")
    accepts = sum(1 for r in rows if r["decision"]["action"] == "accept")

    mean_before = sum(before) / n
    mean_after = sum(after) / n
    mean_delta = sum(delta) / n

    return RunMetrics(
        n=n,
        mean_before=mean_before,
        mean_after=mean_after,
        mean_delta=mean_delta,
        rewrite_rate=rewrites / n,
        abstain_rate=abstains / n,
        accept_rate=accepts / n,
    )
