from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var**0.5


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        raise ValueError("xs must be non-empty")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    ys = sorted(xs)
    n = len(ys)
    if n == 1:
        return ys[0]
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def _count_by(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


@dataclass(frozen=True)
class RunMetrics:
    n: int

    mean_before: float
    mean_after: float
    mean_delta: float

    std_before: float
    std_after: float
    std_delta: float

    before_min: float
    before_max: float
    after_min: float
    after_max: float

    delta_p10: float
    delta_p50: float
    delta_p90: float

    improved_rate: float
    worsened_rate: float
    unchanged_rate: float

    accept_rate: float
    rewrite_rate: float
    abstain_rate: float

    action_counts: dict[str, int]
    category_counts_before: dict[str, int]
    category_counts_after: dict[str, int]
    category_metrics: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "mean_delta": self.mean_delta,
            "std_before": self.std_before,
            "std_after": self.std_after,
            "std_delta": self.std_delta,
            "before_min": self.before_min,
            "before_max": self.before_max,
            "after_min": self.after_min,
            "after_max": self.after_max,
            "delta_p10": self.delta_p10,
            "delta_p50": self.delta_p50,
            "delta_p90": self.delta_p90,
            "improved_rate": self.improved_rate,
            "worsened_rate": self.worsened_rate,
            "unchanged_rate": self.unchanged_rate,
            "accept_rate": self.accept_rate,
            "rewrite_rate": self.rewrite_rate,
            "abstain_rate": self.abstain_rate,
            "action_counts": self.action_counts,
            "category_counts_before": self.category_counts_before,
            "category_counts_after": self.category_counts_after,
            "category_metrics": self.category_metrics,
        }


def compute_metrics(rows: list[dict[str, Any]]) -> RunMetrics:
    if not rows:
        raise ValueError("rows must be non-empty")

    n = len(rows)

    before = [float(r["before"]["score"]) for r in rows]
    after = [float(r["after"]["score"]) for r in rows]
    delta = [a - b for a, b in zip(after, before, strict=True)]

    actions = [str(r["decision"]["action"]) for r in rows]
    action_counts = _count_by(actions)

    rewrites = action_counts.get("rewrite", 0)
    abstains = action_counts.get("abstain", 0)
    accepts = action_counts.get("accept", 0)

    eps = 1e-12
    improved = sum(1 for d in delta if d < -eps)
    worsened = sum(1 for d in delta if d > eps)
    unchanged = n - improved - worsened

    cats_before: list[str] = []
    cats_after: list[str] = []
    for r in rows:
        cb = r["before"].get("categories", [])
        ca = r["after"].get("categories", [])
        if isinstance(cb, list):
            cats_before.extend([c for c in cb if isinstance(c, str)])
        if isinstance(ca, list):
            cats_after.extend([c for c in ca if isinstance(c, str)])

    category_counts_before = _count_by(cats_before)
    category_counts_after = _count_by(cats_after)

    all_cats = sorted(set(category_counts_before) | set(category_counts_after))
    category_metrics: dict[str, dict[str, float | int]] = {}

    for c in all_cats:
        idxs = [i for i, r in enumerate(rows) if c in (r["before"].get("categories") or [])]
        if not idxs:
            continue
        b = [before[i] for i in idxs]
        a = [after[i] for i in idxs]
        d = [delta[i] for i in idxs]
        category_metrics[c] = {
            "n": len(idxs),
            "mean_before": _mean(b),
            "mean_after": _mean(a),
            "mean_delta": _mean(d),
            "improved_rate": sum(1 for x in d if x < -eps) / len(d),
        }

    return RunMetrics(
        n=n,
        mean_before=_mean(before),
        mean_after=_mean(after),
        mean_delta=_mean(delta),
        std_before=_std(before),
        std_after=_std(after),
        std_delta=_std(delta),
        before_min=min(before),
        before_max=max(before),
        after_min=min(after),
        after_max=max(after),
        delta_p10=_quantile(delta, 0.10),
        delta_p50=_quantile(delta, 0.50),
        delta_p90=_quantile(delta, 0.90),
        improved_rate=improved / n,
        worsened_rate=worsened / n,
        unchanged_rate=unchanged / n,
        accept_rate=accepts / n,
        rewrite_rate=rewrites / n,
        abstain_rate=abstains / n,
        action_counts=action_counts,
        category_counts_before=category_counts_before,
        category_counts_after=category_counts_after,
        category_metrics=category_metrics,
    )
