from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    q = max(0.0, min(1.0, q))
    i = int(round(q * (len(s) - 1)))
    return float(s[i])


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(v**0.5)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class DetoxMetrics:
    n: int

    mean_before: float
    mean_after: float
    mean_delta: float

    std_before: float
    std_after: float
    std_delta: float

    before_p50: float
    after_p50: float
    delta_p10: float
    delta_p50: float
    delta_p90: float

    improved_rate: float
    worsened_rate: float
    unchanged_rate: float

    empty_rate: float
    copy_rate: float

    len_ratio_p10: float
    len_ratio_p50: float
    len_ratio_p90: float

    edit_norm_p10: float
    edit_norm_p50: float
    edit_norm_p90: float

    category_counts_before: dict[str, int]
    category_counts_after: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "mean_delta": self.mean_delta,
            "std_before": self.std_before,
            "std_after": self.std_after,
            "std_delta": self.std_delta,
            "before_p50": self.before_p50,
            "after_p50": self.after_p50,
            "delta_p10": self.delta_p10,
            "delta_p50": self.delta_p50,
            "delta_p90": self.delta_p90,
            "improved_rate": self.improved_rate,
            "worsened_rate": self.worsened_rate,
            "unchanged_rate": self.unchanged_rate,
            "empty_rate": self.empty_rate,
            "copy_rate": self.copy_rate,
            "len_ratio_p10": self.len_ratio_p10,
            "len_ratio_p50": self.len_ratio_p50,
            "len_ratio_p90": self.len_ratio_p90,
            "edit_norm_p10": self.edit_norm_p10,
            "edit_norm_p50": self.edit_norm_p50,
            "edit_norm_p90": self.edit_norm_p90,
            "category_counts_before": self.category_counts_before,
            "category_counts_after": self.category_counts_after,
        }


def compute_detox_metrics(rows: list[dict[str, Any]]) -> DetoxMetrics:
    n = len(rows)

    before = [float(r["before"]["score"]) for r in rows]
    after = [float(r["after"]["score"]) for r in rows]
    delta = [a - b for a, b in zip(after, before, strict=False)]

    improved = sum(1 for d in delta if d < 0.0)
    worsened = sum(1 for d in delta if d > 0.0)
    unchanged = n - improved - worsened

    empty = sum(1 for r in rows if not str(r["y_gen"]["text"]).strip())
    copy = sum(1 for r in rows if str(r["y_gen"]["text"]).strip() == str(r["prompt_text"]).strip())

    len_ratios = [float(r["constraints"]["len_ratio"]) for r in rows]
    edit_norms = [float(r["constraints"]["edit_norm"]) for r in rows]

    cat_before: dict[str, int] = {}
    cat_after: dict[str, int] = {}

    for r in rows:
        for c in r["before"].get("categories", []):
            cat_before[c] = cat_before.get(c, 0) + 1
        for c in r["after"].get("categories", []):
            cat_after[c] = cat_after.get(c, 0) + 1

    return DetoxMetrics(
        n=n,
        mean_before=_mean(before),
        mean_after=_mean(after),
        mean_delta=_mean(delta),
        std_before=_std(before),
        std_after=_std(after),
        std_delta=_std(delta),
        before_p50=_quantile(before, 0.50),
        after_p50=_quantile(after, 0.50),
        delta_p10=_quantile(delta, 0.10),
        delta_p50=_quantile(delta, 0.50),
        delta_p90=_quantile(delta, 0.90),
        improved_rate=float(improved / n) if n else 0.0,
        worsened_rate=float(worsened / n) if n else 0.0,
        unchanged_rate=float(unchanged / n) if n else 0.0,
        empty_rate=float(empty / n) if n else 0.0,
        copy_rate=float(copy / n) if n else 0.0,
        len_ratio_p10=_quantile(len_ratios, 0.10),
        len_ratio_p50=_quantile(len_ratios, 0.50),
        len_ratio_p90=_quantile(len_ratios, 0.90),
        edit_norm_p10=_quantile(edit_norms, 0.10),
        edit_norm_p50=_quantile(edit_norms, 0.50),
        edit_norm_p90=_quantile(edit_norms, 0.90),
        category_counts_before=cat_before,
        category_counts_after=cat_after,
    )


def constraint_stats(prompt_text: str, gen_text: str) -> dict[str, float]:
    p = prompt_text.strip()
    g = gen_text.strip()

    lp = max(1, len(p))
    lg = len(g)
    len_ratio = float(lg / lp)

    dist = _levenshtein(p, g)
    edit_norm = float(dist / max(1, max(len(p), len(g))))

    return {"len_ratio": len_ratio, "edit_norm": edit_norm}
