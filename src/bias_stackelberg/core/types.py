from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _is_json_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def _is_jsonable(x: Any) -> bool:
    if _is_json_scalar(x):
        return True
    if isinstance(x, list):
        return all(_is_jsonable(v) for v in x)
    if isinstance(x, dict):
        return all(isinstance(k, str) and _is_jsonable(v) for k, v in x.items())
    return False


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Example.id must be non-empty")
        if not isinstance(self.prompt, str):
            raise TypeError("Example.prompt must be a string")
        if not isinstance(self.meta, dict):
            raise TypeError("Example.meta must be a dict")
        if not _is_jsonable(self.meta):
            raise ValueError("Example.meta must be JSON-serializable")


@dataclass(frozen=True)
class Generation:
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Generation.text must be a string")
        if not isinstance(self.meta, dict):
            raise TypeError("Generation.meta must be a dict")
        if not _is_jsonable(self.meta):
            raise ValueError("Generation.meta must be JSON-serializable")


@dataclass(frozen=True)
class BiasReport:
    score: float
    categories: list[str]
    rationale: str
    evidence_spans: list[str]
    trace: dict[str, Any]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.score, (int, float)):
            raise TypeError("BiasReport.score must be a number")
        if not math.isfinite(float(self.score)):
            raise ValueError("BiasReport.score must be finite")
        if float(self.score) < 0.0 or float(self.score) > 1.0:
            raise ValueError("BiasReport.score must be in [0, 1]")

        if not isinstance(self.categories, list) or not all(
            isinstance(c, str) for c in self.categories
        ):
            raise TypeError("BiasReport.categories must be a list[str]")

        if not isinstance(self.rationale, str):
            raise TypeError("BiasReport.rationale must be a string")

        if not isinstance(self.evidence_spans, list) or not all(
            isinstance(s, str) for s in self.evidence_spans
        ):
            raise TypeError("BiasReport.evidence_spans must be a list[str]")

        if not isinstance(self.trace, dict):
            raise TypeError("BiasReport.trace must be a dict[str, Any]")
        if not _is_jsonable(self.trace):
            raise ValueError("BiasReport.trace must be JSON-serializable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "categories": list(self.categories),
            "rationale": self.rationale,
            "evidence_spans": list(self.evidence_spans),
            "trace": dict(self.trace),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> BiasReport:
        BiasReport.validate_dict(d)
        return BiasReport(
            score=float(d["score"]),
            categories=list(d["categories"]),
            rationale=str(d["rationale"]),
            evidence_spans=list(d["evidence_spans"]),
            trace=dict(d["trace"]),
        )

    @staticmethod
    def validate_dict(d: dict[str, Any]) -> None:
        if not isinstance(d, dict):
            raise TypeError("BiasReport dict must be a dict")

        required = ["score", "categories", "rationale", "evidence_spans", "trace"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"BiasReport dict missing keys: {missing}")

        score = d["score"]
        if not isinstance(score, (int, float)):
            raise TypeError("BiasReport.score must be a number")
        if not math.isfinite(float(score)):
            raise ValueError("BiasReport.score must be finite")
        if float(score) < 0.0 or float(score) > 1.0:
            raise ValueError("BiasReport.score must be in [0, 1]")

        cats = d["categories"]
        if not isinstance(cats, list) or not all(isinstance(c, str) for c in cats):
            raise TypeError("BiasReport.categories must be a list[str]")

        if not isinstance(d["rationale"], str):
            raise TypeError("BiasReport.rationale must be a string")

        spans = d["evidence_spans"]
        if not isinstance(spans, list) or not all(isinstance(s, str) for s in spans):
            raise TypeError("BiasReport.evidence_spans must be a list[str]")

        tr = d["trace"]
        if not isinstance(tr, dict):
            raise TypeError("BiasReport.trace must be a dict[str, Any]")
        if not _is_jsonable(tr):
            raise ValueError("BiasReport.trace must be JSON-serializable")
