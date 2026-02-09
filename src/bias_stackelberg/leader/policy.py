from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    pattern: str
    weight: float

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, flags=re.IGNORECASE)


@dataclass(frozen=True)
class LeaderPolicy:
    """
    Human-readable leader policy: a small set of weighted regex rules.

    score = clip(sum(weight for fired rules), 0, 1)
    """

    rules: list[Rule]
    min_evidence_len: int = 3

    def validate(self) -> None:
        if not self.rules:
            raise ValueError("LeaderPolicy.rules must be non-empty")
        for r in self.rules:
            if not r.id:
                raise ValueError("Rule.id must be non-empty")
            if not r.category:
                raise ValueError("Rule.category must be non-empty")
            if r.weight < 0.0:
                raise ValueError("Rule.weight must be >= 0")
            _ = r.compiled()

    def to_trace(self) -> dict[str, Any]:
        return {
            "policy_type": "weighted_regex_rules",
            "num_rules": len(self.rules),
            "min_evidence_len": self.min_evidence_len,
            "rules": [
                {"id": r.id, "category": r.category, "pattern": r.pattern, "weight": r.weight}
                for r in self.rules
            ],
        }
