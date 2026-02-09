from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bias_stackelberg.core.types import BiasReport, Generation


@dataclass(frozen=True)
class FollowerDecision:
    action: str  # "accept" | "rewrite" | "abstain"
    reason: str
    before_score: float
    after_score: float | None = None


class Follower(Protocol):
    def intervene(
        self, x: str, y: str, report: BiasReport
    ) -> tuple[Generation, FollowerDecision]: ...
