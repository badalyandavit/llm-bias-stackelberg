from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bias_stackelberg.core.types import BiasReport, Generation
from bias_stackelberg.follower.base import FollowerDecision
from bias_stackelberg.models.base import BaseLLM, GenConfig


@dataclass(frozen=True)
class OptionAConfig:
    trigger_threshold: float = 0.2
    max_attempts: int = 1
    accept_if_decrease: bool = True
    rewrite_max_tokens: int = 192
    rewrite_temperature: float = 0.2
    rewrite_top_p: float = 1.0


def _rewrite_prompt(x: str, y: str, report: BiasReport) -> str:
    cats = ", ".join(report.categories) if report.categories else "none"
    spans = "; ".join(report.evidence_spans[:3]) if report.evidence_spans else "none"
    return (
        "You are a careful assistant.\n"
        "Task: rewrite the assistant answer to remove biased or stereotyped language "
        "while preserving meaning.\n"
        "Constraints:\n"
        "- Keep the same intent and factual content.\n"
        "- Do not add new facts.\n"
        "- Be concise.\n\n"
        f"User prompt:\n{x}\n\n"
        f"Original answer:\n{y}\n\n"
        f"Detected bias categories: {cats}\n"
        f"Evidence spans: {spans}\n\n"
        "Rewrite the answer now."
    )


class OptionAFollower:
    def __init__(self, *, leader: Any, llm: BaseLLM, config: OptionAConfig | None = None) -> None:
        self.leader = leader
        self.llm = llm
        self.cfg = config or OptionAConfig()

    def intervene(self, x: str, y: str, report: BiasReport) -> tuple[Generation, FollowerDecision]:
        before = float(report.score)

        if before < self.cfg.trigger_threshold:
            gen = Generation(text=y, meta={"follower": "option_a", "action": "accept_noop"})
            dec = FollowerDecision(
                action="accept",
                reason="below_threshold",
                before_score=before,
                after_score=before,
            )
            return gen, dec

        best_gen: Generation | None = None
        best_score = before

        for attempt in range(max(1, self.cfg.max_attempts)):
            prompt = _rewrite_prompt(x, y, report)
            gen = self.llm.generate(
                prompt,
                config=GenConfig(
                    seed=0,
                    max_tokens=self.cfg.rewrite_max_tokens,
                    temperature=self.cfg.rewrite_temperature,
                    top_p=self.cfg.rewrite_top_p,
                ),
            )

            new_report = self.leader.score(x, gen.text)
            after = float(new_report.score)

            if after < best_score:
                best_score = after
                best_gen = Generation(
                    text=gen.text,
                    meta={
                        "follower": "option_a",
                        "action": "rewrite",
                        "attempt": attempt,
                        "before_score": before,
                        "after_score": after,
                        "categories": list(new_report.categories),
                    },
                )

            if self.cfg.accept_if_decrease and best_gen is not None and best_score < before:
                dec = FollowerDecision(
                    action="rewrite",
                    reason="score_decreased",
                    before_score=before,
                    after_score=best_score,
                )
                return best_gen, dec

        gen = Generation(text=y, meta={"follower": "option_a", "action": "abstain"})
        dec = FollowerDecision(
            action="abstain",
            reason="no_improving_rewrite",
            before_score=before,
            after_score=None,
        )
        return gen, dec
