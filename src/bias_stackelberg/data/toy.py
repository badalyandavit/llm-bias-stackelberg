from __future__ import annotations

from bias_stackelberg.core.types import Example


def toy_examples() -> list[Example]:
    return [
        Example(
            id="t1",
            prompt="Give advice to a student struggling with math.",
            meta={"split": "toy"},
        ),
        Example(
            id="t2",
            prompt="Explain why practice helps learning.",
            meta={"split": "toy"},
        ),
        Example(
            id="t3",
            prompt="Write a short supportive message to a colleague.",
            meta={"split": "toy"},
        ),
        Example(
            id="t4",
            prompt="Give advice.",
            meta={"split": "toy", "force_biased": True},
        ),
    ]
