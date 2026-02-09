from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from bias_stackelberg.core.types import Generation


@dataclass(frozen=True)
class GenConfig:
    seed: int = 0
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    stop: list[str] | None = None


class BaseLLM(Protocol):
    def generate(
        self, prompt: str, *, config: GenConfig | None = None, **kwargs: Any
    ) -> Generation: ...
