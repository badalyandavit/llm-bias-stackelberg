from __future__ import annotations

from pathlib import Path

from bias_stackelberg.core.jsonl import iter_jsonl
from bias_stackelberg.core.types import Example


def load_examples_jsonl(path: str | Path) -> list[Example]:
    p = Path(path)
    out: list[Example] = []

    for i, row in enumerate(iter_jsonl(p), start=1):
        ex_id = row.get("id")
        prompt = row.get("prompt", row.get("text"))
        meta = row.get("meta", {})

        if ex_id is None:
            ex_id = f"line_{i}"
        if not isinstance(ex_id, str) or not ex_id:
            raise ValueError(f"Invalid id on line {i} in {p}")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Invalid prompt/text on line {i} in {p}")

        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise TypeError(f"meta must be a dict on line {i} in {p}")

        out.append(Example(id=ex_id, prompt=prompt, meta=meta))

    if not out:
        raise ValueError(f"No examples found in {p}")

    return out
