from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, obj: Any) -> None:
        if is_dataclass(obj):
            if hasattr(obj, "to_dict"):
                payload = obj.to_dict()
            else:
                raise TypeError("Dataclass objects must implement to_dict() to be logged")
        elif isinstance(obj, dict):
            payload = obj
        else:
            raise TypeError("JsonlWriter.write expects a dict or a dataclass with to_dict()")

        s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._fh.write(s + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {p}") from e
            if not isinstance(obj, dict):
                raise TypeError(f"Expected dict JSON object on line {i} in {p}")
            yield obj
