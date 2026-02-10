from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bias_stackelberg.core.jsonl import JsonlWriter


def _lazy_import_datasets() -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing optional deps for HF datasets. Install with: pip install -e '.[hf]'"
        ) from e
    return load_dataset


@dataclass(frozen=True)
class MakeParaDetoxConfig:
    out_jsonl: str
    n: int = 1000
    seed: int = 0


def make_paradetox_jsonl(cfg: MakeParaDetoxConfig) -> dict[str, Any]:
    load_dataset = _lazy_import_datasets()
    ds = load_dataset("s-nlp/paradetox")["train"]  # type: ignore

    n = min(int(cfg.n), len(ds))
    ds = ds.shuffle(seed=int(cfg.seed)).select(range(n))

    out_path = Path(cfg.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with JsonlWriter(out_path) as w:
        for i, row in enumerate(ds):
            toxic = str(row["en_toxic_comment"])
            neutral = str(row["en_neutral_comment"])
            rec = {
                "id": f"paradetox::{i}",
                "prompt": toxic,
                "meta": {
                    "dataset": "paradetox",
                    "y0_text": toxic,
                    "reference_text": neutral,
                },
            }
            w.write(rec)
            kept += 1

    return {"out_jsonl": str(out_path), "n": kept, "seed": int(cfg.seed)}
