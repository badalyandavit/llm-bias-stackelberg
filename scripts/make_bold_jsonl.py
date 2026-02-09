from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domains", default="gender,race,religion,political_ideology,profession")
    args = ap.parse_args()

    domains = {d.strip() for d in args.domains.split(",") if d.strip()}
    ds = load_dataset("AlexaAI/bold")["train"]

    prompts: list[dict] = []
    for row in ds:
        domain = row.get("domain")
        if domain not in domains:
            continue
        category = row.get("category")
        name = row.get("name")
        ps = row.get("prompts") or []
        for j, p in enumerate(ps):
            if not isinstance(p, str) or not p.strip():
                continue
            prompts.append(
                {
                    "id": f"bold::{domain}::{category}::{name}::{j}",
                    "prompt": p,
                    "meta": {
                        "dataset": "bold",
                        "domain": domain,
                        "category": category,
                        "name": name,
                    },
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(prompts)
    prompts = prompts[: args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in prompts:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print({"out": str(out), "n": len(prompts), "domains": sorted(domains)})


if __name__ == "__main__":
    main()
