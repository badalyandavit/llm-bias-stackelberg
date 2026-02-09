from __future__ import annotations

import argparse
from pathlib import Path

from bias_stackelberg.data.toy import toy_examples
from bias_stackelberg.eval.runner import EvalAConfig, run_option_a
from bias_stackelberg.follower.option_a import OptionAConfig
from bias_stackelberg.models import GenConfig


def _cmd_eval_a(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalAConfig(
        out_dir=str(out_dir),
        gen=GenConfig(
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
        follower=OptionAConfig(trigger_threshold=args.trigger_threshold),
    )

    m = run_option_a(toy_examples(), cfg=cfg)
    print(f"wrote: {out_dir / 'predictions.jsonl'}")
    print(f"wrote: {out_dir / 'metrics.json'}")
    print(m.to_dict())


def main() -> None:
    p = argparse.ArgumentParser(prog="bias-stackelberg")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("eval-a", help="Run Option A end-to-end on toy data")
    a.add_argument("--out-dir", default="runs/m5_eval_a")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--max-tokens", type=int, default=64)
    a.add_argument("--temperature", type=float, default=0.7)
    a.add_argument("--top-p", type=float, default=1.0)
    a.add_argument("--trigger-threshold", type=float, default=0.2)
    a.set_defaults(func=_cmd_eval_a)

    args = p.parse_args()
    args.func(args)
