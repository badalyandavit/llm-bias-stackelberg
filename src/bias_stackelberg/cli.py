from __future__ import annotations

import argparse
from pathlib import Path

from bias_stackelberg.data.sft import BuildSftConfig, build_sft_dataset
from bias_stackelberg.data.toy import toy_examples
from bias_stackelberg.eval.runner import EvalAConfig, run_option_a
from bias_stackelberg.follower.option_a import OptionAConfig
from bias_stackelberg.models import GenConfig
from bias_stackelberg.train.lora_sft import TrainLoRAConfig, train_lora_sft


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


def _cmd_build_sft(args: argparse.Namespace) -> None:
    cfg = BuildSftConfig(
        in_predictions=args.in_predictions,
        out_dir=args.out_dir,
        min_improvement=args.min_improvement,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        require_text_change=not args.allow_no_text_change,
        require_action_rewrite=not args.allow_non_rewrite_action,
    )
    summary = build_sft_dataset(cfg=cfg)
    print(summary)


def _cmd_train_lora(args: argparse.Namespace) -> None:
    target_modules = tuple(x.strip() for x in args.target_modules.split(",") if x.strip())

    cfg = TrainLoRAConfig(
        sft_jsonl=args.sft_jsonl,
        out_dir=args.out_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        seed=args.seed,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    summary = train_lora_sft(cfg=cfg)
    print(summary)


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

    b = sub.add_parser("build-sft", help="Build SFT dataset from predictions.jsonl")
    b.add_argument("--in-predictions", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--min-improvement", type=float, default=1e-9)
    b.add_argument("--min-chars", type=int, default=1)
    b.add_argument("--max-chars", type=int, default=10_000)
    b.add_argument("--allow-no-text-change", action="store_true")
    b.add_argument("--allow-non-rewrite-action", action="store_true")
    b.set_defaults(func=_cmd_build_sft)

    t = sub.add_parser("train-lora", help="Train LoRA adapter from sft.jsonl")
    t.add_argument("--sft-jsonl", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--model-name", default="distilgpt2")
    t.add_argument("--max-length", type=int, default=512)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-steps", type=int, default=50)
    t.add_argument("--learning-rate", type=float, default=2e-4)
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--lora-r", type=int, default=8)
    t.add_argument("--lora-alpha", type=int, default=16)
    t.add_argument("--lora-dropout", type=float, default=0.05)
    t.add_argument("--target-modules", default="c_attn")
    t.set_defaults(func=_cmd_train_lora)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
