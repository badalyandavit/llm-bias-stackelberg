# bias-stackelberg

Repository scaffold (Milestone 0).

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```

## Detox pipeline (end-to-end)

All steps run from the repo root with the venv activated.

```bash
# 1) Build ParaDetox JSONL (optional if you already have it)
bias-stackelberg make-paradetox --out-jsonl data/paradetox_1k.jsonl --n 1000 --seed 0

# 2) Generate Option-A predictions (oracle rewrite) to build SFT
bias-stackelberg eval-a-file \
  --in-jsonl data/paradetox_1k.jsonl \
  --out-dir runs/m9_eval_a_paradetox \
  --leader hf-toxic \
  --hf-score-mode any \
  --gen-backend hf \
  --gen-model distilgpt2 \
  --trigger-threshold 0.01 \
  --reference-rewrite

# 3) Build SFT dataset (uses rewrite template)
bias-stackelberg build-sft \
  --in-predictions runs/m9_eval_a_paradetox/predictions.jsonl \
  --out-dir runs/m9_sft_paradetox

# 4) Train LoRA
bias-stackelberg train-lora \
  --sft-jsonl runs/m9_sft_paradetox/sft.jsonl \
  --out-dir runs/m9_lora_distilgpt2 \
  --model-name distilgpt2 \
  --max-length 512 \
  --seed 0 \
  --max-steps 200 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --grad-accum 1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --target-modules c_attn,c_proj

# 5) Detox eval (base)
bias-stackelberg eval-detox-file \
  --in-jsonl data/paradetox_1k.jsonl \
  --out-dir runs/m9_detox_base_v2 \
  --leader hf-toxic \
  --hf-score-mode any \
  --gen-backend hf \
  --gen-model distilgpt2 \
  --temperature 0.0 \
  --top-p 1.0 \
  --min-new-tokens 8

# 6) Detox eval (LoRA)
bias-stackelberg eval-detox-file \
  --in-jsonl data/paradetox_1k.jsonl \
  --out-dir runs/m9_detox_lora_v2 \
  --leader hf-toxic \
  --hf-score-mode any \
  --gen-backend hf \
  --gen-model distilgpt2 \
  --adapter-dir runs/m9_lora_distilgpt2/adapter \
  --temperature 0.0 \
  --top-p 1.0 \
  --min-new-tokens 8

# 7) Gate check
bias-stackelberg gate-detox \
  --base-metrics runs/m9_detox_base_v2/metrics.json \
  --tuned-metrics runs/m9_detox_lora_v2/metrics.json
```

## Prompting alignment

Detox training and inference use the same rewrite template (see `src/bias_stackelberg/core/prompts.py`).
SFT prompts are built from the raw toxic text (`y0_text`) using this template, and detox eval metrics are computed
against the raw text, not the templated prompt.

## CI

GitHub Actions runs `pytest` and `ruff` on every push and pull request.
