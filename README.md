# HerO × TextGrad — Frozen-Baseline Reproduction

The project applies **Native TextGrad** to existing, frozen HerO Veracity Prediction (VP) outputs. The TextGrad path does **not** rerun initial VP. The original HerO reproduction pipeline is kept separately.

## Quick start (Server01)

```bash
conda activate llama70b-bw
bash scripts/check_env.sh

# Qwen3.5-9B TextGrad optimization on frozen VP
bash run.sh qwen --count 1 --claim-id 59 --af human63 --run-id demo_1 --freeze
bash run.sh qwen --count 10 --af human63 --run-id demo_10 --freeze
bash run.sh qwen --count 40 --af human63 --run-id demo_40 --freeze
bash run.sh qwen --count 169 --af original --run-id demo_169 --freeze

# Compare frozen baseline and updated predictions offline
bash run.sh compare --run-id demo_1
```

The evaluator uses prompts in `prompts/` and the two demonstrations in `fewshot/fixed2.jsonl` by default; `fixed6.jsonl` is kept as an alternative. AF-split inputs are available for the Targeted40 set. Full169 / Full500 use existing 10-QA evidence. See `README.txt` and `导师使用说明.md` for environment-specific details when present.

Large model weights, server secrets, and generated `outputs/` are **not tracked in Git**. The original HerO pipeline requires its own external model services and retrieval data.
