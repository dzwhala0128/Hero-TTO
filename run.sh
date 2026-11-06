#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1 TEXTGRAD_DISABLE_LOGGING=1
if [[ -f .env ]]; then set -a; source .env; set +a; fi
mkdir -p outputs
case "${1:-}" in
  qwen|textgrad)
    shift; exec python -B -u -m pipeline.unified "$@" ;;
  hero)
    shift; exec python -B -u -m pipeline.runner baseline --split full500 --mode full-pipeline "$@" ;;
  evaluate)
    shift; exec python -B -m pipeline.evaluate "$@" ;;
  compare)
    shift; exec python -B -m pipeline.compare_metrics "$@" ;;
  status|freeze)
    exec python -B -m pipeline.runner "$@" ;;
  *)
    cat <<'HELP'
TextGrad只优化已有冻结VP，不重跑初始预测。
  bash run.sh qwen --count 1 --claim-id 9 --run-id demo1 --freeze
  bash run.sh qwen --count 10 --run-id demo10 --freeze
  bash run.sh qwen --count 40 --af human63 --run-id demo40 --freeze
  bash run.sh qwen --count 169 --run-id demo169 --freeze
  bash run.sh qwen --count 500 --run-id demo500 --freeze
  bash run.sh qwen --count 500 --preflight       # 零模型调用
  bash run.sh compare --run-id demo1              # 先冻结再算离线指标
  bash run.sh hero --claim-id 0 --preflight       # 单独的原始HerO复现

HELP
    exit 2 ;;
esac
