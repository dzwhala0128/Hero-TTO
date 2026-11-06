#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
role=${1:-hero}
: "${GPU_IDS:?Set explicit GPU_IDS, e.g. GPU_IDS=0; no default GPU allocation}"
case "$role" in
 hero) : "${HERO_MODEL_PATH:?Set HERO_MODEL_PATH}"; model_path=$HERO_MODEL_PATH; alias_name=${HERO_MODEL:-hero-vp}; port=${PORT:-8000} ;;
 textgrad) : "${TEXTGRAD_MODEL_PATH:?Set TEXTGRAD_MODEL_PATH}"; model_path=$TEXTGRAD_MODEL_PATH; alias_name=${TEXTGRAD_MODEL:-qwen36-evaluator}; port=${PORT:-8005} ;;
 *) echo 'Usage: GPU_IDS=0 bash scripts/start_models.sh hero|textgrad' >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
exec python -m vllm.entrypoints.openai.api_server --model "$model_path" --served-model-name "$alias_name" --host "${BIND_HOST:-127.0.0.1}" --port "$port" --tensor-parallel-size "${TP:-1}" --max-model-len "${MAX_MODEL_LEN:-8192}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" "$@"
