#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The patch never includes .env or outputs; keep previous work exactly as-is.
# Remove only obsolete files created by the PREVIOUS incorrect 'new VP' deliverable.
rm -f prompts/qwen_vp_system.txt
rm -f scripts/finish_unified_update.sh
rm -f 导师运行与复现手册.md
rm -rf experiments/e18_compact_sf_few2_v1
rmdir experiments 2>/dev/null || true
# Preserve any QG reference backup unless it is byte-for-byte equal to restored original.
if [[ -f fewshot/hero_qg_reference.json && -f data/qg_reference.json ]]; then
  if cmp -s fewshot/hero_qg_reference.json data/qg_reference.json; then
    rm -f fewshot/hero_qg_reference.json
  fi
fi
bash scripts/check_env.sh
printf '\n已完成冻结VP版更新：原始HerO源码保留；不执行任何VP模型请求。\n'
printf '请先运行：bash run.sh qwen --count 500 --preflight\n'
