#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

BATCH_ID="${BATCH_ID:-night_$(date +%Y%m%d_%H%M%S)}"
FAILED=0

echo "=========================================="
echo "HerO + TextGrad 夜间批量实验"
echo "批次: $BATCH_ID"
echo "模型: Qwen3.5-9B / 8005"
echo "Few-shot: fixed2"
echo "开始: $(date)"
echo "=========================================="

# 先验证三套输入，不调用模型
for spec in "10 human63" "40 human63" "169 original"; do
    read -r COUNT AF <<< "$spec"

    echo "===== 预检 $COUNT / $AF ====="

    if ! bash run.sh qwen \
        --count "$COUNT" \
        --af "$AF" \
        --preflight; then

        echo "输入预检失败，停止整个批次"
        exit 1
    fi
done

# 顺序运行三个真实实验
for spec in "10 human63" "40 human63" "169 original"; do
    read -r COUNT AF <<< "$spec"

    RUN_ID="${BATCH_ID}_${COUNT}_${AF}"

    echo
    echo "=========================================="
    echo "开始: $RUN_ID"
    echo "时间: $(date)"
    echo "=========================================="

    if bash run.sh qwen \
        --count "$COUNT" \
        --af "$AF" \
        --run-id "$RUN_ID" \
        --freeze; then

        # 冻结后检查样本数量与运行状态
        if python - "$RUN_ID" "$COUNT" <<'PY'
import json
import sys
from pathlib import Path
from collections import Counter

run_id = sys.argv[1]
expected = int(sys.argv[2])
folder = Path("outputs") / run_id

assert (folder / "freeze.json").exists(), "未生成冻结文件"

rows = json.loads(
    (folder / "predictions.json").read_text()
)

assert len(rows) == expected, "样本数量不匹配"
assert all(not r.get("mock") for r in rows), "出现Mock结果"

counts = Counter(r["status"] for r in rows)
print("样本数:", len(rows))
print("运行状态:", dict(counts))

bad = [
    r["claim_id"] for r in rows
    if r["status"] not in ("optimized", "no_change")
]

if bad:
    print("失败/回退Claim:", bad)
    sys.exit(1)

print("工程执行检查通过")
PY
        then
            echo "===== 计算离线指标 ====="

            if bash run.sh compare --run-id "$RUN_ID"; then
                echo "SUCCESS: $RUN_ID"
            else
                echo "METRICS_FAILED: $RUN_ID"
                FAILED=1
            fi
        else
            echo "RESULT_CHECK_FAILED: $RUN_ID"
            FAILED=1
        fi
    else
        echo "RUN_FAILED: $RUN_ID"
        FAILED=1
    fi
done

echo
echo "=========================================="
echo "全部任务已尝试执行"
echo "结束时间: $(date)"
echo "失败标记: $FAILED"
echo "=========================================="

exit "$FAILED"
