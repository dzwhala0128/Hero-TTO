1..服务器内项目所在位置
cd /data/users/dengzhenyun/Hero-TTO/HerO_TextGrad_Advisor_FrozenBaseline

2.激活我自己单独建立的conda环境（环境都配置好了的，可以直接运行项目）：
source "/data1/users/dengzhenyun/wu/use_my_conda.sh"
conda activate llama70b-bw

我自己常用的模型开启命令（在GPU4 和 GPU5，各占用20%显存运行）：
conda activate llama70b-bw

export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4,5 \
vllm serve /data/users/dengzhenyun/wu/models/Qwen3.5-9B/ \
  --served-model-name qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8005 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.20 \
  --trust-remote-code

-------------------项目运行-----------------------------------------
-------------------如果想使用Factscore拆分后再合并精修的67 AF，
只需将--af human63改成--af merged67。-----------------------

3.运行一个 指定claim-id 样本（40个claim经人工拆分成的63个 AF中挑选 + 人工筛选QA）
bash run.sh qwen \
  --count 1 \
  --claim-id 59 \
  --af human63 \
  --run-id demo_1 \
  --freeze

4.运行10个样本（40个claim经人工拆分成的63个 AF中挑选 + 人工筛选QA）
bash run.sh qwen \
  --count 10 \
  --af human63 \
  --run-id demo_10 \
  --freeze

5.运行40个样本（人工拆分成的63个 AF + 人工筛选QA）
bash run.sh qwen \
  --count 40 \
  --af human63 \
  --run-id demo_40 \
  --freeze

6.运行169个样本（不拆分AF，使用原始10 QA，因为只对上面的40个claim进行了AF拆分测试）
bash run.sh qwen \
  --count 169 \
  --af original \
  --run-id demo_169 \
  --freeze


7.指标计算
# 1个样本
bash run.sh compare --run-id demo_1

# 10个样本
bash run.sh compare --run-id demo_10

# 40个样本
bash run.sh compare --run-id demo_40

# 169个样本
bash run.sh compare --run-id demo_169


结果保存在：
outputs/<run-id>/
├── claims/                          # 每个样本的各阶段执行记录
├── predictions.json                 # 最终预测结果
├── freeze.json                      # 冻结记录
└── baseline_vs_textgrad_metrics.json # 优化前后指标对比

注意： --run-id每次运行必须使用新的名称，避免与已有实验目录冲突。
以上命令默认使用fewshot/fixed2.jsonl中的2-shot，并非6-shot



