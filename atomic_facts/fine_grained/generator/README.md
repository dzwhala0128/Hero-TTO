# Lightweight FActScore decomposition V2

独立轻量接口：Original Claim → 一次 FActScore-style 分解 → AF 字符串列表 → Stage A × 10 QA。
不是 `atomic_decomposition_averitec_aligned_v2` 的 Composition Schema，不修改也不代表已验收冻结的 V2 R2 语义契约。Phase 1-A/B1/B2、历史 smoke 和 V1.6 全部保留。

## 来源与复用

- 早期入口 `scripts/run_atomic_claim_decomposition.py`：`main → decompose_claim → sent_tokenize → decompose_sentence → build_factscore_prompt`。NLTK 逐句分解，7 固定 + 1 BM25，bullet 解析。逐句上下文不能消解跨句指代，且含仓库外 NLTK 数据路径；新入口不导入它。
- 选定并实际复制 `scripts/run_atomic_claim_decomposition_v2.py` 为 `decompose.py`：`main → decompose_claim → build_prompt → get_best_demo`。沿用已存在的整条 Claim 单次调用，不新增分句或改写阶段。
- 实际复制 `data_store/atomic_claim_decomposition/factscore_demos/demons.json` 为本目录 `demons.json`，21 条示例逐字保留，7 个固定示例的顺序不变。保留 `rank_bm25.BM25Okapi`、空格分词及 top-1 检索，包括可能重复选择固定示例的旧行为。
- 仓库脚本自述参考 FActScore 的 biography-mode prompt/7+1 策略及 bullet 格式；本仓库未发现上游完整 `atomic_facts.py` 或来源校验信息，因此不声称已独立核实上游逐行来源。
- 项目新增逻辑：Qwen chat API、整 Claim 上下文及独立化 Prompt。`postprocess_atomic_claims.py` 另有模型归一化/消指；`run_atomic_claim_decomposition_v2_refusal_fallback.py` 有二次模型调用及 attribution wrapper。二者都不是可靠的确定性消指，不纳入此入口。

## 最小变化与局限

目标 Prompt 强调最小且有独立核查价值的命题，可适当拆开事件、地点、时间、身份、属性；允许有核查价值的弱命题与更具限定的命题重叠，但不机械枚举所有弱命题或生成完全重复项。同一事件保留足够关联信息，所有具体信息仍须覆盖；不得因拆分改变否定、量词、条件、因果、比较、引用作用域。仅依据 Claim 消指，不编造。原 21 条示例、7 固定 + 1 BM25 策略均不变；原例用于细粒度和有价值的重叠，不模仿无法独立理解的指代或改变含义的省略。

沿用 `- fact` 文本列表。解析改为逐行识别（避免事实正文中的 `- ` 被切碎），拒绝空值或非列表内容；只删除完全相同的事实，不做语义合并。没有多轮修复、逻辑图、span 或 provenance。

记录包含 `claim_id / original_claim / raw_decomposition / atomic_facts / status`，有问题时附简短 `reason`：

- `parsed_unreviewed`：仅格式解析通过，**不代表 atomicity、事实覆盖或语义保真通过**。
- `reference_note`（可选字段，不是失败状态）：命中常见指代词时附非阻断人工检查提醒，状态仍为 `parsed_unreviewed`。词法检测不能判断消指是否成功；例如 `If employees work nights, they receive an allowance.` 中 they 可以在同一 AF 内理解，不再被标记 `needs_reference_review`。真正含糊的 `She…there` 也只保留提醒，不猜测实体。提示不代表存在语义错误，无提示也不代表自足性已验证。历史记录不重写。
- `parse_failed / incomplete / request_failed`：保留可取得的原始输出/原因，无自动重试或 fallback。

包含非阻断提醒的 `parsed_unreviewed` 也自动生成 `stage_a_input.jsonl`；失败或截断记录保留 `decomposition.jsonl` 供人工检查，不导出不完整 AF 子集。模型若编造了实体，格式检查无法检测，因此人工运行后仍需读 AF 与原 Claim 对照；本代码不承诺自动语义验收。

## Stage A 接口与示例

`to_stage_a(record)` 直接转换为 V1.6 `prepare_inputs` 的真实输入：

```json
{"claim_id":"manual_1","claim":"Mira arrived in Paris in 2020. She opened a restaurant there in 2021.","atomic_facts":["Mira arrived in Paris.","Mira arrived in Paris in 2020.","Mira opened a restaurant.","Mira opened a restaurant in Paris.","Mira opened a restaurant in Paris in 2021."]}
```

这是允许的人工细粒度示例，不是真实模型输出，也不是所有 Claim 的固定数量模板。完整运行记录还含同一原 Claim 的 `original_claim`、原始 bullet 文本 `raw_decomposition` 和 `status: parsed_unreviewed`。

现有 `scripts/run_atomic_biaxial_diagnosis_v1_6.py --atomic` 可读取该 JSONL；其 `prepare_inputs` 生成整数 `atomic_fact_id: 1, 2, ...`，无需模型生成 ID。QA 仍由现有 `--runtime` 输入提供，按相同 `claim_id` 对接；应提供 10 条 QA，适配器不读取或修改证据。保留 `claim` 供后续 claim-level 使用。本入口不会启动 Stage A/B。

## 无模型测试

```bash
python -B atomic_decomposition_factscore_v2/test_offline.py
bash -n atomic_decomposition_factscore_v2/run_manual.sh
python -B atomic_decomposition_factscore_v2/decompose.py --help
```

本次结果：8 个 unittest 方法，7 通过，1 因当前环境无 `rank_bm25` 跳过。保留真实 BM25 测试，依赖安装后会自动运行；另用 AST 对照证明 load_demos/get_best_demo/build_prompt 与旧版一致（仅导入延迟），替身检查 7+1 及检索参数。未安装依赖、未调用任何模型或 GPU。

人工语义样例共 9 条，测试固定期望输出的完整保留，不以数量判断语义，也不把手写输出当作模型效果证据：

| 样例 | 人工检查依据 |
|---|---|
| Mira lives in Paris | 一个居住关系，主体地点完整 |
| Mira sings and Omar dances | 两个独立谓词分别保留 |
| Mira…She…there | 五条粒度参考均明确主体；保留巴黎地点及两事件各自年份，允许有用重叠 |
| She…there，无上下文 | 原样保留并提示复核，不补名字地点 |
| Mira did not visit Rome in 2020 | 否定、地点、年份都保留 |
| If employees work nights… | 保留原条件及句内 they，非阻断提醒，不判为 unresolved |
| road flooded because dam broke | 两个被断言事件及单独的因果关系均保留 |
| Mira sings 重复 | 仅精确重复去重 |
| Omar said all trains were faster…2020 | 引用、全称、比较及年份同在，不提升为无引用事实 |

另检查非法/空列表、截断、正文连字符、单次 mock 调用明确关闭 thinking，以及从原 V1.6 文件提取并执行未修改的纯输入函数，用五条细粒度 AF 和 10 条合成 QA 验证接口；另验证条件 AF 和含糊指代均可经适配器保留原文，失败/截断仍阻断。没有读取 Gold、真实数据集或 Full169。

## 用户手动运行（本次未执行）

在已有包含 `openai` 和 `rank_bm25==0.2.2` 的 Python 环境中：

```bash
cd source-project://hero-v3-clean
bash atomic_decomposition_factscore_v2/run_manual.sh \
  'Mira arrived in Paris in 2020. She opened a restaurant there in 2021.'
bash atomic_decomposition_factscore_v2/run_manual.sh \
  'The road flooded because the dam broke.'
bash atomic_decomposition_factscore_v2/run_manual.sh \
  'If employees work nights, they receive an allowance.'
```

脚本显式使用 `http://127.0.0.1:8006/v1`、`qwen36-atomic`；Python 请求明确携带 `extra_body={"chat_template_kwargs":{"enable_thinking":false}}`，temperature=0、max_tokens=1024、客户端重试=0。只发一条 Claim、一次请求，不启动模型服务。

第二个可选 shell 参数为新输出目录；默认在本目录 `runs/manual_<时间>_<PID>/` 下写入。目录已存在即失败，避免覆盖历史结果。可替换第一参数运行另一条人工 Claim。不提供真实数据集批量入口。
