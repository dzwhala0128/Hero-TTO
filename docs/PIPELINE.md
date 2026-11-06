# 完整流程与源码对应

1. `pipeline/full_pipeline.py`只读取`data/<split>/claims.json`。`hero/stages/hyde_fc_generation.py`对Claim生成8篇假设性核查文本，按原生成logprob排序；本地vLLM、原始Prompt与生成参数保留。GPU资源参数通过环境配置。
2. `hero/stages/retrieval.py`读取外部每Claim知识库，去重，使用Claim和过滤后的HyDE文本进行rank-bm25检索。保留本仓库真实`hyde_doc_utils.py`过滤规则；这反映当前clean-main副本，未声称是未经修改的上游HerO首版。
3. `hero/stages/reranking.py`用SFR-Embedding-2_R查询与文档向量，HyDE查询均值、相似度排序，保留去除近重复Claim证据的策略。副本修复跳过句子时索引不前进的无限循环，以及硬编码top_10000访问；证据不足明确报错。
4. `hero/stages/question_generation.py`用训练QA示例BM25检索10个fewshot，为重排句子生成问题，答案就是证据句。训练label/justification未携带到QG参考数据。
5. `pipeline/baseline.py`从`hero/stages/veracity_prediction.py`AST提取原VP Prompt。参数为temperature=.9、top_p=.7、top_k=1、max_tokens=512及原stop序列。解析不到标签后top_k=2；原代码在整个运行后续样本继续使用2，本包保留状态。原无限重试增设默认10次运维上限，可配置`--max-attempts`。HTTP用服务`/tokenize`复现不追加assistant generation header的聊天模板，再向`/v1/completions`提交token IDs。权重、服务版本及聊天模板仍须真实环境验证。
6. `pipeline/build_inputs.py`白名单加载冻结数据，不访问Gold。Targeted40初始VP来自9B历史显式最终标签预测，完整原响应仍保留；20、141历史响应不完整，未重写。Full169使用原HerO初始Solution及冻结V4-A fixed6系统/用户消息、响应；Full500有完整5000 QA与500 VP。
7. `pipeline/evaluator.py`将冻结Solution位置替换为唯一可训练Variable，其他Claim/AF/QA/Prompt保持常量。E18使用`textgrad_native/textgrad/autograd/llm_ops.py::FormattedLLMCall`；真实反馈独立保存。Full169通过历史faithful扩展回放完整原Evaluator请求，原始响应及规范化响应分别存于runtime。
8. `pipeline/native_backward.py`调用`loss.backward(engine=...)`。E18断言实际绑定`LLMCall.backward`。Full169使用`hero_faithful_backward_v2_compact.py`的Backward扩展，保留原生autograd与Variable.gradients连接；不是手写Prompt冒充原生TextGrad。Backward模型请求、响应、文本梯度和上下文分别保存。
9. `pipeline/optimizer.py`直接调用Native `TextualGradientDescent.step()`。历史HERO_CONSTRAINTS、原Optimizer Prompt及更新标签保留。Optimizer修改justification和verdict。默认一次迭代，E18允许`--max-iterations`设置；Full169冻结Evaluator协议只支持一次，防止多轮使用旧反馈。Full169真实NO_CHANGE必须经过Backward模型生成，再保留初始预测，不算成功优化。
10. `pipeline/validator.py`只解析JSON或E18严格两字段文本、检查字段类型、非空justification和四分类枚举，不访问证据、不判断语义、不做URL或标签保持Gate。失败恢复原Baseline并记`baseline_fallback`。历史Full169 validator源文件也保留作为对照，但**不在当前接纳链中执行**；原URL/监督字段边界规则与本次纯结构要求有差异，不能宣称逐位复现旧接纳结果。
11. `pipeline/runner.py`保存逐Claim输出，校验运行指纹支持resume，预测freeze附SHA。`pipeline/evaluate.py`独立进程先校验freeze、结果、运行清单SHA，然后才能打开`data/evaluation_only/dev.json`。原`averitec_evaluate.py`用于可选官方证据指标；默认准确率/F1用本包无额外依赖实现。

## 原始解析器与后来显式最终标签解析器

原始VP对子串按顺序选择：Not Enough Evidence → Conflicting Evidence/Cherrypicking（含简写）→ Supported（含小写）→ Refuted（含小写）。因此正文中出现其他标签也可能影响结果，绝非只解析末尾标签。包内没有擅自改善它。

Targeted40冻结9B历史预测来自后来明确提取`Final Answer:`等终止标签的协议；它与原HerO子串解析器不是同一种Baseline。AF入口直接保留历史pred_label与完整原输出，不用原解析器重新标注它。Optimizer最终标签只来自两字段Solution的verdict；不会把Evaluator反馈或文本梯度当作分类标签。原始VP重新运行结果与冻结9B结果须分别报告。

## 追踪与错误

每个`outputs/<run>/claims/<id>/iteration_1/`包含`evaluator/`、`backward/`、`optimizer/`的request/response与原始响应、token审计，以及独立evaluator_feedback、textual_gradient、optimizer_solution、structural_validator。最终result记录状态与标签，失败附failure.json。模型请求写pending后才发送；中断不自动重发。所有请求日志不保存HTTP Authorization密钥。

Native源码除HTTP-only engine registry外保持计算链实现原样；裁掉tasks、无关vendor服务、实验训练代码。默认Native磁盘日志关闭，避免不受控外部缓存；本包提供逐阶段审计替代。详见source_provenance中原始字节SHA。
