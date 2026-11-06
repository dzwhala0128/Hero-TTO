# 实际支持矩阵

| 数据入口 | Claim | QA | AF | 冻结初始VP | TextGrad协议 | 状态 |
|---|---:|---:|---:|---|---|---|
| full500 | 500，Dev索引0–499 | 5000 | 无 | 500条历史hero70b baseprompt same-evidence预测 | 无冻结AF/Evaluator协议 | VP-only输入齐全；GPU待验证 |
| targeted40 original_qa | 固定40 | 400 | 无 | 原始400 QA对应的9B历史预测在AF入口中保留 | Baseline命令重新运行原始HerO VP | 输入齐全；GPU待验证 |
| targeted40 human63 | 同一固定40 | 154 | 63，历史人工审核 | 40条9B显式最终标签预测；20、141历史推理不完整 | E18 + 原生FormattedLLMCall.backward + TGD | frozen-input；离线图通过，GPU待验证 |
| targeted40 merged67 | 同一固定40 | 154，与human63逐条相同 | 67，历史合并结果 | 同human63 | 同上 | frozen-input；缺原始合并器，GPU待验证 |
| targeted40 fine146 | 未携带冻结文件 | 未核实 | 146为用户给定待核实数量 | 未绑定 | 不得强行采用154 QA/E18 | 不可运行，明确报错 |
| full169 | Rare71+SR98，固定169 | 1690 | 无 | 169条历史HerO VP | 冻结V4-A fixed6 Evaluator回放 + faithful v2 compact backward + TGD | 离线图通过；GPU待验证 |

Full169排除原173集合的103、109、220、223，具体顺序见`data/full169/claim_ids.json`。Targeted40不是Dev前40条。Full169/500没有63/67/146 AF映射，指定AF直接报错。Full500有完整QA/VP，但缺少历史TextGrad实验协议，不能声明AF500复现。

模型配置：原始VP默认服务别名`hero-vp`（70B HerO）；Targeted40历史E18默认`qwen3.5-9B`；Full169默认`qwen36-evaluator`（Qwen3.6-27B）。运行时可用环境变量显式替换服务，`run.json`记录实际别名；替换后属于不同模型设置，不能报告为历史同模型复现。包内`configs/human63.yaml`、`merged67.yaml`保存真实历史配置，统一HTTP端点配置仅改变传输。

Full169保留完整历史Evaluator系统/用户消息（含fixed6示例），使用原冻结响应回放构建图，因此Evaluator不是新一次模型推理。Backward仍通过`loss.backward()`真正调用模型。原历史validator包含额外URL边界检查；本包根据任务要求使用纯结构Validator，该差异在PIPELINE中声明。因此本包Full169是保留历史Prompt/图/初值的结构校验整合协议，不承诺逐位复现历史全部接纳结果。其他four-arm协议未作为同名入口混入。
