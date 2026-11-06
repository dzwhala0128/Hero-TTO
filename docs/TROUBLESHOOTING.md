# 故障与限制

- `fine146`缺失：需要获得仓库外对应冻结文件的具体只读授权并核实SHA、40 ID、146 AF及历史QA协议。不能重跑分解器生成近似AF后冒充历史数据。
- `Full500 ... no frozen AF/Evaluator protocol`：QA/VP齐全不代表AF/Evaluator协议齐全。可运行500条原HerO Baseline；不支持把40条AF复用到500条。
- 端点别名不存在：先检查真实`/v1/models`，修改models.yaml或环境变量；不要直接把模型权重路径当服务alias。
- `/tokenize`不存在或参数不受支持：该服务不满足本包原聊天模板与严格token预算协议；使用支持chat tokenize的vLLM服务。没有自动跳过预算或静默截断路径。
- token超预算、模型异常、格式不合法：TextGrad记录baseline_fallback；不计成功优化。检查failure.json、每阶段request/token_audit/raw_response。没有语义重试或按Gold调规则。
- pending.json：上次请求可能已经到达模型。若response已缓存则校验请求SHA后回放；没有缓存响应时禁止自动重发。保留旧目录，换run-id重新运行并独立报告该次尝试。
- `.env`服务模型切换后，resume配置指纹不匹配：使用新run-id。不要更改已冻结输出。
- Shell环境无python：激活setup_env.sh创建的venv。核心mock不需要torch/vLLM，但需要requirements.txt中四个库。
- 本地完整检索缺大型资产：见ASSETS.md；知识库、NLTK、SFR、HyDE/QG模型必须自行配置。离线验收未覆盖GPU、CUDA、驱动、显存、Pyserini或Java兼容性。
- 已完成样本数：`bash run.sh status --run-id NAME`。运行ID有互斥文件锁；不得后台重复启动同一ID。full-pipeline按完整阶段标记续跑，未完成阶段文件存在时保留该目录并使用新ID。
- 参数预检查可用：`bash run.sh textgrad --split targeted40 --af merged67 --preflight`；真实HTTP/GPU环境检查需另执行`bash scripts/check_env.sh --online`，本次没有执行模型请求。
- 原始full-pipeline源脚本为GPU依赖模块，无法在无GPU库的核心环境导入；语法检查和无GPU依赖的核心闭包导入分别验收。安装requirements-retrieval.txt后才能验证这些可选模块的真实导入，当前为待验证项。
