> **提示**：以下是最初Codex交付包在原服务器环境中的历史验收记录，并非本次精简包在当前环境的全部重新验收结果。精简包的文件列表、删除依据及本地检查见根目录 `CLEANUP_REPORT.md`。

# 历史初版验收记录（清理前）

验收对象是本地clean-main工作副本整理的独立交付包，原Git HEAD为`9c819c43cad0b9aee0042a88dd02ad914c0e175c`。原工作区已有两处tracked修改及多个历史untracked实验，本次未修改它们、未提交或推送。

已实际通过：

- 所有交付Python源码编译检查；全部Shell脚本`bash -n`。
- 核心pipeline、Native TextGrad、hero_extension全闭包导入；原始Variable、autograd/llm_ops、原生Backward Prompt、config、Optimizer/TGD与Prompt逐字节匹配来源。
- 500/40/169准确Claim ID覆盖；Full500 5000 QA/500 VP、Targeted40原400 QA、Full169 1690 QA/169 VP。
- 人工63/合并67 AF逐Claim映射、文本SHA，154 QA内容一致；两臂冻结初始VP相同。所有40/40/169初始Solution可解析。
- 14项离线单元测试：真实原生Evaluator→autograd backward→TGD图、faithful分支、NO_CHANGE必须通过Backward、原始VP标签优先级、显式最终解析器差异、纯结构校验、格式失败回退不算优化、Gold字段拒绝与运行时路径拦截、冻结后评价及篡改拒绝、全部声明支持的参数/路径预检查。
- 已实际运行`bash run.sh textgrad --split targeted40 --af human63 --claim-id 9 --mock --run-id acceptance_smoke --freeze`及status，返回1/1 completed、frozen=True。测试产物没有作为历史科研预测交付。
- 静态检查运行源码/配置没有原项目绝对路径依赖；历史来源路径仅在溯源元数据中保留。

可重复验收：`python -B scripts/verify_offline.py`。结果只代表CPU离线闭包。

**未通过或未执行的项目不标成功：**

- fine146冻结文件/证据协议/原始映射缺失，无法核实40/146或其SHA；63/67原始外部文件和67合并器未重新读取。AGENTS.md禁止跨仓库读取，已发出具体路径只读授权请求；本次没有收到授权，未访问这些目录。
- 当前Python 3.13.13，未安装torch、vLLM、Transformers、Tokenizers、Pyserini、numpy等完整GPU/检索依赖。可选GPU阶段导入、CUDA、SFR、Java/Lucene以及安装依赖版本兼容性均待目标环境验证。
- 对localhost:8000和8005执行过只读`/v1/models`探测，但环境网络权限返回Operation not permitted；实际服务别名未验证，详细记录在model_service_probe.json。运行时会强制验证别名。
- 没有启动模型，没有运行任何真实GPU推理；40/169/500 GPU复现均为**待模型环境验证**。
- Full169纯结构Validator与历史含URL边界Validator存在已声明差异；不宣称历史接纳轨迹逐位一致。
- 原HerO根目录许可证、数据许可正文未找到，详见LICENSES.md。

ZIP独立解压的环境检查与14项测试结果见根目录交付MANIFEST；压缩包不含测试日志、缓存、凭据或模型权重。

初版历史验收记录为59个Python文件、7个Shell脚本；当时14项测试均通过（精简后文件数可能不同）。ZIP验收在当前仓库内临时创建的独立解压目录运行，验证结束移除临时副本，不访问其他项目目录。
