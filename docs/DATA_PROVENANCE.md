# 数据、源码与SHA追溯

来源为当前本地`clean-main`工作副本。远程参考是 https://github.com/dzwhala0128/hero-textgrad-research/tree/clean-main ，本次远程页面获取失败，未据此替换本地文件或宣称已逐文件比对远程。

`data/manifests/source_provenance.json`记录每个实际读取的本地原始来源及文件SHA256；有destination时可对照交付副本。`data/manifests/package_sha256.json`为最终包逐文件SHA封存（不自包含、不含outputs）。适配后文件与原始SHA不同属正常，原始来源SHA与交付SHA分别记录。

| 内容 | 当前仓库来源 |
|---|---|
| Full500 Claim/Gold | data_store/averitec/dev.json，ID为原始数组索引 |
| Full500 5000 QA与500 VP | data_store/exp018_textgrad_vp_prompt/heldout_eval/heldout_0_500_baseprompt_hero70b_same_evidence.pred.json |
| Targeted40 原始400 QA | targeted40_hero9b_final_recovery/frozen_inputs_no_gold.json |
| 63 AF/154 QA/冻结9B VP | targeted40_9b_textgrad_af63_v1/initial_predictions.json、af_mapping.json、prepared_evaluator_inputs.json、config.json |
| 67 AF/154 QA/冻结9B VP | targeted40_9b_textgrad_af67_v1/对应文件 |
| Full169固定清单 | data_store/fewshot_v4/textgrad_full169_four_arms_v1/target169_manifest.json |
| Full169冻结VP/QA | 清单splits指定的Rare73与SR100原始冻结runtime，按固定169 ID选择 |
| Full169 Evaluator历史请求/反馈 | textgrad_source_native_hero/hero_outputs/evaluator_v4a_full169/evaluation/evaluation.jsonl |
| 原生TextGrad与Prompt | textgrad_source_native_hero/textgrad/ 与 hero_extension/的必要闭包 |
| AF生成方法 | atomic_decomposition_factscore_v2/decompose.py、decompose_original_prompt.py、demons.json |

63 AF为历史人工审核，不能标为自动生成。67 AF只保留真实冻结合并结果；原始合并规则代码不在当前仓库，因此不可重新执行合并器。提供的FActScore生成脚本不是缺失合并器的替代品，也不能从其存在推断146 AF已复现。

仓库已有input_manifest记载的外部来源SHA（只作为历史元数据，本次没有访问外部路径）：

- 63 AF原文件：targeted40_experiment/frozen_inputs_v162/Targeted40_40Claims_ApprovedAF_SelectedQA_ReadyToImport.json；SHA256 `1f97dc7b50a789b4796f49dc312aad49a4389fae6c6d4e56be95c2b9348f7197`。
- 67 AF原文件：targeted40_factscore_refined_v1/prepared_runtime_input.json；SHA256 `5eca90a5e1892458bc79b2faa324799c59efa108a735227c327835d4d375bcc4`。
- 146 AF原目录targeted40_factscore_finegrained_v2不在当前仓库，缺精确冻结文件、逐Claim映射、证据协议、生成请求与初始VP绑定，故无可信SHA/实际计数。仅保留生成方法，不填造数据。

原始63/67数据字节值、AF顺序及逐Claim对应关系从仓库已有副本提取，原始文件未改写；交付mapping仅把QA编号键`label: QA1`更名为`qa_tag: QA1`，避免与Gold的label字段混淆。每条AF文本SHA和整包SHA均验收。

154 QA两臂逐Claim完全相同。历史mapping已表明其中80条问题+答案匹配400 QA，54条仅答案匹配而问题不同，20条与原400 QA不同；不能把154 QA说成简单抽取400 QA的子集。没有臆造历史per-AF边，每条Claim的AF共享该Claim的完整选择QA。

Gold保留在evaluation_only；其余runtime用字段白名单提取，原始gold justification、claim_types、fact_checking_article等不传入模型。Full169 fixed6示例是既有冻结Evaluator请求的一部分；未通过目标Gold选例或改Prompt。原始历史筛选可能受实验设计影响，本包只保留既定集合，不重新筛样本。

额外核查仓库现存AF输出：full169_v1_6/atomic_v2.jsonl抽取同40 ID合计109 AF；factscore_9b_targeted40_20261104_192126合计165 AF，均非146冻结版本，未混入fine146入口。

Full500未找到配套的完整HyDE/10000候选检索/SFR重排中间快照，本包保留完整最终QA及VP；缺失中间产物可通过README full-pipeline命令重建，依赖外部知识库与权重。
