# 外部资产、实际BM25与可选Lucene索引

不包含模型权重、目标知识库、Lucene索引或NLTK资源。禁止把缺失资产当作已具备。原始Claim不含人工QA进入full-pipeline；训练集只保留QG固定检索示例所需的claim/questions，删除label与justification。

仓库实际retrieval.py用rank_bm25，对每条Claim专属知识库分词、去重与检索；没有可复用的Lucene检索器。因此**忠实运行不需要Lucene重建**。full-pipeline要求外部AVeriTeC抓取知识库，每个Dev ID一个`<ID>.json`（JSONL），每行包含`url`字符串和`url2text`句子数组。包中没有网页抓取器；需要由AVeriTeC对应知识库发布资产取得，不能用Gold问答替代。知识库和抓取时间会影响完整复现。

取得知识库后，先校验内容与记录哈希：

```bash
mkdir -p outputs/assets
python -B scripts/assets.py --store "$KNOWLEDGE_STORE" --split full500 --seal outputs/assets/knowledge_sha256.json
# full-pipeline预检还要求显式GPU分配及本地模型路径
bash run.sh baseline --split full500 --mode full-pipeline --preflight
```

可选Lucene资产重建（这是资产导出/重建路线，**不是本包默认HerO检索实现**；本包没有声称Lucene查询适配器已验证）：

```bash
# 在独立、支持Java21的环境安装pyserini，确认java -version
python -m pip install pyserini
python -B scripts/assets.py --store "$KNOWLEDGE_STORE" --split full500 --seal outputs/assets/knowledge_sha256.json --export-lucene outputs/assets/lucene_docs
python -m pyserini.index.lucene --collection JsonCollection --input outputs/assets/lucene_docs --index outputs/assets/lucene_index --generator DefaultLuceneDocumentGenerator --threads 4 --storePositions --storeDocvectors --storeRaw
```

导出保留Claim ID；跨Claim全局索引查询需要按Claim过滤，不能直接当作原始专属知识库BM25的同等协议。Pyserini/Java/索引重建在当前环境均未实际执行。

模型路径只写在`.env`：源码内提供模型家族参考，不访问原服务器模型目录。原项目路径文件名实际为`model_path (2) (1).txt`，记录的模型包括70B HerO、70B Instruct、8B Instruct、SFR-Embedding-2_R、Qwen3.6-27B。目标机器可更换路径，不需要原项目目录。Targeted40最近历史9B模型不在该路径清单，需自行提供；可改用其他模型做新实验，但不能宣称同模型历史复现。
