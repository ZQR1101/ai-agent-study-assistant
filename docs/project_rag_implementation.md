# 当前项目的 RAG 实现

- source: `backend/rag_store.py`, `backend/rag_service.py`, `backend/reranker.py`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

## 索引构建

`rag_store.load_documents()` 扫描 `docs/` 顶层的 `.txt`、`.md` 和 `.pdf`。PDF 先尝试普通文本解析；低文本页、扫描件或损坏文件可进入 OCR fallback。文本按 500 字符切块，重叠 100 字符，并经过最小长度和有效字符比例过滤。

向量索引使用 SentenceTransformer 生成 float32 embedding，L2 归一化后写入 FAISS `IndexFlatIP`。因此当前向量分数是归一化向量的内积，可按余弦相似度理解。索引和 chunk 元数据分别保存到 `rag_index/index.faiss` 与 `rag_index/chunks.json`。

## 三种召回模式

- Vector：对 expanded query 编码，在 FAISS 中检索，并使用 `SIMILARITY_THRESHOLD=0.55` 过滤低分候选。
- BM25：对中英文混合文本分词，在内存 BM25 索引上计算关键词相关性。
- Hybrid：分别取 Vector 与 BM25 候选，再用 Reciprocal Rank Fusion 合并；当前权重为 Vector 1.0、BM25 1.15。

Hybrid 的融合分数不是余弦相似度，不能直接套用 Vector 的 0.55 阈值。BM25 仍只要有结果就视为通过；Hybrid 额外要求融合后的 top-1 自身有 vector 命中（即向量侧过 0.55），或满足更强的 BM25 证据（`bm25_score >= 25`，或命中全部查询实体词），否则整次检索按未过阈值处理，避免 BM25-only 的弱关联 chunk 仅靠 RRF 名次排到第一。

## Reranker 与 fallback

Reranker 只对候选 chunks 精排，不扫描全库。它默认由请求字段控制，同时还要求服务端 `ENABLE_RERANKER=true` 且配置模型。模型使用 CrossEncoder 懒加载；加载失败时返回原召回顺序并记录 `reranker_error`。

开启 query rewrite 时，两路检索先各自召回，融合候选池后再统一精排一次；精排不可用而回退融合顺序时，Hybrid 会对最终 top-1 重新应用上述质量 gate（精排成功则信任 CrossEncoder 排序，不再重复把关）。

## 离线参数实验

`scripts/rag_param_sweep.py` 用于替代经验调参：对每个 `chunk_size/overlap` 组合在内存中重建索引与 embedding（不写 `rag_index/`，不影响线上指针），用生产检索路径跑 `eval_cases`，并对 RRF 权重（`HYBRID_VECTOR_WEIGHT/HYBRID_BM25_WEIGHT`）和 `SIMILARITY_THRESHOLD` 做网格扫描。产出 `outputs/rag_param_sweep.json` 与 `rag_param_sweep.md`，报表含复现信息（git commit、cases 文件 sha256、embedding 模型、语料文档数），并按 top-3 命中率 → MRR → 低污染排序给出推荐配置。示例：

```bash
python scripts/rag_param_sweep.py \
  --chunk-sizes 300,500,800 --overlaps 0,100 \
  --bm25-weights 0.6,1.0,1.15,1.5 --similarity-thresholds 0.45,0.5,0.55,0.6
```

chunk 参数组合的成本是重建索引 + 全量编码，融合参数组合只重跑检索，二者可按需缩小网格。

## 三轮离线实验结论（balanced_v1，2026-09）

- **Query rewrite 只对改写层有效，且不应默认开启**（`--query-rewrite-mode always`，146 条）：semantic_rewrite 层 Recall@1 0.571→0.686、MRR 0.648→0.714，fact/multi_hop 层完全不动；代价是负样本拒答受损（cross_domain 拒答 1.0→0.8、污染 0→0.2；out_of_knowledge 拒答 1.0→0.9）、端到端延迟 110ms→3-7s（LLM 改写）。离线评测无对话历史，conditional 模式不会触发（等价 off），生产中正确的开启方式是 conditional 且仅对口语/指代追问生效。
- **引用正确率**（`--with-answer`，hybrid top_k=5）：fact_lookup 0.898、multi_hop 0.909、semantic_rewrite 0.743（被该层召回上限封顶）、整体 0.853。负样本侧强弃答提示词保持 out_of_knowledge/cross_domain 拒答 100%。
- **chunk 参数扫描确认当前生产配置已是最优**（`--chunk-sizes 300,500,800 --overlaps 0,100 --bm25-weights 0.6,1.15 --similarity-thresholds 0.5,0.55`，146 条）：hybrid Recall@3 在 cs500 最优（ov0/ov100 均 0.897），cs300 相当但索引多 30-78%，cs800 全面回落（vector Recall@3 0.578→0.517）；融合权重/阈值网格全部 favor 现行 `w=1.0/1.15、t=0.55`。后续调参应把精力放在语义改写层召回与负样本拒答上，而不是 chunk/融合常量。

## 多轮追问评测（balanced 之外的独立集）

`eval_cases/rag_multiturn_v1_cases.json`：33 条带真实对话历史的多轮用例——27 条追问（"那它怎么部署？/这些方式里选哪个？/继续说说压缩时该保留什么？"等，history 含上一轮用户问题与基于语料的助手回答，追问目标跨文档或深挖同文档）+ 6 条负样本追问（指代真实话题但细节不存在，如"那 MCP 认证服务的收费标准是什么？"）。评测框架已支持 `history` 字段：`load_cases` 校验并解析、`run_retrieval` 以 `history_context` 传入 search_fn（普通检索忽略之，query rewrite 包装器用它做指代消解；改写缓存键为 `(question, history)`）。`_QUERY_REWRITE_FOLLOW_UP_PATTERN` 补充了"继续/接着/然后呢/还有呢/展开/细说"触发词。

三组对照（hybrid top_k=5，33 条）：

| 层 | 指标 | baseline（不解析指代） | conditional | always |
|---|---|---:|---:|---:|
| follow_up（27） | Recall@1 / @3 / MRR | 0.444 / 0.556 / 0.495 | 0.704 / 0.815 / 0.760 | **0.852 / 0.963 / 1.000** |
| follow_up_distractor（6） | 拒答率 / 污染率 | 0.333 / 0.667 | 0.167 / 0.833 | 0.000 / 1.000 |
| 改写触发 | attempted / used | 0 / 0 | 15 / 33 | 33 / 33 |
| 平均延迟 | ms | ~363 | ~2243 | ~4956 |

结论：追问场景下改写收益巨大（不解析指代时 R@3 只有 0.556，conditional 修复到 0.815，always 到 0.963），生产默认 conditional 正确；但改写对负样本追问是反向作用——把上一轮主题注入查询后干扰题污染率升高（0.667→1.0），干扰追问的拦截仍靠答案层强弃答。另外 conditional 的触发模式只覆盖了 15/33 的追问形态（如"那记忆的时间维度呢"缺"它/这个"等触发词），触发词表还有调优空间。

## 两阶段自适应改写（当前 conditional 语义）

触发词表方案已被替换：conditional 不再做模式匹配预判，而是**先原 query 检索、后自适应决定**的两阶段流程（`rag_service.search_with_conditional_rewrite`，生产 `get_rag_context` 与离线评测包装器共用）：

1. 第一阶段恒用原 query 检索；带历史且 reranker 生效时精排延迟到改写决策之后，最终候选池只精排一次。
2. 改写仅在三个条件同时成立时触发：检索信号不足（`passed_threshold=False` 或无 chunks）、问题非自包含（`_question_is_self_contained`：含精确值（版本/数值/路径）或实体/数值锚点即视为可独立检索，历史无法改变其目标）、历史非空。
3. 自包含分类器同时是污染闸门：含实体锚点的干扰追问（"那 MCP 认证服务的收费标准"）被拒于改写之外，维持原检索的拒答判定。

33 条固定多轮集上的双门实测（hybrid top_k=5）：

| 层 | baseline | 旧 conditional（触发词表） | always | **adaptive conditional（当前）** |
|---|---|---|---|---|
| follow_up R@1 / R@3 / MRR | 0.444 / 0.556 / 0.495 | 0.704 / 0.815 / 0.760 | 0.852 / 0.963 / 0.910 | **0.778 / 0.889 / 0.828** |
| follow_up_distractor 拒答 / 污染（检索层） | 0.333 / 0.667 | 0.167 / 0.833 | 0.000 / 1.000 | 0.000 / 1.000 |
| 改写触发 | 0/33 | 15/33 | 33/33 | 11/33 |

Recall 门通过（0.889，显著优于旧 conditional 的 0.815，改写仅触发 11/33 次）；检索层污染为**已知、可解释的观测项**（决策记录，2026-09）：2 条无锚点指代负样本（免费额度/认证考试）按设计触发了改写并被主题文档命中——"解析指代"与"不污染无锚点干扰追问"在检索层不可兼得（两者在问题层面不可分），这两条保留在报表中作为观测项，不作清零处理；用户可见的端到端拒答由答案层强弃答兜底：6 条干扰追问实测全部拒答（6/6），用户看到的仍是诚实的"知识库中没有找到"而非编造回答。balanced_v1 回归确认无历史场景行为与基线完全一致（top1/top3/MRR 逐位相同）。仅当产品明确要求"返回来源列表也不得出现主题相关但不支持细节的文档"时，才实施更严格的二次 gate（如改写后 top-1 实体覆盖率复检），并接受对应的无锚点正样本召回损失。

## 分层评测用例集

`eval_cases/rag_balanced_v1_cases.json` 是按比例构建的 146 条用例：59 条精确事实题（`fact_lookup`）、35 条语义改写题（`semantic_rewrite`）、22 条多跳/多源题（`multi_hop`）、30 条负样本（`out_of_knowledge`/`distractor_keywords`/`cross_domain` 各 10 条），全部基于 `docs/` 语料真实内容编写，`expected_keywords` 逐字来自原文。评测脚本按 `case_type` 分层汇总指标（`build_case_type_summary`）：正样本看 Recall@1/3/K 与 MRR，负样本看拒答率（fallback）与 source pollution，含回答时输出引用正确率（`answer_citation_hit_rate`），报表新增 "Case Type Breakdown" 表。基线（top_k=5，版本 18 索引）：fact_lookup hybrid Recall@3=0.881，semantic_rewrite hybrid Recall@3=0.743（明显弱于 fact 层，是 query rewrite 的主战场），multi_hop hybrid Recall@3=0.909；负样本中 `distractor_keywords` 层 bm25/hybrid 污染率 100%（真实术语+不存在答案会通过 BM25 强证据 gate）。

## 拒答分层实测结论（balanced_v1）

针对干扰词负样本做过两轮验证，结论如下：

- **精排分数无法在检索层拒绝干扰题**。接入 `RERANKER_MIN_SCORE`（CrossEncoder 分数下限，低于阈值的候选被丢弃，全部低于阈值即整题拒答；`reranker_filtered_count` 记录过滤量）后实测：10 条干扰题的最高分 0.00–0.37，而正样本（尤其语义改写层）有约 25 条也落在 0.00–0.38，BM25 全词覆盖率同样大面积重叠——"主题相关但答案缺失"与"答案存在但问法口语"在检索侧不可分。因此阈值保持默认 0.0，只用作粗噪声过滤与正样本重排（hybrid_reranker 实测 top3 0.845→0.862、MRR 0.779→0.837，开销为 CrossEncoder 延迟）。更高阈值会以大量误杀正样本为代价，不建议。
- **干扰题的拒答在答案层解决**。`context_prompt` 原为软约束（"不足以回答，再说明不足之处"，模型会边声明边用背景知识作答），已改为强弃答约束：知识库没有直接包含答案（缺具体数字/配置/版本号/条款）时，只允许回答未找到并说明缺口，禁止背景知识补全。实测（30 负样本 + 12 正样本抽样，hybrid top_k=5）：distractor_keywords 拒答 6/10 → 10/10，out_of_knowledge 与 cross_domain 维持 10/10，正样本误弃答无新增（2 例误弃均为检索未命中正确文档所致，属改写层召回问题而非弃答策略问题）。

`rag_service.get_rag_context()` 把命中结果转换为 context 和 sources。组装 context 在统一 `CONTEXT_BUDGET_CHARS=6000` 字符预算内进行：锚点按最终排名优先准入（完全重复文本去重、每来源上限 `CONTEXT_MAX_CHUNKS_PER_SOURCE=3`、单块 `CONTEXT_MAX_CHUNK_CHARS=800` 截断、超预算时从尾部截断并以 `CONTEXT_MIN_TRUNCATED_CHARS` 兜底），邻居窗口用剩余预算补齐（前后各 1 个同文档 chunk，标记 `检索方式：neighbor`，不占 sources、无得分行）。确定性压缩包括：相邻同文档块裁掉 chunking overlap 造成的重复接缝（`CONTEXT_SEAM_*`），以及省略冗余元数据行（文档==来源文件、标题==章节/文档时不再重复输出）。结果中的 `context_chars / context_anchor_count / neighbor_chunk_count / context_truncated_chunk_count / context_dropped_chunk_count` 用于观测实际注入量。Vector 未过阈值时返回空来源；回答层使用固定 fallback 文案，不把弱相关 chunk 注入模型。

