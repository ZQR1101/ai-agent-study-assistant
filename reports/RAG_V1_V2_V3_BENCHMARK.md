# RAG V1/V2/V3 本地 Benchmark

- generated_at：`2026-07-06T06:25:15.311060+00:00`（原始时间）；数据已在 Source Pollution Fix 后更新
- branch：`codex/frontend-workspace-ui`
- base_commit：`b28d30c`（前端学习工作区改版）
- working_tree：4 个 modified files，约 20 个 new untracked files（详情见 `reports/RAG_V1_V2_V3_METRICS.json`）
- environment：本地 Windows 进程；每种 Mode 先执行 1 次 warm-up query，每个 Case 测量 1 次 Retrieval
- generation：仅评测 Retrieval，不调用 LLM Answer 或 LLM-as-Judge
- Top-K：5
- reports_source：`outputs/rag_corpus_benchmark/*.json`（原始 V1/V2）；`outputs/rag_source_pollution_fix/v3_fix3_retrieval.json`（Source Pollution Fix）；`outputs/rag_query_rewrite_opt/*.json`（结构化 Chunk 与 Query Rewrite Ablation）
- retrieval_config：见下方 Retrieval 配置

## Retrieval 配置

| 参数 | 配置 |
|---|---|
| Vector Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2`（384 dims，SentenceTransformer） |
| Vector Similarity Threshold | 0.55（Cosine Similarity，Hard Filter） |
| BM25 k1 / b | 1.5 / 0.75 |
| BM25 Min Score（Soft Gate） | 1.0 |
| Hybrid Fusion | Reciprocal Rank Fusion（k=60） |
| Hybrid Vector/BM25 Weight | 1.0 / 1.15 |
| Hybrid Strong BM25 Threshold | 25.0 |
| Reranker Model | `BAAI/bge-reranker-base`（CrossEncoder，本地缓存：`models/bge-reranker-base/`） |
| Reranker Top-N | 20（Retrieve 20 → Rerank → 保留 Top-5） |
| Reranker Min Score | 0.0（Disabled；不对 Reranker Score 做 Hard Cutoff） |
| OCR Engine | RapidOCR（通过 `backend.ocr_adapter.py`） |
| OCR Enabled | V3 On：Yes；V3 Off：No |
| Source Pollution Fix | V3 已启用（Unified Gate：Vector 无结果且 BM25 Top Score < 25.0 时，Hybrid 拒绝返回） |
| Chunk Strategy | 默认启用结构化 Chunk：按标题、章节、段落聚合；Chunk 元数据进入 BM25、Embedding Text 与 Reranker 输入 |
| Query Rewrite | 代码保留能力，但默认关闭；全量开启在当前 V3 Benchmark 中损伤正样本召回 |

## Corpus 与 Index 对比

| Batch | Documents | Chunks | Index Build ms | Parse Failures | OCR Triggered | OCR Used | Cases（正/负） |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 51 | 341 | 17927.051 | 1 | 2 | 2 | 20（16/4） |
| V2 | 56 | 352 | 17710.692 | 1 | 2 | 2 | 35（26/9） |
| V3 | 63 | 1438 | 20933.168 | 1 | 4 | 4 | 55（40/15） |

V1 包含本地项目文档。V2 在 V1 基础上增加 5 篇精选官方文档摘要。V3 在 V2 基础上增加 4 篇原始研究论文 PDF、2 个从真实论文页面生成的 OCR Fixtures，以及 1 份 Provenance Manifest。

`index_build_ms` 是每批执行 `rebuild_rag_index` 的实测耗时。V3 复用了已完成 Parse 的 Document Objects，以避免重复执行 OCR，因此不能把 V2/V3 的 Build Time 变化表述为严格的端到端性能提升。

## Retrieval 指标对比

| Batch | Mode | Top-1 | Top-3 | MRR | Avg ms | P95 ms | Fallback Success | Source Pollution |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | `Vector` | 31.2% | 43.8% | 0.380 | 18.626 | 22.530 | 100.0% | 0.0% |
| V1 | `BM25` | 75.0% | 93.8% | 0.833 | 14.591 | 16.716 | 0.0% | 100.0% |
| V1 | `Hybrid` | 68.8% | 81.2% | 0.768 | 29.086 | 32.551 | 0.0% | 100.0% |
| V1 | `Hybrid+Reranker` | 81.2% | 87.5% | 0.861 | 2217.814 | 2501.265 | 0.0% | 100.0% |
| V2 | `Vector` | 46.2% | 61.5% | 0.542 | 17.949 | 22.627 | 100.0% | 0.0% |
| V2 | `BM25` | 80.8% | 96.2% | 0.872 | 14.358 | 18.289 | 0.0% | 100.0% |
| V2 | `Hybrid` | 69.2% | 84.6% | 0.796 | 29.738 | 36.328 | 0.0% | 100.0% |
| V2 | `Hybrid+Reranker` | 84.6% | 92.3% | 0.886 | 2246.045 | 2584.587 | 0.0% | 100.0% |
| V3 | `Vector` | 57.5% | 65.0% | 0.617 | 18.070 | 22.608 | 86.7% | 13.3% |
| V3 | `BM25` | 72.5% | 80.0% | 0.762 | 40.796 | 50.500 | 0.0% | 100.0% |
| V3 | `Hybrid` | 75.0% | 85.0% | 0.800 | 52.658 | 64.289 | 73.3% | 26.7% |
| V3 | `Hybrid+Reranker` | 90.0% | 97.5% | 0.933 | 1882.940 | 2721.975 | 73.3% | 26.7% |

> **V1/V2 与 V3 负样本指标说明**：V1/V2 使用修复前的原始代码，没有针对 BM25/Hybrid/Reranker 的 Source Pollution Gate；V3 使用下节描述的修复版本。该 Fix 位于 Retrieval Layer，重新运行 V1/V2 时同样会生效；这里保留原始 V1/V2 数据作为对照。

`Top-1 / Top-3 / MRR` 仅基于正样本计算，`Fallback Success / Source Pollution` 仅基于负样本计算。对于负样本，只有 Retrieval 不返回任何 Sources 才算 Fallback Success；返回任意 Source 均计为 Source Pollution。

## Source Pollution Fix（仅 V3）

**问题**：修复前，`BM25 / Hybrid / Reranker` 对 100% 的知识库外负样本都返回了 Sources。BM25 没有 Similarity Threshold，任何非零 Lexical Overlap 都会产生结果；Hybrid 会把弱 BM25 Signal 融合进输出；Reranker 只会重新排序已经被污染的 Hybrid Results。

**修复方案**（位于 `backend/rag_store.py`）：

1. `BM25_MIN_SCORE = 1.0`：用于 BM25 `passed_threshold` 计算的 Soft Floor，不对结果执行 Hard Filter。
2. `BM25_STRONG_THRESHOLD = 25.0`：仅用于 Hybrid Mode。当 Vector Component 没有返回结果（所有 Cosine Similarity < 0.55）时，只有 BM25 Top Score ≥ 25.0，Hybrid Gate 才接受该 Query；否则按 Out-of-knowledge Query 拒绝。
3. `search_relevant_chunks` 使用 Unified Gate：当 `passed_threshold` 为 `False` 时，返回前清空全部 Chunks。

**结果**：

| Metric | Before | After |
|---|---|---|
| Hybrid Fallback Success | 0.0% | 73.3% |
| Hybrid Source Pollution | 100.0% | 26.7% |
| Hybrid Top-1（正样本） | 62.5% | 60.0% |
| Reranker Top-1（正样本） | 82.5% | 82.5%（不变） |

> Reranker 继承 Hybrid 的 Upstream Retrieval，因此两者的 Fallback Success 与 Source Pollution 相同，均为 73.3% / 26.7%。

**Trade-off**：1 条正样本（`v1_term_rrf`）被 Hybrid Gate 错误拒绝，原因是其 Vector Similarity 未达到 0.55，且 BM25 Top Score（13.4）低于 25.0。该 Query 的正确 Source 仍可在 BM25 Rank 1 和 Reranker Rank 2 检索到。

**剩余 Source Pollution**（4/15 = 26.7%）：

- 3 条：Out-of-knowledge Query 的 Vector Cosine Similarity ≥ 0.55，属于 Vector 固有的 20% False-positive Rate
- 1 条（`v2_negative_django`）：BM25 Top Score 为 33.7，超过 25.0 的 Strong Threshold

将 `BM25_STRONG_THRESHOLD` 提高到 34 以上可以过滤 Django Case，但会增加误拒绝 Legitimate Positive Query 的风险，尤其是 BM25 Score 较高而 Vector Match 较弱的 Query。当前数值是有意选择的 Trade-off。

## Structured Chunking / Metadata（V3）

**方案**：将固定窗口切分改为结构化切分。文档先按 Markdown 标题、章节标题和段落形成语义块；超长块再使用滑窗切分。每个 Chunk 附带 `document / document_title / title / section / headings`，检索时不只匹配正文，也匹配这些元数据。

**进入检索链路的位置**：

1. BM25：标题、章节、文档名与正文一起进入词项索引。
2. Vector：新建索引时将 `文档 / 章节 / 标题 / 内容` 拼成 Embedding Text。
3. Reranker：CrossEncoder 的 document side 同样包含标题和章节元数据。

**结果**：

| Mode | Before Top-1 | After Top-1 | Before Top-3 | After Top-3 | Before MRR | After MRR |
|---|---:|---:|---:|---:|---:|---:|
| `Hybrid` | 60.0% | 75.0% | 82.5% | 85.0% | 0.710 | 0.800 |
| `Hybrid+Reranker` | 82.5% | 90.0% | 90.0% | 97.5% | 0.863 | 0.933 |

结论：结构化 Chunk + 元数据增强对最终 RAG 主路径是正收益。它的收益来自更完整的语义边界和标题/章节元数据参与检索，不是单纯增加 Chunk 数。

## Query Rewrite Ablation（V3）

**方案**：在检索前调用 LLM，将用户问题改写成更适合检索的一行 Query，目标是去掉口语、补全指代、保留专有名词。该能力已保留在代码中，但默认 RAG 路径不启用。

**全量开启结果**（Hybrid+Reranker，55 cases）：

| Metric | Original Query | Rewritten Query | Delta |
|---|---:|---:|---:|
| Top-1（40 正样本） | 90.0% | 82.5% | -7.5 pts |
| Top-3（40 正样本） | 97.5% | 90.0% | -7.5 pts |
| MRR（40 正样本） | 0.933 | 0.875 | -0.058 |
| Fallback Success（15 负样本） | 73.3% | 80.0% | +6.7 pts |
| Source Pollution（15 负样本） | 26.7% | 20.0% | -6.7 pts |

Rewrite 调用成功率为 100.0%，Fallback Count 为 0，平均 Rewrite Latency 为 3850.936 ms。逐例对比中，50 条不变，1 条负样本改善，4 条正样本变差。

结论：当前 Query Rewrite 全量开启对负样本更谨慎，但对正样本召回是负收益，因此默认关闭。后续优化方向是条件式启用：只处理明显口语化、指代省略或上下文依赖 Query。

## V3 PDF Parse 结果

| Source | Parse Method | Characters | Need OCR | OCR Used |
|---|---:|---:|---:|---:|
| `paper_faiss_2017.pdf` | text | 54858 | False | False |
| `paper_faiss_2017_scanned_pages.pdf` | ocr | 9135 | True | True |
| `paper_rag_2020.pdf` | text | 59181 | False | False |
| `paper_rag_2020_mixed_pages.pdf` | mixed | 5812 | True | True |
| `paper_rag_survey_2023.pdf` | text | 94098 | False | False |
| `paper_react_2023.pdf` | text | 93104 | False | False |

## OCR On / Off 对比

| Setting | Chunks | Parse Failures | OCR Triggered | OCR Used |
|---|---:|---:|---:|---:|
| OCR On | 1438 | 1 | 4 | 4 |
| OCR Off | 1256 | 2 | 4 | 0 |

### OCR 专项正样本（n=2）

| Mode | OCR On Top-1 | OCR Off Top-1 | OCR On Top-3 | OCR Off Top-3 | OCR On Top-K | OCR Off Top-K | OCR On MRR | OCR Off MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Vector` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.000 | 0.000 |
| `BM25` | 50.0% | 0.0% | 50.0% | 0.0% | 50.0% | 0.0% | 0.500 | 0.000 |
| `Hybrid` | 0.0% | 0.0% | 50.0% | 0.0% | 50.0% | 0.0% | 0.167 | 0.000 |
| `Hybrid+Reranker` | 0.0% | 0.0% | 0.0% | 0.0% | 50.0% | 0.0% | 0.100 | 0.000 |

### OCR Case Rank

| Case | Mode | OCR On Rank | OCR Off Rank |
|---|---|---:|---:|
| `v3_ocr_scanned_marker` | `Vector` | miss | miss |
| `v3_ocr_scanned_marker` | `BM25` | 1 | miss |
| `v3_ocr_scanned_marker` | `Hybrid` | 3 | miss |
| `v3_ocr_scanned_marker` | `Hybrid+Reranker` | 5 | miss |
| `v3_ocr_mixed_marker` | `Vector` | miss | miss |
| `v3_ocr_mixed_marker` | `BM25` | miss | miss |
| `v3_ocr_mixed_marker` | `Hybrid` | miss | miss |
| `v3_ocr_mixed_marker` | `Hybrid+Reranker` | miss | miss |

## 结论

1. 在最终 40 条正样本上，结构化 Chunk + 元数据增强后，`Hybrid+Reranker` 达到 Top-1 90.0%、Top-3 97.5%、MRR 0.933；`Hybrid` 达到 Top-1 75.0%、Top-3 85.0%、MRR 0.800。
2. 结构化 Chunk 对最终 RAG 主路径是正收益：`Hybrid+Reranker` 相比原 V3 Top-1 提升 7.5 pts，Top-3 提升 7.5 pts，MRR 提升 0.070。
3. Query Rewrite 当前全量开启是负收益：正样本 Top-1 从 90.0% 降至 82.5%，MRR 从 0.933 降至 0.875；虽然负样本 Fallback Success 从 73.3% 提升至 80.0%，但不适合作为默认策略。
4. OCR On 保留 46 个 OCR Chunks，并将 Parse Failures 从 OCR Off 的 2 降至 1。纯扫描 Fixture 在 BM25 中变为 Rank 1，在 Hybrid 中为 Rank 3，在 Hybrid+Reranker 中为 Rank 5；Vector 仍然 Miss。
5. 完成 Source Pollution Fix 后，Hybrid 与 Reranker 的 Fallback Success 为 73.3%，Source Pollution 为 26.7%。BM25-only 仍无法通过简单 Threshold 可靠修复，因为正负样本的 BM25 Score 严重重叠。
