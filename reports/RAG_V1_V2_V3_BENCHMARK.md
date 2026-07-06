# RAG V1/V2/V3 本地 Benchmark

- generated_at：`2026-07-06T06:25:15.311060+00:00`（原始时间）；数据已在 Source Pollution Fix 后更新
- branch：`codex/frontend-workspace-ui`
- base_commit：`b28d30c`（前端学习工作区改版）
- working_tree：4 个 modified files，约 20 个 new untracked files（详情见 `reports/RAG_V1_V2_V3_METRICS.json`）
- environment：本地 Windows 进程；每种 Mode 先执行 1 次 warm-up query，每个 Case 测量 1 次 Retrieval
- generation：仅评测 Retrieval，不调用 LLM Answer 或 LLM-as-Judge
- Top-K：5
- reports_source：`outputs/rag_corpus_benchmark/*.json`（原始 V1/V2）；`outputs/rag_source_pollution_fix/v3_fix3_retrieval.json`（完成 Source Pollution Fix 的 V3）
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

## Corpus 与 Index 对比

| Batch | Documents | Chunks | Index Build ms | Parse Failures | OCR Triggered | OCR Used | Cases（正/负） |
|---|---|---|---:|---:|---:|---:|---:|---:|
| V1 | 51 | 341 | 17927.051 | 1 | 2 | 2 | 20（16/4） |
| V2 | 56 | 352 | 17710.692 | 1 | 2 | 2 | 35（26/9） |
| V3 | 63 | 1292 | 23055.662 | 1 | 4 | 4 | 55（40/15） |

V1 包含本地项目文档。V2 在 V1 基础上增加 5 篇精选官方文档摘要。V3 在 V2 基础上增加 4 篇原始研究论文 PDF、2 个从真实论文页面生成的 OCR Fixtures，以及 1 份 Provenance Manifest。

`index_build_ms` 是每批执行 `rebuild_rag_index` 的实测耗时。V3 复用了已完成 Parse 的 Document Objects，以避免重复执行 OCR，因此不能把 V2/V3 的 Build Time 变化表述为严格的端到端性能提升。

## Retrieval 指标对比

| Batch | Mode | Top-1 | Top-3 | MRR | Avg ms | P95 ms | Fallback Success | Source Pollution |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | `Vector` | 31.2% | 43.8% | 0.380 | 18.626 | 22.530 | 100.0% | 0.0% |
| V1 | `BM25` | 75.0% | 93.8% | 0.833 | 14.591 | 16.716 | 0.0% | 100.0% |
| V1 | `Hybrid` | 68.8% | 81.2% | 0.768 | 29.086 | 32.551 | 0.0% | 100.0% |
| V1 | `Hybrid+Reranker` | 81.2% | 87.5% | 0.861 | 2217.814 | 2501.265 | 0.0% | 100.0% |
| V2 | `Vector` | 46.2% | 61.5% | 0.542 | 17.949 | 22.627 | 100.0% | 0.0% |
| V2 | `BM25` | 80.8% | 96.2% | 0.872 | 14.358 | 18.289 | 0.0% | 100.0% |
| V2 | `Hybrid` | 69.2% | 84.6% | 0.796 | 29.738 | 36.328 | 0.0% | 100.0% |
| V2 | `Hybrid+Reranker` | 84.6% | 92.3% | 0.886 | 2246.045 | 2584.587 | 0.0% | 100.0% |
| V3 | `Vector` | 52.5% | 67.5% | 0.596 | 17.490 | 19.730 | 80.0% | 20.0% |
| V3 | `BM25` | 72.5% | 82.5% | 0.771 | 55.821 | 62.946 | 0.0% | 100.0% |
| V3 | `Hybrid` | 60.0% | 82.5% | 0.710 | 73.402 | 83.939 | 73.3% | 26.7% |
| V3 | `Hybrid+Reranker` | 82.5% | 90.0% | 0.863 | 2114.387 | 2472.041 | 73.3% | 26.7% |

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
| OCR On | 1292 | 1 | 4 | 4 |
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

1. 在最终 40 条正样本上，`Hybrid+Reranker` 达到 Top-1 82.5%、Top-3 90.0%、MRR 0.863。与 Hybrid 相比，MRR 提升 0.153，但 Avg Latency 增加 28.8 倍（73.4 ms → 2114.4 ms）。Source Pollution Fix 未影响 Reranker 的 Top-1 和 MRR。
2. `BM25` 仍是较强 Baseline：Top-1 72.5%、Top-3 82.5%、MRR 0.771；在当前 Corpus 上优于未使用 Reranker 的 Hybrid（MRR 0.710）。
3. OCR On 增加 36 个 Chunks，并将 Parse Failures 从 2 降至 1。纯扫描 Fixture 在 BM25 中变为 Rank 1，在 Hybrid 中为 Rank 3，在 Hybrid+Reranker 中为 Rank 5；Vector 仍然 Miss。所有 Mode 都未在 Top-5 检索到 Mixed-PDF Marker，原因是 OCR 将 Marker 连接成了一个 Token。
4. 完成 Source Pollution Fix 后，Hybrid 与 Reranker 的 Fallback Success 从 0% 提升至 73.3%，Source Pollution 从 100% 降至 26.7%。剩余 Pollution 来自 Vector 固有的 20% False-positive Rate（3/4 Cases）以及 1 个异常高分 BM25 Case。BM25-only 仍无法通过简单 Threshold 可靠修复，因为正负样本的 BM25 Score 严重重叠。

## 可用于简历的谨慎 STAR 表述

- 分三个可追踪阶段扩展本地 AI Study Assistant Knowledge Base：Documents 从 51 增至 63，Indexed Chunks 从 341 增至 1292；数据包含项目笔记、精选官方文档和 4 篇 arXiv Papers。
- 构建 55-Case Offline Retrieval Benchmark（40 Positive / 15 Negative），覆盖 Fact Lookup、Concept Explanation、Acronyms、中英混合 Query、OCR 和 Out-of-knowledge Query。
- 对比 Vector、BM25、Hybrid RRF 与 CrossEncoder Reranker；最终本地实验中，Reranking 将 Hybrid MRR 从 0.710 提升至 0.863，Top-1 从 60.0% 提升至 82.5%。
- 量化 Reranking Trade-off：同一台机器和同一组 Cases 上，Avg Retrieval Latency 从 Hybrid 的 73.4 ms 增加到 Hybrid+Reranker 的 2114.4 ms。
- 增加 Native、Scanned 与 Mixed PDF Parse 检查；OCR 恢复 36 个额外 Chunks，并让纯扫描 PDF Source 在 BM25 中从 Miss 变为 Rank 1。
- 定位并部分修复 Source Pollution：通过要求 Vector Signal 或 Strong BM25 Evidence 的 Unified Gate，将 Hybrid/Reranker Pollution 从 100% 降至 26.7%。
- 保留 Source URLs、Collection Timestamps、SHA-256、Per-document Parse Methods 与 JSON Metrics，使 Benchmark Claims 可复现、可审计。

## 不应过度表述的结果

- 不要宣称通用 OCR Accuracy：OCR 专项正样本只有 2 条，并且使用的是 2 个派生 Fixtures。
- 不要宣称 Production Latency 或 Throughput：每个 Query 仅在一台本地 Windows 机器上测量 1 次，没有并发测试，也没有重复实验的 Confidence Interval。
- 不要把当前 Ranking Winner 泛化到其他 Corpus：最终正样本只有 40 条，且研究论文主题只有 4 个。
- 不要把 V1/V2/V3 的 Index Build Time 当作严格的 Scaling Curve：V3 Runner 复用了 Parsed Documents，以避免重复 OCR。
- 不要从本次实验宣称 Answer Correctness：本报告只评测 Source Ranking，没有调用 LLM Answer Generator、Human Annotator 或 LLM-as-Judge。
- 15 条负样本只能视为 Diagnostic Signal，不能视为经过校准的生产环境 Out-of-domain Distribution。
- 不要宣称 BM25 Source Pollution 已解决：当前 Simple-threshold Strategy 下，BM25 Mode 仍为 100% Pollution；Fix 仅通过 Vector Gate Component 作用于 Hybrid 和 Reranker。
- 不要宣称 V1/V2 的负样本指标与 V3 一致：V1/V2 数据采集于 Fix 之前，BM25/Hybrid/Reranker 仍显示 100% Source Pollution。
