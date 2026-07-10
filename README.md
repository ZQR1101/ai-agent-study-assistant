# AI Study Assistant

## 项目概览

AI Study Assistant 是一个面向学习场景的 AI 应用，而不是普通的 ChatGPT 套壳。它通过统一的 `POST /chat` API 连接 Local RAG、Agent Tool Registry 与可选 LangGraph Runtime，让回答可以检索本地知识、调用工具并保留 Sources。系统同时记录 `Run / Trace / Tool Calls / Latency / Token Usage / Estimated Cost / Judge`，便于调试和评估。项目包含可复现的 Offline RAG Benchmark，覆盖 Positive Cases、Negative-case Fallback、Source Pollution 与 PDF/OCR Chunk Quality。

## 核心亮点

- **Unified `/chat` API**：统一承载 Chat、RAG、Agent 与 LangGraph Runtime。
- **Local RAG**：本地 FAISS/Document Index，支持 `Vector / BM25 / Hybrid RRF / CrossEncoder Reranker`，默认启用结构化 Chunk + 元数据增强。
- **Offline RAG Benchmark**：V1/V2/V3 分阶段构建 Corpus，V3 覆盖 55 Cases；Metrics 和原始结果可复现、可审计。
- **Retrieval Quality Controls**：修复 Negative-case Source Pollution，并过滤 PDF/OCR 产生的 Low-quality Chunks；Query Rewrite 已实测为当前负收益，默认关闭。
- **Tool Registry Safety**：Tools 按 `read / write / dangerous` 分级；Agent 对危险操作只创建持久化 Pending Action，必须由用户在对话中显式批准。
- **Run Observability**：统一查看 `Plan / Trace / Tool Calls / Sources / Latency / Token / Cost / Judge`。

## 功能截图

### 主界面

<img src="images/主界面截图.png" width="900" alt="AI Study Assistant 主界面">

### 功能设置

<img src="images/功能设置.png" width="500" alt="功能设置">

### Tool Calls / Runtime Trace

<img src="images/Tool%20%20Calls.png" width="900" alt="Tool Calls and Runtime Trace">

### Judge Evaluation

<img src="images/Judge.png" width="500" alt="Judge Evaluation">

## 架构与请求流程

```text
React / Vite Frontend
  → POST /chat (FastAPI)
  → Local RAG / Agent Tool Registry / optional LangGraph Runtime
  → RunRepository
  → Trace / Sources / Tool Calls / Judge
```

一次请求生成 `run_id`，RAG Sources、Plan 和 Tool Calls 随 Run 聚合保存；前端使用同一份 Run 数据展示结果与诊断信息。

## RAG Benchmark / Evaluation

这是本地 Offline Retrieval Benchmark，不调用 LLM API。Top-1、Top-3 和 MRR 基于 V3 的 40 个 Positive Cases；Fallback Success 与 Source Pollution 基于 15 个 Out-of-knowledge Negative Cases。

### Corpus

| Version | Docs | Chunks | Cases | Corpus 说明 |
|---|---:|---:|---:|---|
| V1 | 51 | 341 | 20 | 本地项目文档 |
| V2 | 56 | 352 | 35 | V1 + 5 篇精选官方文档摘要 |
| V3 | 63 | 1438 | 55 | V2 + arXiv Paper PDFs + OCR Fixtures；结构化 Chunk + 元数据增强 |

### V3 Positive Cases

| Retrieval Mode | Top-1 | Top-3 | MRR | Avg Latency | P95 Latency |
|---|---:|---:|---:|---:|---:|
| `Vector` | 57.5% | 65.0% | 0.617 | 17.0 ms | 20.7 ms |
| `BM25` | 75.0% | 87.5% | 0.812 | 41.4 ms | 50.5 ms |
| `Hybrid` | 75.0% | 85.0% | 0.800 | 53.3 ms | 64.3 ms |
| `Hybrid + Reranker` | **90.0%** | **97.5%** | **0.933** | 1945.7 ms | 2722.0 ms |

### Source Pollution Fix

| Metric | Before | After |
|---|---:|---:|
| `Hybrid / Reranker Source Pollution` | 100.0% | **26.7%** |
| `Hybrid / Reranker Fallback Success` | 0.0% | **73.3%** |
| `Reranker Top-1 (Positive)` | 82.5% | 82.5% — Unchanged |
| `Reranker MRR (Positive)` | 0.863 | 0.863 — Unchanged |

修复在 Hybrid 上游增加 Unified Gate：当 Vector 无有效结果时，只有足够强的 BM25 Signal 才允许返回 Sources。Reranker 继承该 Gate，因此 Negative-case Fallback 得到改善，同时 Positive-case Top-1/MRR 保持不变。

### Chunk Quality Filter

| Before | After | Dropped | Flagged `low_quality` | OCR Chunks Retained | Retrieval Metrics |
|---:|---:|---:|---:|---:|---|
| 1500 raw candidates | **1438 indexed** | 62 | 13 | **46** | `Hybrid / Reranker improved` |

Hard Filter Dropped 62 个噪声 Chunks；另外 13 个 Chunks 仅标记为 `low_quality`，未直接删除。当前重建索引保留 46 个 OCR Chunks，Chunk Quality Filter 正常工作。

### Structured Chunking / Metadata

当前默认启用结构化 Chunk 切分：按 Markdown 标题、章节标题和段落聚合，超长段落再滑窗切分。每个 Chunk 附带 `document / document_title / title / section / headings`，并让这些元数据进入 BM25、Embedding Text 和 Reranker 输入。

| Mode | Before Top-1 | After Top-1 | Before MRR | After MRR | Verdict |
|---|---:|---:|---:|---:|---|
| `Hybrid` | 60.0% | **75.0%** | 0.710 | **0.800** | Positive |
| `Hybrid + Reranker` | 82.5% | **90.0%** | 0.863 | **0.933** | Positive |

结论：结构化 Chunk + 元数据增强对最终 RAG 主路径是正收益，尤其提升 Hybrid 与 Hybrid+Reranker。收益来自更完整的语义边界和标题/章节元数据参与检索，而不是单纯增加 Chunk 数。

### Query Rewrite Ablation

Query Rewrite 能把用户问题交给 LLM 改写成更“检索友好”的形式，但当前全量开启会损伤正样本召回。因此代码保留 `rewrite_query_for_retrieval` 能力，默认 RAG 路径不启用，后续只做条件式启用。

| Hybrid + Reranker | Original Query | Rewritten Query | Delta |
|---|---:|---:|---:|
| Top-1 | **90.0%** | 82.5% | -7.5 pts |
| Top-3 | **97.5%** | 90.0% | -7.5 pts |
| MRR | **0.933** | 0.875 | -0.058 |
| Fallback Success | 73.3% | **80.0%** | +6.7 pts |
| Source Pollution | 26.7% | **20.0%** | -6.7 pts |

Rewrite 调用成功率为 100.0%，Fallback Count 为 0，平均 Rewrite Latency 为 3850.9 ms。结论：当前 Rewrite 对负样本更谨慎，但对正样本召回是负收益；默认关闭，待优化为”仅在明显口语化、指代省略或上下文依赖时启用”。

### BM25 Term Coverage / Entity Gate

BM25 路径新增 Term Coverage 与 Entity Matching Gate，用于过滤 BM25 的假阳性候选。对每条候选分别计算：

- `bm25_term_coverage`：命中 BM25 词项 / Query 词项
- `bm25_entity_match_count`：命中实体词项数量
- `bm25_entity_term_coverage`：命中实体词项 / Query 实体词项

过滤规则：若 Query 含实体词（缩写、英文术语、API 路径、文件名、数字等），至少一个实体必须命中且实体覆盖率达标；若无实体词，候选必须通过最低词项覆盖率门槛。目标是对 “Kubernetes HPA” 匹配到通用中文项目文档这类假阳性做精准拦截。

Benchmark 在 359 Docs / 2579 Chunks 的当前工作区索引上运行，对比关闭/开启 Gate 两组实验（40 Positive + 15 Negative Cases，top_k=5）：

#### With BM25 Gate（当前默认）

| Mode | Top-1 | Top-3 | MRR | Fallback Success | Source Pollution |
|---|---:|---:|---:|---:|---:|
| `BM25` | 75.0% | 87.5% | 0.812 | 40.0% | 60.0% |
| `Hybrid` | 72.5% | 85.0% | 0.799 | 80.0% | 20.0% |
| `Hybrid+Reranker` | **90.0%** | **97.5%** | **0.938** | 80.0% | 20.0% |

#### Delta（Gate 开启 vs 关闭）

| Mode | Top-1 | Top-3 | MRR | Fallback | Pollution |
|---|---:|---:|---:|---:|---:|
| `BM25` | +2.5 pts | +7.5 pts | +0.050 | +40.0 pts | **-40.0 pts** |
| `Hybrid` | -2.5 pts | 0.0 pts | -0.001 | +13.3 pts | -13.3 pts |
| `Hybrid+Reranker` | 0.0 pts | 0.0 pts | +0.005 | +13.3 pts | -13.3 pts |

#### Source Pollution 修复详情

| Mode | 已修复 | 仍残留 |
|---|---|---|
| `BM25` | 6 例（blockchain, ios, kubernetes, celery, cv, graph_neural_network） | 9 例 |
| `Hybrid` | 2 例（kubernetes, prometheus） | 3 例（django, graph_neural_network, quantum） |
| `Hybrid+Reranker` | 2 例（kubernetes, prometheus） | 3 例（django, graph_neural_network, quantum） |

**结论**：BM25 Gate 整体为正收益。BM25-only Source Pollution 从 100.0% 降至 60.0%，同时 Top-1 / Top-3 / MRR 均提升。Hybrid+Reranker 保持 Top-1 和 Top-3 不变，MRR 微增，Source Pollution 从 33.3% 降至 20.0%。残留污染案例指向语义相邻但超出知识范围的实体，下一步应加强多实体 Query 的实体覆盖率要求，并维护高价值领域别名 Allowlist 以避免过度过滤正样本。

### Reports / Reproduction

- [Full benchmark report](reports/RAG_V1_V2_V3_BENCHMARK.md)
- [BM25 Term/Entity Gate report](reports/RAG_BM25_ENTITY_GATE_REPORT.md)
- [Machine-readable metrics](reports/RAG_V1_V2_V3_METRICS.json)
- [V3 evaluation cases](eval_cases/rag_v3_cases.json)
- [Benchmark runner](scripts/benchmark_rag_batch.py)
- Latest local run：`outputs/rag_query_rewrite_opt/v3_query_rewrite_opt_full_retrieval.json`
- Query rewrite ablation：`outputs/rag_query_rewrite_opt/v3_query_rewrite_eval.json`

## 工具安全

Tool Registry 按 `read / write / dangerous` 分类。Dangerous Tool 必须经过独立的 requester / approver 凭据确认；一次性 confirmation token 与 tool name、arguments 和 requester 绑定，arguments 变化或 token 复用都会被拒绝。所有调用状态、耗时、参数摘要和审批事件写入 append-only JSONL audit log，并关联到对应 Run。

Agent 遇到删除、清空、重置或重建操作时不会直接执行，而是创建持久化的 Pending Action，并将 Run 标记为 `awaiting_action`。对话内确认卡片会展示影响范围、精确参数、可撤销性和过期时间；用户可批准或拒绝。批准后后端仍通过原有一次性 confirmation token 执行，拒绝、过期、失败和重复提交都会被记录或拦截。默认 Pending Action 保存在 `data/pending_actions`，有效期为 300 秒，可通过 `PENDING_ACTIONS_DIR` 和 `PENDING_ACTION_TTL_SECONDS` 调整。

默认情况下，`TOOL_APPROVAL_KEY` 与 `TOOL_APPROVER_KEY` 都需要 32 位以上且互不相同。本地开发如果嫌长 key 在 Swagger / Dev Tool Debugger 里复制麻烦，可以在 `.env` 打开：

```env
ENABLE_INSECURE_DEV_TOOL_KEYS=true
TOOL_APPROVAL_KEY=dev-req1
TOOL_APPROVER_KEY=dev-app1
```

这个开关只适合本机开发；共享、测试、生产环境请保持关闭。

## 运行可观测性

`/chat` 返回 `run_id`，`/runs/{run_id}` 可查看 `status / plan / tools / audit / artifacts / output / metadata`。Run View 聚合 `Trace / RAG Sources / Step Latency / Token Usage / Estimated Cost / Judge`，便于定位 Planner Fallback、Retrieval Miss 与 Tool Failure。

## 快速开始

要求 Python 3.11+ 与 Node.js 18+。

### 安装后端依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中配置模型 API key；本地离线测试不需要 key。

### 启动后端

```powershell
uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
```

### 启动前端

```powershell
npm install
npm run dev
```

前端默认地址：`http://127.0.0.1:5500`。

### 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_setup.py
npm run build
```

## 测试结果

2026-07-06 本地验证：

| Check | Result |
|---|---|
| Python Test Suite | **235 passed, 60 subtests passed** |
| `scripts/check_setup.py` | **OK** |
| Frontend Build (`npm run build`) | **Successful** |

## 当前限制

- Benchmark 是本地 Offline Evaluation，不代表生产环境指标或通用准确率。
- V3 仅包含 40 个 Positive Cases 和 15 个 Negative Cases，样本规模仍有限。
- Reranker 效果最好，但本地 Avg Latency 约 1.9 秒。
- OCR-specific Eval Cases 有限，不能据此宣称通用 OCR 准确率提升。
- BM25-only Source Pollution 经 Term/Entity Gate 已从 100.0% 降至 60.0%，但仍残留在语义相邻但超出知识范围的实体上，如 quantum / django / graph_neural_network。
- Query Rewrite 当前全量开启会降低正样本召回，因此默认关闭。
- File-based RunRepository 与可选 SQLite 配置更适合单机原型，尚未面向 Distributed Execution 设计。

## 后续计划

- 对 `low_quality` Chunks 做 Down-ranking，而不是只做 Hard Filter。
- 增加 Context Selection Gate。
- 增加 History Relevance Filter。
- 将 Query Rewrite 改成条件式启用：仅处理口语化、指代省略或上下文依赖 Query。
- BM25 Term Coverage / Entity Gate 已实现，后续加强多实体 Query 覆盖率与领域别名 Allowlist。
- 扩充 Positive/Negative Cases 与 PDF/OCR Eval Cases。
