# AI Study Assistant

## 项目概览

AI Study Assistant 是一个面向学习场景的 AI 应用，而不是普通的 ChatGPT 套壳。它通过统一的 `POST /chat` API 连接 Local RAG、Agent Tool Registry 与可选 LangGraph Runtime，让回答可以检索本地知识、调用工具并保留 Sources。系统同时记录 `Run / Trace / Tool Calls / Latency / Token Usage / Estimated Cost / Judge`，便于调试和评估。项目包含可复现的 Offline RAG Benchmark，覆盖 Positive Cases、Negative-case Fallback、Source Pollution 与 PDF/OCR Chunk Quality。

## 核心亮点

- **Unified `/chat` API**：统一承载 Chat、RAG、Agent 与 LangGraph Runtime。
- **Local RAG**：本地 FAISS/Document Index，支持 `Vector / BM25 / Hybrid RRF / CrossEncoder Reranker`。
- **Offline RAG Benchmark**：V1/V2/V3 分阶段构建 Corpus，V3 覆盖 55 Cases；Metrics 和原始结果可复现、可审计。
- **Retrieval Quality Controls**：修复 Negative-case Source Pollution，并过滤 PDF/OCR 产生的 Low-quality Chunks。
- **Tool Registry Safety**：Tools 按 `read / write / dangerous` 分级，Dangerous Tools 必须显式审批。
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
| V3 | 63 | 1292 | 55 | V2 + arXiv Paper PDFs + OCR Fixtures |

### V3 Positive Cases

| Retrieval Mode | Top-1 | Top-3 | MRR | Avg Latency |
|---|---:|---:|---:|---:|
| `Vector` | 52.5% | 67.5% | 0.596 | 17.7 ms |
| `BM25` | 72.5% | 82.5% | 0.771 | 47.5 ms |
| `Hybrid` | 60.0% | 82.5% | 0.710 | 73.4 ms |
| `Hybrid + Reranker` | **82.5%** | **90.0%** | **0.863** | 2114.4 ms |

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
| 1292 | **1242** | 50 | 16 | **45 / 45** | `No Regression` |

Hard Filter Dropped 50 个噪声 Chunks；另外 16 个 Chunks 仅标记为 `low_quality`，未直接删除。全部 45 个 OCR Chunks 均保留，Retrieval Metrics 无 Regression。

### Reports / Reproduction

- [Full benchmark report](reports/RAG_V1_V2_V3_BENCHMARK.md)
- [Machine-readable metrics](reports/RAG_V1_V2_V3_METRICS.json)
- [V3 evaluation cases](eval_cases/rag_v3_cases.json)
- [Benchmark runner](scripts/benchmark_rag_batch.py)

## 工具安全

Tool Registry 按 `read / write / dangerous` 分类。Dangerous Tool 必须经过独立的 requester / approver 凭据确认；一次性 confirmation token 与 tool name、arguments 和 requester 绑定，arguments 变化或 token 复用都会被拒绝。所有调用状态、耗时、参数摘要和审批事件写入 append-only JSONL audit log，并关联到对应 Run。

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
- Reranker 效果最好，但本地 Avg Latency 约 2.1 秒。
- OCR-specific Eval Cases 有限，不能据此宣称通用 OCR 准确率提升。
- BM25-only Source Pollution 仍是 Known Limitation；简单 threshold 无法可靠区分 Positive/Negative Cases。
- File-based RunRepository 与可选 SQLite 配置更适合单机原型，尚未面向 Distributed Execution 设计。

## 后续计划

- 对 `low_quality` Chunks 做 Down-ranking，而不是只做 Hard Filter。
- 增加 Context Selection Gate。
- 增加 History Relevance Filter。
- 为 BM25 增加 Term Coverage / Entity Matching。
- 扩充 Positive/Negative Cases 与 PDF/OCR Eval Cases。
