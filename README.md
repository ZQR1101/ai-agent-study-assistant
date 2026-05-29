# AI Study Assistant

![AI Study Assistant Cover](images/cover.png)

## 项目简介

AI Study Assistant 是一个面向学习场景的 AI 应用开发项目，基于 **FastAPI + RAG + Planner/Executor Agent + Tool Registry** 构建。它不是简单的 chatbot，而是围绕学习流程设计的 AI 助手：支持普通聊天、知识解释、内容总结、自动出题、本地知识库问答、学习模式、结构化记忆卡片生成、多轮上下文对话，以及可解释的 `sources / trace / plan / flashcards` 输出。

项目的目标是把一个练习型 AI Demo 逐步升级成更接近真实产品形态的 AI 应用：后端提供统一 `/chat` 接口，前端提供参数控制和学习卡片 UI，RAG 支持相关性阈值过滤和 fallback，Agent 使用 Planner + Executor 执行多步骤学习任务。

## 项目亮点

- 统一 `/chat` 接口管理多种 AI 能力，兼容旧接口。
- 可配置 `mode / model / temperature / use_rag / use_agent / top_k`。
- RAG 支持 query expansion、chunk 清洗、相似度阈值过滤和 fallback，减少弱相关文档误命中。
- `sources` 返回 `source / score / snippet`，方便查看回答参考了哪段知识库内容。
- Agent 使用 Planner + Executor 架构，能够把复合请求拆成多个工具步骤。
- Planner 使用 JSON schema prompting、few-shot 示例、Pydantic validation 和 fallback plan，提高 JSON 输出稳定性。
- Tool Registry 统一管理 `chat / rag / explain / summarize / quiz / flashcard` 工具。
- Agent step shared_context 支持步骤间上下文传递，例如先 RAG，再基于 RAG 结果解释和出题。
- structured trace 展示请求参数、RAG 检索、Planner 状态、工具执行和 fallback 状态。
- flashcard 工具返回结构化记忆卡片，前端支持点击翻面和 Canvas PNG 导出。
- conversation history 支持前端内存版多轮对话。
- `scripts/smoke_test.py` 支持核心 API 冒烟测试，便于修改后快速回归。
- 后端已拆分为多个模块，便于维护、调试和继续扩展。

## 核心功能

- 普通聊天：通用问答和多轮学习交流。
- 知识解释：用简单中文解释概念。
- 内容总结：总结用户输入或知识库检索内容。
- 自动出题：根据主题、解释结果或知识库内容生成练习题。
- PDF / 文档知识库：支持将 `.txt / .md / .pdf` 文档放入 `docs/`，也支持通过 `/upload` 上传 PDF。
- RAG 知识库问答：基于本地文档检索相关 chunk，再生成回答。
- RAG fallback：知识库未可靠命中时，不返回无关来源，并切换到普通 LLM。
- 学习模式 `learn`：生成解释、总结、练习题和学习建议。
- Agent 自动规划：`mode=auto` 且 `use_agent=true` 时，由 Planner 生成执行计划。
- Agent Plan 展示：前端单独显示 Planner 生成的工具步骤。
- Tool Registry：以统一工具接口封装聊天、RAG、解释、总结、出题和记忆卡片能力。
- Flashcard 记忆卡片：生成结构化 `front / back / tags / difficulty` 数据。
- 可视化翻转卡片：前端渲染为可点击翻面的学习卡片。
- PNG 正反面下载：前端使用 Canvas API 导出每张卡片的正面和背面图片。
- 多轮 history：前端维护内存版 `chatHistory`，后端和 Planner 可参考最近历史。
- sources 参考来源展示：显示文件名、相似度和命中文本片段。
- trace 执行路径展示：分组展示系统如何处理本次请求。
- smoke test：通过 HTTP 请求快速检查核心 `/chat` 路径。

## 技术栈

**Backend**

- Python
- FastAPI
- Pydantic
- Uvicorn
- python-dotenv

**LLM / Agent**

- LangChain
- langchain-openai
- OpenAI-compatible Chat API
- Planner + Executor Agent
- Tool Registry

**RAG**

- SentenceTransformer
- FAISS (`faiss-cpu`)
- pypdf
- 本地 `.txt / .md / .pdf` 文档加载
- query expansion
- chunk 清洗与相似度阈值过滤

**Frontend**

- HTML
- CSS
- JavaScript
- Fetch API
- Canvas API

**Testing**

- Python 标准库 `urllib.request`
- `scripts/smoke_test.py`

## 项目架构图

![AI Study Assistant Architecture](images/architecture.png)

## 项目结构

```text
ai-study-assistant/
├── backend/
│   ├── server.py          # FastAPI 路由、旧接口、/chat、/health、上传和调试接口
│   ├── schemas.py         # ChatRequest / ChatResponse / AgentPlan / SourceChunk / Flashcard schema
│   ├── ai_core.py         # 统一门面层，保留旧函数并负责 /chat 调度
│   ├── llm_service.py     # LLM 构建、模型配置、chat/explain/summarize/quiz 基础函数
│   ├── rag_service.py     # RAG 上下文获取、sources 格式化、RAG 回答和 fallback
│   ├── rag_store.py       # 文档加载、chunk 清洗、FAISS 索引构建和检索
│   ├── agent_core.py      # Planner、JSON 解析、Pydantic 校验、fallback、Executor
│   ├── tools.py           # ToolSpec 与 TOOL_REGISTRY
│   ├── history_utils.py   # history 截取、格式化和 prompt 注入辅助函数
│   ├── langgraph_demo.py  # LangGraph 学习/实验文件，主流程暂未依赖它
│   └── __init__.py
├── frontend/
│   ├── index.html         # 前端页面
│   ├── app.js             # 请求、渲染、history、flashcard 翻面和 PNG 下载逻辑
│   └── style.css          # 页面和卡片样式
├── docs/                  # 本地知识库文档，支持 txt / md / pdf
├── images/
│   ├── cover.png
│   └── architecture.png
├── scripts/
│   └── smoke_test.py      # 冒烟测试脚本
├── rag_index/             # FAISS 索引和 chunks 缓存
├── uploads/               # 上传文件目录
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 `.env`

在项目根目录创建 `.env` 文件，填写自己的 API key。不要把真实 key 提交到 Git。

```env
MY_MIMO_API_KEY=your_api_key_here
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
```

### 3. 启动后端

```bash
uvicorn backend.server:app --reload
```

后端默认地址：

```text
http://127.0.0.1:8000
```

FastAPI docs：

```text
http://127.0.0.1:8000/docs
```

健康检查：

```text
http://127.0.0.1:8000/health
```

### 4. 启动前端

在项目根目录运行：

```bash
python -m http.server 5500
```

然后访问：

```text
http://127.0.0.1:5500/frontend/
```

## 环境变量配置

当前 LLM 创建逻辑会优先读取以下环境变量：

```env
MY_MIMO_API_KEY=your_api_key_here
MIMO_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
```

说明：

- `MY_MIMO_API_KEY`、`MIMO_API_KEY`、`OPENAI_API_KEY` 三者任选其一即可，代码会按顺序读取。
- `MIMO_BASE_URL` 可选，默认值是 `https://token-plan-cn.xiaomimimo.com/v1`。
- 当前后端默认模型是 `mimo-v2.5`。
- 真实 API key 应只放在本地 `.env` 中，`.gitignore` 已忽略 `.env`。

## 后端启动

```bash
uvicorn backend.server:app --reload
```

常用后端接口：

- `GET /health`：健康检查。
- `POST /chat`：统一 AI 能力接口。
- `POST /upload`：上传 PDF 并重建知识库索引。
- `POST /rebuild-index`：手动重建 RAG 索引。
- `GET /debug-index-sources`：查看当前索引包含的 source 文件。
- `POST /debug-rag`：调试 RAG 检索结果。
- `POST /explain`、`POST /summarize`、`POST /quiz`、`POST /rag`、`POST /agent`、`POST /learn`：旧接口，仍保留用于兼容和调试。

## 前端启动

```bash
python -m http.server 5500
```

访问：

```text
http://127.0.0.1:5500/frontend/
```

前端默认请求后端：

```text
http://127.0.0.1:8000
```

## API 示例

### `/chat` 请求示例

```json
{
  "message": "根据知识库解释 agentic rag，并出 3 道练习题",
  "mode": "auto",
  "model": "mimo-v2.5",
  "temperature": 0.3,
  "use_agent": true,
  "use_rag": true,
  "top_k": 3,
  "history": []
}
```

### `/chat` 简化响应示例

```json
{
  "answer": "...",
  "mode": "agent",
  "model": "mimo-v2.5",
  "sources": [
    {
      "source": "agents_05_agentic-rag.md",
      "score": 0.7337,
      "snippet": "..."
    }
  ],
  "plan": [
    {
      "tool": "rag",
      "input": "agentic rag",
      "reason": "用户要求根据知识库回答，先检索相关内容"
    },
    {
      "tool": "explain",
      "input": "基于知识库内容解释 agentic rag",
      "reason": "解释检索到的概念"
    },
    {
      "tool": "quiz",
      "input": "基于 agentic rag 生成 3 道练习题",
      "reason": "用户要求生成练习题"
    }
  ],
  "trace": [
    {
      "title": "Agent 执行",
      "items": [
        "Agent Planner JSON parse：成功",
        "Agent Planner schema validate：成功",
        "Agent Step 1 tool=rag"
      ]
    }
  ],
  "flashcards": []
}
```

## 前端使用说明

前端页面提供一个轻量参数控制台：

- `mode`：选择 `chat / rag / explain / summarize / quiz / learn / auto`。
- `model`：当前主要使用 `mimo-v2.5`。
- `temperature`：控制输出随机性，范围 `0.0 - 2.0`。
- `use_agent`：开启后进入 Agent Planner + Executor。
- `use_rag`：开启后在普通模式中使用知识库增强；选择 `mode=rag` 时会自动启用。
- `top_k`：RAG 检索候选数量，范围 `1 - 10`。
- PDF 上传：上传后会触发知识库索引重建。
- 清空对话：清空当前浏览器内存中的 history 和页面消息。

常见使用方式：

- 普通问答：`mode=chat`，关闭 `use_rag` 和 `use_agent`。
- 知识库问答：`mode=rag`。
- 基于知识库总结：`mode=summarize`，开启 `use_rag`。
- 基于知识库出题：`mode=quiz`，开启 `use_rag`。
- 学习模式：`mode=learn`，可按需开启 `use_rag`。
- Agent 多步骤任务：`mode=auto`，开启 `use_agent`，可按需开启 `use_rag`。
- 记忆卡片：在 Agent 请求中包含“记忆卡片 / flashcard”等意图，Planner 可选择 `flashcard` 工具。

## RAG 工作流程

当前 RAG 流程如下：

1. 文档进入知识库：将 `.txt / .md / .pdf` 文件放入 `docs/`，或通过前端上传 PDF。
2. 文本提取：文本文件直接读取，PDF 使用 `pypdf` 提取文本。
3. 文本切分：按固定窗口和 overlap 切分为 chunks。
4. chunk 清洗：过滤空字符串、过短文本、Markdown 表格分隔符、纯符号片段和语义字符过少的 chunk。
5. query expansion：对 `agentic rag`、`langgraph`、`prompt engineering`、`rag` 等关键词做简单扩展。
6. 向量化：使用 `SentenceTransformer` 生成 embedding。
7. 相似度检索：使用 FAISS 本地向量索引检索 top_k 候选。
8. 阈值判断：默认相似度阈值为 `0.55`，低于阈值会判定为未可靠命中。
9. 命中处理：返回 `context + sources`，并让 LLM 优先基于知识库内容回答。
10. 未命中处理：不返回无关 sources，不把空 RAG 提示交给总结、出题或学习模式，而是 fallback 到普通 LLM。
11. 返回结果：统一返回 `answer / sources / trace`，Agent 场景还会返回 `plan`。

重点行为：

当 RAG 检索结果低于相似度阈值时，系统不会返回无关 sources，也不会让总结、出题、学习模式基于空内容继续生成，而是切换到普通 LLM fallback，并在 trace 中记录 `max_score`、`threshold`、`sources` 和 fallback 状态。

## Agent 工作流程

当前 Agent 是基础版 Planner + Executor，没有在主流程中使用 LangGraph。

执行流程：

1. 用户请求进入 `/chat`。
2. 当 `mode=auto` 且 `use_agent=true` 时，进入 Agent。
3. Planner 根据用户请求和最近 history 生成 JSON plan。
4. Planner prompt 中包含工具列表、AgentPlan schema、few-shot 示例和 JSON-only 要求。
5. 后端使用 Pydantic 对 Planner 输出进行 schema validation。
6. 如果 JSON 解析失败、schema 校验失败、steps 为空或工具名未知，则进入 fallback plan。
7. Executor 按 plan 顺序执行工具。
8. Executor 通过 Tool Registry 查找工具实现。
9. shared_context 在 step 间传递 `rag_context / sources / step_outputs / last_output`。
10. 最终返回 `answer / sources / plan / trace / flashcards`。

示例工作流：

```text
rag step 检索知识库
↓
explain step 基于 rag_context 解释
↓
quiz step 基于 rag_context + previous_step_output 出题
```

## Tool Registry 说明

Tool Registry 位于 `backend/tools.py`，通过 `ToolSpec` 统一描述工具名称、说明和执行函数。

当前注册工具：

- `chat`：普通聊天或通用问答。
- `rag`：从本地知识库检索相关内容并回答。
- `explain`：用简单中文解释概念。
- `summarize`：总结输入内容。
- `quiz`：根据内容生成练习题。
- `flashcard`：根据知识点生成适合复习的记忆卡片，包括正面问题、背面答案、标签和难度。

这样做的好处是 Planner 不需要硬编码所有能力，Executor 也可以通过 registry 统一调用工具。后续新增工具时，只需要实现工具函数并注册到 `TOOL_REGISTRY`。

## Flashcard 功能说明

flashcard 工具用于生成结构化学习记忆卡片。每张卡片包含：

- `front`：卡片正面问题。
- `back`：卡片背面答案。
- `tags`：标签列表。
- `difficulty`：难度，取值为 `easy / medium / hard`。

前端展示能力：

- 将 `flashcards` 渲染为真正的可视化卡片。
- 点击卡片可以翻面查看答案。
- 难度标签按颜色区分：简单绿色、中等蓝色、困难红色。
- 每张卡片可以下载正面和背面 PNG。
- PNG 下载由前端 Canvas API 生成，不走后端。
- 部分浏览器可能会提示是否允许多个文件下载。

如果 flashcard 的结构化 JSON 生成失败，后端会 fallback 为 Markdown 文本回答，并保持服务不崩溃。

## sources / trace / plan / flashcards 响应说明

### sources

`sources` 表示本次回答参考的知识库来源。每项通常包含：

- `source`：文件名。
- `score`：相似度分数。
- `snippet`：命中的文本片段。

示例：

```json
{
  "source": "agents_05_agentic-rag.md",
  "score": 0.7337,
  "snippet": "Agentic RAG ..."
}
```

RAG 未可靠命中时，`sources` 返回空列表。

### trace

`trace` 表示系统执行路径，目前是结构化分组数据。它可能包含：

- 请求参数。
- 是否使用 history。
- RAG 检索过程。
- RAG `top_k / max_score / threshold / sources`。
- 是否通过阈值。
- 是否启用 fallback。
- Agent Planner JSON parse 状态。
- Agent Planner schema validate 状态。
- 工具执行情况。
- 每个 Agent step 是否使用上下文。

### plan

`plan` 表示 Agent Planner 生成的任务计划。每个 step 包含：

- `tool`：工具名。
- `input`：传给工具的输入。
- `reason`：Planner 选择该工具的原因。

非 Agent 模式下，`plan` 通常为空列表。

### flashcards

`flashcards` 表示结构化记忆卡片数据。每项包含：

- `front`
- `back`
- `tags`
- `difficulty`

非 flashcard 请求下，`flashcards` 返回空列表。

## Conversation History 说明

当前项目支持基础多轮上下文：

- 前端在浏览器内存中维护 `chatHistory`。
- 每次发送请求时，前端会附带最近几条 history。
- 后端会把最近 history 注入 prompt，帮助模型理解“它”“刚才内容”等指代。
- Agent Planner 也可以参考 history 生成更合适的计划。
- 刷新页面后 history 会丢失。
- 当前没有数据库持久化，也没有用户级长期记忆。

## Smoke Test 冒烟测试说明

先启动后端：

```bash
uvicorn backend.server:app --reload
```

运行所有测试：

```bash
python scripts/smoke_test.py --case all
```

单独运行某个测试：

```bash
python scripts/smoke_test.py --case explain
python scripts/smoke_test.py --case rag
python scripts/smoke_test.py --case agent
python scripts/smoke_test.py --case agentic-rag
python scripts/smoke_test.py --case flashcard
```

smoke test 会检查：

- `explain`：解释模式是否返回 `answer / mode / trace`。
- `rag`：RAG 模式是否返回 `answer / sources / trace`。
- `agent`：Agent 模式是否返回 `answer / plan / trace`。
- `agentic-rag`：Agent + RAG 是否返回 `answer / sources / plan / trace`。
- `flashcard`：flashcard 请求是否返回 `answer / flashcards / trace`。

脚本依赖 `/health` 判断后端是否启动。它使用 Python 标准库 `urllib.request`，不需要额外安装 requests。

注意：这些测试会真实调用后端和 LLM，因此需要有效 API key，也可能产生 API 调用消耗。

## 当前限制

- 当前 history 是前端内存版，刷新页面后会丢失。
- 当前没有用户系统、登录态和数据库。
- 当前没有完整部署流程。
- 当前主流程没有接入 LangGraph workflow。
- 当前没有 reranker，RAG 精度主要依赖 embedding 检索、关键词增强、chunk 清洗和阈值过滤。
- 当前没有完整单元测试，仅有 smoke test 覆盖核心路径。
- Flashcard 下载由前端 Canvas 生成 PNG，最终样式可能受浏览器字体和下载策略影响。
- 前端使用原生 HTML / CSS / JavaScript，适合轻量演示，复杂状态管理能力有限。
- 部分后端日志和旧接口文案仍有早期编码显示问题，后续可以统一整理。

## Roadmap 未来计划

以下是未来计划，不代表当前已完成：

- 接入 LangGraph，重构 Agent workflow。
- 将 Agent 进一步升级为更完整的 Planner + Executor + Evaluator 流程。
- 增加 reranker，提高 RAG 精度。
- 优化向量库持久化、增量更新和检索质量。
- 增加后端持久化 session memory。
- 增加用户文件管理和知识库管理页面。
- 增加更多工具，例如 calculator、web search、code explainer。
- 增加单元测试、集成测试和 CI。
- 增加 Docker / 云部署说明。
- 增加更多截图、Demo GIF 和在线演示。

## Demo Screenshots

TODO:

- 主界面截图
- RAG 命中示例
- Agent Plan 示例
- Flashcard 可视化卡片示例
