# AI Study Assistant

![Cover](images/cover.png)

## 项目简介

AI Study Assistant 是一个基于 **FastAPI + RAG + Agent Routing** 的 AI 学习助手项目。它不是一个简单的 chatbot，而是一个面向学习场景的 AI 应用原型：用户可以进行普通聊天、知识解释、内容总结、自动出题、本地知识库问答和学习模式生成。

项目的核心目标是把练习型 AI Demo 升级成更接近真实产品形态的 AI 应用：后端提供统一可配置的 `/chat` 接口，前端提供参数控制台，RAG 返回 `answer / sources / trace`，并且在知识库未命中时进行明确 fallback，避免模型基于空知识库内容继续胡乱总结或出题。

## 项目架构

![Architecture](images/architecture.png)

## 核心功能

- 普通聊天：基础 AI 对话能力。
- 知识解释：把复杂概念解释成适合学习者理解的中文。
- 内容总结：总结用户输入或知识库检索内容。
- 自动出题：根据知识点生成练习题和答案。
- PDF 上传：上传 PDF 后自动重建本地知识库索引。
- 本地知识库问答：基于 `docs/` 中的本地资料进行 RAG 问答。
- 学习模式 `learn`：生成知识讲解、总结、练习题和学习建议。
- 统一 `/chat` 接口：通过 `mode` 统一调度 `chat / rag / explain / summarize / quiz / learn / auto`。
- 前端参数控制：可调整 `mode / model / temperature / use_rag / use_agent / top_k`。
- RAG fallback：知识库低相关或未命中时，切换到普通 LLM，并明确提示未使用知识库。
- `sources` 展示：显示回答参考的知识库文件来源。
- `trace` 展示：显示系统执行路径，包括 RAG 分数、阈值、fallback 状态等。
- Agent / Router：当前版本提供轻量任务路由，将请求分发到不同能力模块。

## 技术栈

### Backend

- Python
- FastAPI
- LangChain / langchain-openai
- FAISS
- SentenceTransformer
- pypdf
- python-dotenv
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### AI / RAG

- OpenAI-compatible Chat API
- 本地文档切分
- Embedding 向量化
- FAISS 相似度检索
- 相似度阈值过滤
- RAG fallback
- Agent Routing

## 项目结构

```text
ai-study-assistant/
|-- backend/
|   |-- server.py          # FastAPI 路由与旧接口
|   |-- ai_core.py         # LLM 构造、RAG 调度、Agent Router、/chat 核心逻辑
|   |-- rag_store.py       # 文档加载、chunk 构建、FAISS 索引、相似度检索
|   |-- schemas.py         # ChatRequest / ChatResponse schema
|   |-- langgraph_demo.py  # LangGraph 学习工作流实验
|   `-- .env.example       # 环境变量示例
|-- frontend/
|   |-- index.html         # 前端页面
|   |-- app.js             # 前端请求与渲染逻辑
|   `-- style.css          # 页面样式
|-- docs/                  # 本地知识库文档，支持 txt / md / pdf
|-- rag_index/             # FAISS 索引与 chunks 缓存
|-- images/
|   |-- cover.png
|   `-- architecture.png
|-- requirements.txt
`-- README.md
```

## 运行方式

### 1. 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

### 2. 配置 `.env`

参考 `backend/.env.example`，在项目根目录创建 `.env` 文件：

```env
MY_MIMO_API_KEY=your_api_key_here
```

当前后端默认使用 OpenAI-compatible 接口：

```text
https://token-plan-cn.xiaomimimo.com/v1
```

默认模型：

```text
mimo-v2.5
```

### 3. 启动后端

在项目根目录执行：

```bash
uvicorn backend.server:app --reload
```

后端默认运行在：

```text
http://127.0.0.1:8000
```

FastAPI Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

### 4. 打开前端

推荐用本地静态服务打开前端：

```bash
python -m http.server 5500
```

然后访问：

```text
http://127.0.0.1:5500/frontend/
```

也可以直接双击打开：

```text
frontend/index.html
```

## API 示例

### 统一 `/chat` 接口

```http
POST /chat
Content-Type: application/json
```

请求示例：

```json
{
  "message": "请用简单中文解释 RAG 是什么",
  "mode": "explain",
  "model": "mimo-v2.5",
  "temperature": 0.7,
  "use_agent": false,
  "use_rag": false,
  "top_k": 3
}
```

响应示例：

```json
{
  "answer": "RAG 是检索增强生成...",
  "mode": "explain",
  "model": "mimo-v2.5",
  "sources": [],
  "trace": [
    "收到用户请求",
    "mode：explain",
    "model：mimo-v2.5",
    "temperature：0.7",
    "use_rag：False",
    "最终执行的模式：explain",
    "是否启用 fallback：否"
  ]
}
```

### RAG 请求示例

```json
{
  "message": "prompt engineering 是什么？",
  "mode": "summarize",
  "model": "mimo-v2.5",
  "temperature": 0.3,
  "use_agent": false,
  "use_rag": true,
  "top_k": 3
}
```

命中知识库时，`sources` 会返回参考文件：

```json
{
  "sources": ["genai_04_prompt_engineering.md"]
}
```

### 旧接口

项目仍保留早期接口，便于兼容、调试和对比：

- `POST /explain`
- `POST /summarize`
- `POST /quiz`
- `POST /rag`
- `POST /agent`
- `POST /learn`
- `POST /upload`
- `POST /rebuild-index`
- `POST /debug-rag`

## 前端使用说明

前端页面相当于一个轻量参数控制台：

- `mode`：选择 `chat`、`rag`、`explain`、`summarize`、`quiz`、`learn`、`auto`
- `model`：当前前端默认使用 `mimo-v2.5`
- `temperature`：控制模型输出随机性，范围 `0.0 - 2.0`
- `use_agent`：是否启用 Agent Router
- `use_rag`：是否启用知识库检索
- `top_k`：检索最相关的文档片段数量，范围 `1 - 10`

常见使用方式：

1. 普通提问：选择 `chat`，关闭 `use_rag`。
2. 知识库问答：选择 `rag`，开启 `use_rag`。
3. 基于知识库总结：选择 `summarize`，开启 `use_rag`。
4. 基于知识库出题：选择 `quiz`，开启 `use_rag`。
5. 系统化学习：选择 `learn`，可根据需要开启 `use_rag`。
6. 自动路由：选择 `auto` 或开启 `use_agent`。

## RAG 工作流程

当前项目的 RAG 流程如下：

1. **上传 PDF / 文档**：用户可以通过前端上传 PDF，也可以把 `.txt / .md / .pdf` 文件放入 `docs/`。
2. **文本提取**：后端读取文本文件；PDF 文件通过 `pypdf` 提取文本。
3. **文本切分**：文档内容被切分成多个 chunk，并保留来源文件名。
4. **向量化**：使用 `SentenceTransformer` 将 chunk 转为 embedding。
5. **向量索引**：使用 FAISS 保存本地向量索引，并缓存 chunks。
6. **相似度检索**：用户提问时，将问题向量化，在 FAISS 中检索 top_k 个相似 chunk。
7. **阈值判断**：系统检查最高相似度 `max_score` 是否达到阈值 `threshold`。
8. **命中则基于知识库生成**：如果通过阈值，将检索到的 context 注入 prompt，生成基于知识库的回答。
9. **未命中则 fallback 到普通 LLM**：如果低于阈值或没有 chunk，系统不会返回无关 sources，也不会让总结、出题、学习模式基于空内容继续生成，而是切换到普通 LLM fallback。
10. **返回结构化结果**：最终返回 `answer / sources / trace`。

重点行为：

- 当 RAG 检索结果低于相似度阈值时，`sources` 返回空列表。
- 系统不会展示无关文件，例如把不相关的 prompt engineering 文档当成 LangGraph 来源。
- `summarize / quiz / learn` 不会基于“知识库中没有相关内容”这句话继续总结或出题。
- fallback 会在回答中明确说明未使用知识库。
- trace 会记录 `max_score`、`threshold`、`sources` 和 fallback 状态，方便调试和解释。

RAG context 的核心结构：

```python
{
    "found": bool,
    "context": str,
    "sources": list[str],
    "max_score": float | None
}
```

## Agent / Router 工作流程

当前版本的 Agent / Router 是一个轻量任务分类器，不是完整的 Planner + Executor Agent。

它主要负责根据用户请求和 `mode / use_agent` 参数，将任务分发到不同能力模块，例如：

- `chat`
- `explain`
- `summarize`
- `quiz`
- `rag`
- `learn`

当前 Router 的简化分类逻辑：

```text
1 = explain
2 = summarize
3 = quiz
4 = rag
```

执行流程：

1. 用户输入请求。
2. 如果选择 `auto` 或开启 `use_agent`，后端进入 `agent_router()`。
3. Router prompt 请求模型判断任务类别。
4. 根据分类结果调用对应函数。
5. 返回最终结果。

未来可以把这部分升级为更完整的 Agent 系统，例如：

- Planner + Executor
- Tool Registry
- LangGraph workflow
- 多步骤 Agent
- 更强的工具调用和状态管理

## sources 和 trace 说明

### sources

`sources` 表示本次回答参考的知识库文件来源。

示例：

```json
{
  "sources": ["genai_04_prompt_engineering.md"]
}
```

如果 RAG 没有可靠命中，`sources` 会返回空列表：

```json
{
  "sources": []
}
```

### trace

`trace` 表示系统执行路径，用来解释本次请求是如何被处理的。

trace 可能记录：

- 使用了哪个 `mode`
- 使用了哪个 `model`
- `temperature` 是多少
- 是否启用 RAG
- RAG `top_k`
- RAG `max_score`
- RAG 是否通过阈值
- RAG 返回了哪些 `sources`
- 是否启用 fallback
- 最终执行了哪个模式

示例：

```json
{
  "trace": [
    "收到用户请求",
    "mode：quiz",
    "model：mimo-v2.5",
    "temperature：0.3",
    "use_rag：True",
    "RAG top_k：3",
    "RAG max_score：0.4029",
    "RAG 阈值：0.5500",
    "RAG 是否通过阈值：否",
    "RAG sources：[]",
    "最终执行的模式：quiz",
    "是否启用 fallback：是"
  ]
}
```

## 当前项目亮点

- **统一可配置 `/chat` 接口**：用一个接口承载多种模式和参数，结构更接近真实 AI 应用。
- **前端参数控制台**：用户可以直接控制模式、模型、temperature、RAG、Agent 和 top_k。
- **RAG 相关性阈值过滤**：根据相似度分数过滤低相关文档，减少错误引用。
- **知识库未命中 fallback**：未命中时明确切换到普通 LLM，避免基于空 RAG 内容继续生成。
- **sources + trace 可解释输出**：不仅返回答案，也返回来源和执行路径，便于调试和展示。
- **多模式学习助手能力**：支持聊天、解释、总结、出题、RAG 问答和 learn 学习模式。
- **后端功能模块化**：`server.py / ai_core.py / rag_store.py / schemas.py` 分工清晰，便于后续升级。
- **可扩展到 LangGraph / Planner + Executor**：当前 Router 是轻量实现，后续可以自然升级为多步骤 Agent。
- **适合实习展示**：覆盖 FastAPI、LLM API、RAG、FAISS、前后端联调、可解释性输出等能力点。

## 未来优化计划 Roadmap

以下内容是未来计划，不是当前已完成能力：

- 接入 LangGraph，重构 learn 工作流。
- 将 Agent Router 升级为 Planner + Executor。
- 增加 Tool Registry，统一管理可调用工具。
- 增加会话记忆和历史上下文。
- 支持更多文档格式，例如 Word、网页链接、Markdown 批量导入。
- 增加截图示例和运行演示 GIF。
- 增加单元测试和端到端测试。
- 优化向量库持久化与检索质量。
- 增加 rerank 模块，提高 RAG 命中准确率。
- 支持流式输出，优化聊天体验。
- 使用 React / Vue 重构前端，提高组件化程度。
- 增加 Docker 或云部署说明。
