## server.py 接口说明

### GET /

用于测试后端是否启动成功，直接返回欢迎信息，不调用 AI 功能。

### POST /explain

接收用户输入的文本，调用 `explain()` 生成知识解释。

### POST /summarize

接收用户输入的文本，调用 `summarize()` 生成总结。

### POST /quiz

接收用户输入的文本，调用 `generate_questions()` 生成练习题。

### POST /rag

接收用户问题，调用 `rag_answer_with_sources()`，根据知识库内容生成回答，并返回引用来源。

### POST /agent

接收用户输入，调用 `agent_router()`，由 Agent Router 判断任务类型并调用对应功能。

### POST /upload

接收用户上传的文件，将文件保存到 `docs/` 目录，然后调用 `rebuild_rag_index()` 重建知识库索引。

### POST /learn

接收学习主题，调用 `learning_workflow()`，执行学习工作流。

### POST /rebuild-index

手动调用 `rebuild_rag_index()`，重新构建 RAG 知识库索引。

## ai_core.py：模型初始化

`ai_core.py` 会通过 `load_dotenv()` 读取 `.env` 文件中的 API Key，然后使用 `ChatOpenAI` 创建大模型对象。后续解释、总结、出题、RAG 回答和 Agent Router 都会调用这个 `llm` 对象。

## backend/ai_core.py：AI 功能层

`ai_core.py` 是项目的 AI 功能中心，主要负责创建大模型对象、封装基础 AI 功能、调用 RAG 检索、执行 Agent Router 和学习工作流。

### 1. 模型初始化

通过 `ChatOpenAI` 创建 `llm` 对象，API Key 从 `.env` 文件中的 `MY_MIMO_API_KEY` 读取，模型使用 `mimo-v2.5`。

### 2. 基础 AI 功能

- `explain(text)`：解释知识点
- `summarize(text)`：总结文本内容
- `generate_questions(text)`：根据内容生成练习题

这三个函数的共同流程是：接收文本 → 拼接 Prompt → 调用大模型 → 返回回答。

### 3. RAG 问答功能

- `rag_answer_with_sources(question)`：调用 `search_relevant_chunks()` 检索知识库片段，再把相关内容作为上下文交给大模型回答，返回答案和来源。
- `rag_answer(question)`：对 `rag_answer_with_sources()` 的结果进行格式化，生成带参考来源的文本回答。

### 4. Agent Router

`agent_router(user_input)` 会先让大模型判断用户请求属于解释、总结、出题还是 RAG 问答，然后根据分类结果调用对应函数。当前版本属于基于 LLM 的简单任务路由器。

### 5. 学习工作流

`learning_workflow(topic)` 按照固定流程执行：RAG 查询 → 总结 → 出题 → 学习建议。它是一个线性 Workflow，后续可以迁移为 LangGraph 节点流程。

## 代码清理记录：ai_core.py

本次清理删除了 `ai_core.py` 中未使用的 `load_documents()` 函数，并移除了相关无用导入。同时确认 `rag_store.py` 中负责 RAG 文档读取和索引构建的函数保留不动。

清理后项目分层更加清晰：

- `ai_core.py`：负责大模型调用、Prompt 封装、RAG 回答、Agent Router
- `rag_store.py`：负责文档读取、切块、Embedding、FAISS 建索引和检索

## frontend：前端交互层

前端主要由页面结构、样式和交互逻辑组成。其中 `app.js` 负责读取用户输入、上传文件、调用后端接口，并把返回结果显示到聊天区域。

### 前端函数与后端接口对应关系

| 前端函数        | 后端接口       | 后端功能                              |
| --------------- | -------------- | ------------------------------------- |
| `uploadPDF()`   | `POST /upload` | 上传 PDF 到 `docs/`，并重建 RAG 索引  |
| `sendMessage()` | `POST /agent`  | 调用 Agent Router 自动判断任务类型    |
| `learnMode()`   | `POST /learn`  | 执行学习工作流：RAG、总结、出题、建议 |
| `ragMode()`     | `POST /rag`    | 执行知识库问答，并返回答案和引用来源  |

### 前端到后端的数据流

用户在页面输入内容  
↓  
`app.js` 使用 `fetch()` 调用 FastAPI 接口  
↓  
`server.py` 接收请求  
↓  
`ai_core.py` 调用大模型或 RAG  
↓  
`rag_store.py` 检索知识库片段  
↓  
后端返回 JSON  
↓  
前端显示结果

## FastAPI 请求与响应

前端通过 `fetch()` 调用 FastAPI 后端接口。普通文本请求会发送 JSON：

```json
{
  "text": "用户输入"
}
```

## FastAPI 如何接收前端数据

前端通过 `fetch()` 发送 JSON 数据，例如：

```json
{
  "text": "帮我解释 RAG"
}
```
