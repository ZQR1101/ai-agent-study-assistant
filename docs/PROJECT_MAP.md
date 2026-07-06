# Project Map

- source: `backend/`, `frontend/`, `scripts/`, `tests/`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

## 请求路径

```text
React/Vite frontend
  -> POST /chat
  -> backend.server.chat_api
  -> backend.ai_core.run_chat_request
  -> Legacy Agent 或 LangGraph Runtime
  -> ToolRegistry.execute
  -> RAG / study / chat / persistence tools
```

`chat_api` 在执行前创建 Run，按配置加载数据库会话历史；执行后可以调用 LLM-as-Judge、保存 user/assistant 消息，并将 plan、tools、sources、flashcards、trace 和完整输出写回 RunRepository。

## 主要模块

| 模块 | 当前职责 |
|---|---|
| `backend/server.py` | FastAPI 路由、上传安全、会话与 Run API |
| `backend/schemas.py` | Pydantic 请求、响应、计划、卡片和 Judge schema |
| `backend/agent_core.py` | Legacy planner、计划校验、工具执行 |
| `backend/langgraph_runtime.py` | StateGraph 节点、条件路由、finalizer、runtime_info |
| `backend/tool_registry.py` | ToolSpec、分类、确认令牌、审计日志、统一执行入口 |
| `backend/tools.py` | chat、rag_search、study 及写入/危险工具注册 |
| `backend/rag_store.py` | 文档加载、切块、向量/BM25/Hybrid 检索、Reranker 接入 |
| `backend/rag_service.py` | RAG context、来源结构、阈值与 fallback |
| `backend/run_repository.py` | JSON Run 聚合与软删除 |
| `backend/database.py` | SQLAlchemy engine/session factory 和 schema 初始化 |
| `backend/session_store.py` | 会话、消息、Judge 结果读写 |
| `backend/judge_service.py` | LLM-as-Judge prompt、解析、分数归一化与 verdict |

## 存储边界

- `docs/`：知识库原始文件。
- `rag_index/`：FAISS 索引与 chunks 缓存，未提交 Git。
- `data/runs/`：RunRepository JSON 文件，未提交 Git。
- PostgreSQL：可选会话、消息和 JudgeEvaluation 表。
- 浏览器 localStorage：数据库历史不可用时的前端历史 fallback。

