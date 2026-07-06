# AI Study Assistant 项目概览

- source: `README.md`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

AI Study Assistant 是一个学习场景 AI 应用，后端使用 FastAPI，前端使用 React/Vite。统一入口是 `POST /chat`，请求可以选择普通聊天、Legacy Agent、RAG 或 LangGraph Runtime。返回结构包含回答、来源、计划、轨迹、卡片、运行时信息和 Run 标识。

项目当前实现的主要边界：

- `backend/server.py` 提供 `/chat`、工具、Run、知识库、会话历史和 Judge 相关 API。
- `backend/agent_core.py` 实现 Legacy Planner + Executor。
- `backend/langgraph_runtime.py` 提供可选 LangGraph 状态图执行路径。
- `backend/tool_registry.py` 统一工具执行、危险操作确认和审计。
- `backend/rag_store.py` 负责文档解析、chunk、FAISS、BM25、Hybrid 和可选 Reranker。
- `backend/run_repository.py` 将每次运行保存为独立 JSON 文件。
- `backend/database.py`、`backend/db_models.py` 和 `backend/session_store.py` 负责可选 SQLAlchemy 会话历史与 Judge 结果持久化。

项目不要求数据库才能聊天。数据库历史关闭或连接失败时，核心聊天路径仍可运行，前端可以继续使用本地历史。RAG、LangGraph、Judge 和 Reranker也分别有独立开关或 fallback。

