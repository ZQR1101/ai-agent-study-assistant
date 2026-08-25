# Agent 与 LangGraph Runtime

- source: `backend/agent_core.py`, `backend/langgraph_runtime.py`, `backend/ai_core.py`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

项目保留两条执行路径。Legacy Agent 是默认 Planner + Executor；LangGraph 通过 `/chat` 的 `use_langgraph=true` 可选启用。

Legacy planner 生成结构化计划并限制工具名。LLM 规划结果需要通过 schema 校验；解析或调用失败时会生成 fallback plan。Executor 按步骤调用 Tool Registry，并在 shared context 中传递 RAG context、sources、历史和前序输出。

LangGraph Runtime 使用 `StateGraph`。当前节点与 Tool Registry 对齐：planner、`rag_search`、`study`、chat 和 finalizer。`study` 通过 `arguments.operation`（explain / summarize / flashcard / quiz）在同一节点内顺序执行学习操作。条件路由根据 intent 与计划决定下一个节点。节点仍复用同一 Tool Registry，而不是维护第二套工具实现。

`planner_mode=rule` 是默认稳定选项；`planner_mode=llm` 使用 JSON-only prompt、Pydantic 计划校验和 rule fallback。`runtime_info` 记录 graph_path、tool_calls、planner_mode、fallback、RAG 候选和 Reranker 状态，供前端与评测脚本检查。

