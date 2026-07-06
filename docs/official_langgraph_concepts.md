# LangGraph Concepts

- source: `https://docs.langchain.com/oss/python/langgraph/overview`
- source: `https://docs.langchain.com/oss/python/langgraph/graph-api`
- curated_at: `2026-07-06 Asia/Shanghai`
- publisher: `LangChain official documentation`

LangGraph 是面向长运行、有状态工作流和 Agent 的低层 orchestration runtime。它关注状态、节点、边、持久执行、streaming 和 human-in-the-loop，而不是替用户决定 prompt 或 Agent 架构。

Graph API 以 StateGraph 为主要构建方式。节点是接收 state 并返回 state 更新的函数；普通 edge 表示固定流转，conditional edge 根据路由函数选择下一个节点。图在 compile 后才能 invoke。START 和 END 表示图的入口与结束。

官方文档把 LangGraph 与 LangChain 区分：LangChain 提供较高层 Agent 抽象和集成，LangGraph 提供更细粒度的执行控制。LangGraph 可以独立使用，也可以在节点中复用 LangChain model 和 tool 组件。

本项目的 planner、rag、chat、explain、summarize、flashcard、quiz 和 finalizer 都是节点；路由函数根据 state 中的 intent 和执行状态选择后续节点。

