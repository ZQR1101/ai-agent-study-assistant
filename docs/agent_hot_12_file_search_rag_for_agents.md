# File Search 与 Agentic RAG

tags: rag, file-search, vector-store, retrieval, agents, 2026

## 摘要

Agentic RAG 不只是“问答前检索文档”。Agent 可以根据中间状态多次检索、改写查询、过滤 metadata、读取引用、把结果交给工具或下游子任务。

## File Search 的位置

OpenAI File Search 通过 vector store 管理上传文件，并让模型在生成回答前做语义和关键词检索。它适合客服知识库、法律文档、技术文档、企业制度、产品手册和代码文档。

## Agentic RAG 与普通 RAG

普通 RAG 多为一次检索加一次生成。Agentic RAG 会在任务中动态决定是否检索、检索什么、如何使用结果。例如研究 agent 先检索背景，再调用 web search 补充最新信息，最后对冲突来源做比较。

## 文档颗粒度

适合 RAG 的文档应主题单一、标题清楚、包含同义词和上下位词、避免大量重复。每篇文档最好回答一个知识点，例如“MCP tools 与 resources 区别”，而不是把所有 agent 知识堆进一个大文件。

## 质量控制

RAG 需要评测召回率和答案忠实性。常见问题包括：文档重复导致召回噪声，旧文档覆盖新事实，片段缺少来源，metadata 不足导致无法按时间或领域过滤。

## Sources

- OpenAI File search guide: https://developers.openai.com/api/docs/guides/tools-file-search
- OpenAI New tools for building agents: https://openai.com/index/new-tools-for-building-agents/
