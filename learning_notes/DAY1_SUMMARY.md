## 新增 /debug-rag 接口

今天新增了 `POST /debug-rag` 接口，用于直接查看 RAG 检索阶段返回的 chunks。

通过对比 `/debug-rag` 和 `/rag`，我理解了 RAG 的两个阶段：

- `/debug-rag`：只做 Retrieval，返回 source、text、score
- `/rag`：先 Retrieval，再把检索内容交给 LLM 做 Generation

这个接口可以帮助排查 RAG 回答不准确的问题：如果 debug 检索结果不相关，说明问题在检索阶段；如果检索结果相关但回答不好，说明问题在生成阶段。
