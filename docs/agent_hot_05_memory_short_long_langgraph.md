# Agent 记忆：短期、长期、语义、情节与程序性记忆

tags: memory, langgraph, semantic-memory, episodic-memory, procedural-memory, 2026

## 摘要

Agent 记忆不是“把所有聊天记录塞进上下文”。更稳妥的做法是区分短期记忆和长期记忆，并继续拆分长期记忆的类型：语义记忆、情节记忆、程序性记忆。

## 短期记忆

短期记忆通常是 thread-scoped 的会话状态，包括消息历史、当前任务状态、上传文件、临时检索结果、正在生成的 artifact。它的生命周期与一个会话或任务线程绑定，常通过 checkpoint 持久化，便于恢复。

## 长期记忆

长期记忆跨会话存在，可以被不同线程调用。它常按 user、org、project、application 等 namespace 组织，既可做语义搜索，也可做结构化读取。

## 三类长期记忆

语义记忆保存事实，例如用户偏好、项目规范、业务实体关系。它适合个性化和知识补全。

情节记忆保存经历，例如过去某次任务的步骤、失败原因、成功轨迹。它常以 few-shot 示例形式影响未来任务。

程序性记忆保存“如何做事”的规则，例如系统提示、开发规范、工作流模板。Skill、AGENTS.md、CLAUDE.md 和可更新的系统 prompt 都可视为程序性记忆。

## 热路径与后台写入

热路径写记忆能立即生效，但增加延迟，也可能让 agent 同时承担任务执行和记忆整理。后台写入能降低主流程延迟，适合定期整合对话和业务事件，但需要处理新记忆何时可见的问题。

## Sources

- LangGraph Memory overview: https://docs.langchain.com/oss/python/concepts/memory
