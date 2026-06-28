# 上下文压缩与会话续航

tags: context, compaction, summarization, long-running-agents, 2026

## 摘要

长会话 agent 的主要瓶颈不是模型“不聪明”，而是上下文窗口逐渐被旧文件、日志、失败尝试和无关对话占满。压缩（compaction）通过总结关键状态、删除冗余内容、保留必要证据，让 agent 在长期任务中维持可用上下文。

## 压缩保留什么

压缩应保留目标、约束、已修改文件、关键决策、失败尝试、验证命令、外部事实来源、未完成事项。对于代码任务，还应保留测试命令、错误输出摘要、架构边界和用户明确要求。

## 压缩丢弃什么

可丢弃内容包括重复日志、完整依赖安装输出、已废弃方案、长文件全文、无关闲聊、重复检索片段。注意：丢弃不是遗忘事实，而是把事实折叠成更小的状态表示。

## OpenAI 与 Claude Code 的共同信号

OpenAI Responses API 提供 server-side compaction，把之前状态以更少 token 带入后续上下文。Claude Code 文档强调在任务之间清空上下文、在接近上限时压缩、用子 agent 分离调查上下文。

## RAG 检索要点

- 压缩是上下文管理，不是摘要写作。
- 好的压缩面向后续决策，保留可验证状态。
- 长期 agent 需要压缩策略、记忆策略和检索策略共同工作。

## Sources

- OpenAI Compaction guide: https://developers.openai.com/api/docs/guides/compaction
- Anthropic Claude Code best practices: https://code.claude.com/docs/en/best-practices
