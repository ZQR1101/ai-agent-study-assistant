# 多 Agent 与 Subagent 模式

tags: multi-agent, subagents, delegation, verification, context-isolation, 2026

## 摘要

多 agent 系统不应只是“多开几个模型”。它的价值在于上下文隔离、专业化分工、并行调查、独立验证和权限拆分。

## 常见模式

研究 subagent：在独立上下文中读取大量资料，返回短报告，避免污染主上下文。

执行 agent：负责修改文件、调用工具、推进任务。

验证 agent：从新上下文审查结果，寻找边界情况、遗漏测试或安全问题。

路由 agent：根据用户意图把任务交给不同专业 agent。

对抗审查：让一个 agent 尝试反驳另一个 agent 的结论，适合高风险决策。

## 什么时候值得用

当任务需要大量探索、多个专业领域、并行处理或独立审查时，多 agent 有价值。若任务很小，单 agent 加清晰工具和测试更简单、更便宜。

## 设计注意

每个 subagent 应有明确输入、输出格式、权限和停止条件。不要让所有 agent 共享完整历史，否则会失去上下文隔离的好处。最终决策应由主流程合并证据，而不是盲目投票。

## Sources

- Anthropic Claude Code best practices: https://code.claude.com/docs/en/best-practices
- OpenAI New tools for building agents: https://openai.com/index/new-tools-for-building-agents/
- Google A2A announcement: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
