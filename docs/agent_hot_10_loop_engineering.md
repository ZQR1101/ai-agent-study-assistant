# Loop Engineering：从写 Prompt 到设计闭环

tags: loop-engineering, autonomous-agents, verification, codex, claude-code, 2026

## 摘要

Loop engineering 指把一次性提示变成可持续执行的工作闭环。核心不是“让模型多想”，而是明确输入、动作、反馈、验证、停止条件和人工接管点。

## 一个有效 loop 的结构

目标：任务完成后世界应该处于什么状态。

上下文：模型可读到哪些文件、知识库、记忆和外部信号。

动作：模型可以调用哪些工具，每个工具的权限和副作用是什么。

反馈：动作后返回什么可观测证据，例如测试输出、截图、API 响应、diff。

验证：什么检查可以证明目标达成。

停止：通过、失败、超时、超预算或需要人类决策时如何退出。

## 编码 agent 的典型闭环

探索代码库，形成计划，修改代码，运行测试，读取失败，修复，再次运行测试，最后总结证据。Claude Code 文档强调给 agent 一个可运行检查；OpenAI Codex 和 Agents SDK 生态也强调 trace、eval 和工具化验证。

## 与 Prompt Engineering 的区别

Prompt engineering 关注单次模型输入的表达；loop engineering 关注多次模型调用之间如何传递状态、如何纠错、如何证明完成。Prompt 是 loop 的一部分，但不是整个系统。

## 设计提醒

不要给 agent 无限自主权。每个 loop 都应有最大迭代、预算、沙箱、审批和可恢复 checkpoint。越自动化，越需要更强的观察和回滚。

## Sources

- Anthropic Claude Code best practices: https://code.claude.com/docs/en/best-practices
- Anthropic Building Effective AI Agents: https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Computer use guide: https://developers.openai.com/api/docs/guides/tools-computer-use
