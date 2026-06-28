# Agent 循环与 ACI

tags: ai-agent, loop, tool-use, agent-computer-interface, 2026

## 摘要

现代 AI agent 的核心通常不是复杂框架，而是一个持续循环：接收目标，读取上下文，选择工具，执行动作，观察环境反馈，再决定下一步。Anthropic 在工程实践中把 agent 描述为“LLM 使用工具并依据环境反馈循环”的系统；OpenAI 的 computer use 文档也把 UI agent 拆成“模型给出动作、宿主执行、截图回传、重复直到停止”的循环。

## 关键概念

Agent 与普通聊天机器人的差别在于环境反馈。聊天机器人主要生成回答；agent 会把工具结果、测试结果、文件差异、网页截图、数据库返回值等作为下一步推理的输入。

ACI（Agent-Computer Interface）是 agent 面对计算机环境时的接口设计。好的 ACI 不只是“给模型更多工具”，而是让工具名、描述、输入 schema、错误信息、权限边界都足够清楚。工具描述越像稳定 API 文档，模型越容易正确规划。

停止条件很关键。常见停止条件包括：测试通过、目标状态达成、达到最大轮数、遇到需人工确认的高风险动作、预算或时间耗尽。没有停止条件的 agent 容易进入重复搜索、重复修复或无意义重试。

## RAG 检索要点

- 问“agent loop 是什么”时，优先回答观察-行动-反馈循环。
- 问“为什么 agent 不可靠”时，强调错误会在多步中累积，必须加入验证和沙箱。
- 问“ACI 怎么设计”时，强调工具 schema、错误语义、权限、可观察性。

## 实践建议

从一个最小闭环开始：目标、工具、观察、验证。先用单 agent 跑通，再加入多 agent、记忆、计划器或复杂编排。复杂度必须由评测结果证明，而不是因为框架看起来高级。

## Sources

- Anthropic, Building Effective AI Agents: https://www.anthropic.com/engineering/building-effective-agents
- OpenAI, Computer use tool guide: https://developers.openai.com/api/docs/guides/tools-computer-use
