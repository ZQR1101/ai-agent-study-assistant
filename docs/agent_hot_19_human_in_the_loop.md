# Human-in-the-loop：审批、纠偏与证据

tags: human-in-the-loop, approvals, checkpoints, evidence, safety, 2026

## 摘要

高质量 agent 不是完全无人值守，而是在正确位置让人介入。人类不应反复替 agent 做低级验证，但应在高风险、模糊目标和不可逆操作前提供判断。

## 介入点

任务开始：澄清目标、范围、成功标准和禁止事项。

计划阶段：确认多文件改动、迁移、删除、外部写入等方案。

高风险工具前：发送邮件、付款、删除文件、修改生产数据、发布内容。

验证阶段：审查证据，而不是只听 agent 声称“完成了”。

失败循环中：连续失败后重置上下文、缩小范围或改变方法。

## 证据优先

Agent 完成任务时应展示证据：测试输出、构建结果、截图、diff 摘要、来源链接、工具返回值。证据比自然语言保证更可靠，也更方便异步审查。

## Checkpoint 与回滚

长期 agent 应在关键动作前创建 checkpoint。代码任务可依赖 git diff、工作树隔离和测试；UI 或文档任务可依赖截图、版本备份和导出文件；业务动作应依赖审批日志。

## Sources

- Anthropic Claude Code best practices: https://code.claude.com/docs/en/best-practices
- OpenAI Computer use guide: https://developers.openai.com/api/docs/guides/tools-computer-use
