# Agent 工具安全、权限与 Prompt Injection

tags: security, tool-permissions, prompt-injection, mcp, skills, 2026

## 摘要

Agent 的风险不只来自模型输出，还来自工具层。一个会写文件、发邮件、转账、调用内部 API 的 agent，即使回答文字看起来无害，也可能通过工具造成真实影响。

## 风险类型

工具投毒：恶意工具描述诱导模型泄露上下文或调用错误工具。

间接 prompt injection：网页、文档、邮件或工单中的恶意文本被 agent 当作指令执行。

权限过宽：读工具和写工具混在一起，低风险任务获得高风险权限。

组合攻击：单个工具看似安全，但多个工具串联后可读取敏感数据并发出外部请求。

Skill 风险：Skill 既是说明又可能包含脚本，能影响模型规划和命令执行。

## 防护策略

最小权限：按任务授权工具，不要默认暴露整个工具库。

动作分级：读操作、草稿操作、写操作、不可逆操作分开审批。

来源隔离：把外部内容标为不可信数据，禁止其覆盖系统规则。

可观察性：记录每次工具调用的参数、来源、返回值和用户确认。

沙箱：文件系统、网络、凭据和进程权限都应限制在任务需要范围内。

## Sources

- OpenAI Skills risk guidance: https://developers.openai.com/api/docs/guides/tools-skills
- MCP Architecture: https://modelcontextprotocol.io/docs/learn/architecture
- OpenAI Computer use safety notes: https://developers.openai.com/api/docs/guides/tools-computer-use
