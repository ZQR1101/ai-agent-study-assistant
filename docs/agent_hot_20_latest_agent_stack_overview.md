# 2026 Agent 技术栈速览

tags: agent-stack, mcp, skills, memory, evals, a2a, rag, 2026

## 摘要

当下热门 agent 技术栈正在从“一个聊天框加几个工具”演进为分层系统：模型、上下文、工具协议、技能、记忆、评测、沙箱、互操作协议和人类审批共同组成生产 agent。

## 分层视图

模型层：通用模型、推理模型、代码模型、视觉/电脑使用模型、开放权重模型。

上下文层：静态系统提示、动态检索结果、会话状态、压缩摘要、prompt cache 友好的前缀结构。

工具层：function calling、MCP tools/resources/prompts、computer use、shell、OCR、文件检索、浏览器。

技能层：Agent Skills、项目 manifest、领域工作流模板、可复用脚本。

记忆层：短期 checkpoint、长期 namespace store、语义记忆、情节记忆、程序性记忆、时间图谱。

编排层：单 agent loop、多 agent handoff、subagent 调查、A2A 跨系统协作。

安全层：沙箱、最小权限、审批、prompt injection 防护、工具审计。

评测层：任务 harness、trace、回归集、LLM judge、确定性验证、成本和延迟指标。

## 重要趋势

MCP 正在成为工具和数据连接标准；Skills 正在把工作流知识产品化；A2A 关注 agent 之间协作；长期记忆从简单向量库走向图谱和时间推理；loop engineering 让 agent 从“会回答”变成“能持续完成”。

## Sources

- OpenAI New tools for building agents: https://openai.com/index/new-tools-for-building-agents/
- MCP Introduction: https://modelcontextprotocol.io/docs/getting-started/intro
- Anthropic Building Effective AI Agents: https://www.anthropic.com/engineering/building-effective-agents
- LangGraph Memory overview: https://docs.langchain.com/oss/python/concepts/memory
- Google A2A announcement: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
