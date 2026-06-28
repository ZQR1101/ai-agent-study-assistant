# MCP 架构与核心原语

tags: mcp, tools, resources, prompts, json-rpc, 2026

## 摘要

MCP（Model Context Protocol）是连接 AI 应用与外部系统的开放协议。它把“模型如何拿到工具、数据和提示模板”标准化，避免每个应用都为每个数据源写一套私有集成。

## 架构

MCP 采用 host-client-server 模式。Host 是 Claude、ChatGPT、VS Code、Cursor 这类 AI 应用；client 是 host 内部为每个 MCP server 建立的连接；server 暴露工具、资源和提示模板。

MCP 分为两层：数据层和传输层。数据层基于 JSON-RPC 2.0，定义初始化、能力协商、工具发现、工具调用、资源读取、提示获取、通知等消息。传输层负责本地 stdio 或远程 HTTP/SSE 等连接方式。

## 三类服务端原语

Tools 是可执行动作，例如查询数据库、调用 API、写文件、提交工单。它们需要清晰的名称、描述和 JSON Schema 输入。

Resources 是上下文数据，例如文件内容、日志、数据库 schema、API 响应、知识库片段。资源偏“读”，工具偏“做”。

Prompts 是可复用的提示模板或工作流模板。它们帮助 server 向 host 暴露领域内的最佳使用方式，而不是只暴露裸 API。

## 客户端能力

MCP 也允许 server 请求 client 提供能力，例如 sampling（让 host 的模型完成一次推理）和 elicitation（向用户请求补充信息或确认）。这使 server 不必自己持有模型密钥，也能参与更复杂的交互。

## Sources

- MCP Introduction: https://modelcontextprotocol.io/docs/getting-started/intro
- MCP Architecture: https://modelcontextprotocol.io/docs/learn/architecture
- MCP 2025-06-18 specification: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
