# A2A：Agent 之间的互操作协议

tags: a2a, interoperability, multi-agent, google, mcp, 2026

## 摘要

A2A（Agent2Agent）是 Google 在 2025 年发布的开放协议，目标是让不同厂商、框架和平台中的 agent 能发现彼此、交换消息、协作完成任务。它与 MCP 互补：MCP 解决 agent 连接工具和数据，A2A 解决 agent 连接 agent。

## 核心机制

Agent Card：远程 agent 通过 JSON 描述自己的能力、入口、认证方式和交互模式，便于客户端 agent 发现和选择。

Task：A2A 以任务完成为中心，支持短任务和长任务。长任务可以持续提供状态更新、通知和中间结果。

Message 与 Parts：agent 之间交换结构化消息，parts 可以承载文本、文件、图像、表单、视频等不同模态内容。

安全：A2A 设计时考虑企业认证与授权，建立在 HTTP、SSE、JSON-RPC 等常见标准之上。

## 什么时候用 A2A

当一个 agent 需要把任务委派给另一个专门 agent 时，A2A 比“把对方包装成工具”更自然。例子包括招聘 agent 调用背景调查 agent，旅行 agent 调用支付 agent，客服 agent 调用物流 agent。

## 与多 agent 框架的关系

A2A 是协议，不是编排框架。框架决定内部如何计划、路由和调度；A2A 提供跨系统协作的通信层。

## Sources

- Google Developers Blog, Announcing Agent2Agent Protocol: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
- A2A Protocol specification: https://a2a-protocol.org/latest/specification/
