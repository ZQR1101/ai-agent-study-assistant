# MCP 部署模式：stdio、远程 HTTP 与 MCPB

tags: mcp, deployment, stdio, streamable-http, mcpb, 2026

## 摘要

MCP server 的部署方式会直接影响安装体验、权限边界、认证方式和可维护性。常见路径包括本地 stdio、远程 Streamable HTTP、MCP Apps、MCP Bundles（MCPB）。

## 本地 stdio

本地 stdio 适合个人原型、读取本机文件、调用本地命令或访问 localhost 服务。优点是延迟低、实现简单；缺点是分发复杂，用户需要安装运行时，安全边界也更依赖 host 的权限控制。

## 远程 Streamable HTTP

远程 HTTP 更适合云 API、企业服务和多人共享工具。它可以集中部署，统一升级，支持 OAuth、API key、bearer token 等常见认证。对 SaaS 型 MCP server，远程部署通常更容易治理。

## MCP Apps

MCP Apps 用于把交互式 UI 放进 AI 客户端，例如表单、选择器、图表和仪表盘。当纯文本结果或简单 elicitation 不够用时，MCP Apps 可以让用户在对话里完成结构化交互。

## MCPB

MCPB 适合“必须碰用户机器”的场景，例如读取本地文件、驱动桌面软件、访问本地开发环境。它把 server 与运行时打包，降低普通用户安装 Node 或 Python 的门槛。

## 选择准则

云 API 优先远程 HTTP；本地原型优先 stdio；需要复杂 UI 用 MCP Apps；需要分发本地 server 用 MCPB。部署模式不是技术偏好，而是由数据位置、认证边界、用户规模和操作风险决定。

## Sources

- MCP Build with Agent Skills: https://modelcontextprotocol.io/docs/develop/build-with-agent-skills
- MCP Architecture: https://modelcontextprotocol.io/docs/learn/architecture
