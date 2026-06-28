# Agentic Coding Manifest：AGENTS.md、CLAUDE.md 与项目规则

tags: coding-agent, agents-md, manifest, project-context, procedural-memory, 2026

## 摘要

编码 agent 需要稳定的项目规则：如何运行测试、如何启动服务、代码风格、架构边界、不要碰哪些文件、提交规范、review 偏好。AGENTS.md、CLAUDE.md、规则文件和 repo-local skill 都是在给 agent 提供程序性记忆。

## 好的 Manifest 包含什么

项目地图：关键目录、模块边界、入口文件。

常用命令：安装、测试、lint、类型检查、构建、启动开发服务器。

编码规则：风格、命名、错误处理、日志、国际化、性能约束。

安全规则：密钥位置、禁止外传的数据、危险命令、审批要求。

验证规则：完成任务前必须运行哪些检查，UI 任务是否需要截图。

协作规则：是否允许提交、PR 标题格式、变更摘要格式。

## 为什么适合 RAG

Manifest 是高价值检索文档，因为它回答的是“这个项目里应该怎么做”。在 RAG 中，它应比通用教程优先级更高，但也要避免过长。把稳定规则写入 manifest，把一次性任务信息留在当前 prompt。

## 常见错误

把所有历史决策都塞进一个巨大文件，导致 agent 每次加载过多上下文；只写抽象价值观，不写可执行命令；规则过期但没有版本或日期。

## Sources

- OpenAI AGENTS.md guide: https://developers.openai.com/codex/guides/agents-md
- Anthropic Claude Code best practices: https://code.claude.com/docs/en/best-practices
- Agentic coding manifest empirical study: https://arxiv.org/abs/2509.14744
