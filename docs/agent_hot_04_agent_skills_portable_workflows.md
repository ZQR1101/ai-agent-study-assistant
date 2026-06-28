# Agent Skills：可移植的工作流知识包

tags: skills, agent-skills, workflow, codex, claude, mcp, 2026

## 摘要

Agent Skill 是一组可复用、可版本化的任务说明文件，通常包含 `SKILL.md` 清单和必要的脚本、模板、参考资料。它把“怎样做某类任务”的过程知识从一次性 prompt 中抽离出来，变成可复用资产。

## Skill 与 Prompt 的区别

普通 prompt 常用于一次性任务，写完即过期。Skill 更像小型操作手册：说明何时触发、如何分阶段读取参考材料、哪些脚本可复用、哪些边界不能越过。它适合重复任务，例如生成文档、处理表格、构建 MCP server、做代码审查。

## Skill 的优点

Skill 能降低上下文噪声。模型不必每次都读完整知识库，而是在触发后按需读取 `references/` 中的材料。Skill 还利于版本治理：当流程改变，只需要更新 skill，而不是让每个用户重新学习 prompt。

## 风险

Skill 会影响模型计划、工具使用和命令执行，因此应视为有权限的代码与指令。开放的 skill 市场如果允许终端用户任意安装，可能带来 prompt injection、数据外泄和破坏性自动化风险。生产环境应由开发者审核、白名单化、绑定到具体产品流程。

## RAG 检索要点

- Skill = 可复用工作流知识包，不等于普通提示词。
- Skill 适合低频复杂任务和高频标准流程。
- Skill 的安全边界接近插件和脚本，需要审查。

## Sources

- OpenAI Skills guide: https://developers.openai.com/api/docs/guides/tools-skills
- MCP Build with Agent Skills: https://modelcontextprotocol.io/docs/develop/build-with-agent-skills
