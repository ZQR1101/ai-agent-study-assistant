# Demo 截图指南

这份文档用于整理 AI Study Assistant 的 GitHub 展示截图素材。截图文件建议统一放在 `images/` 目录下。

## 1. 建议截图文件名

建议后续补充以下截图：

```text
images/dashboard.png
images/langgraph-runtime-info.png
images/agent-plan-trace.png
images/flashcards.png
images/knowledge-files.png
images/runtime-settings.png
```

当前 README 已引用并确认存在：

```text
images/cover.png
images/architecture.png
```

## 2. 截图内容说明

### dashboard.png

展示前端 Dashboard 主界面，包括：

- 左侧导航
- 当前学习会话
- 运行设置
- 输入框

### langgraph-runtime-info.png

展示 LangGraph Runtime 可观测性，包括：

- 勾选 LangGraph Workflow
- Planner Mode
- runtime_info 面板
- graph_path
- tool_calls

### agent-plan-trace.png

展示 Agent 执行信息，包括：

- Agent Plan
- Trace / 执行路径
- sources

### flashcards.png

展示卡片学习能力，包括：

- 结构化 flashcards
- 卡片正面 / 背面
- 翻转效果
- 下载按钮

### knowledge-files.png

展示知识库文件能力，包括：

- Knowledge Files 面板
- docs 文件列表
- 文件预览或文件信息

### runtime-settings.png

展示运行参数设置，包括：

- RAG 开关
- LangGraph 开关
- Planner Mode
- model / temperature / top_k

## 3. 推荐截图流程

截图前建议先启动后端和前端：

```bash
uvicorn backend.server:app --reload
npm run dev
```

然后在前端发送测试问题：

```text
根据知识库解释 agentic rag，生成记忆卡片，并出 3 道题
```

建议使用以下运行设置：

- 开启 RAG。
- 勾选 LangGraph Workflow。
- Planner Mode 选择 `rule` 或 `llm`。

对应请求字段可以理解为：

```json
{
  "use_langgraph": true,
  "planner_mode": "rule",
  "use_rag": true
}
```

## 4. README 引用规则

- 只在 README 中引用已经存在的图片文件。
- 不要引用尚未生成的截图路径。
- 新截图加入 `images/` 后，再更新 README 的 Demo Screenshots 区域。
- 不要提交 `frontend/dist/`、`node_modules/`、`outputs/` 或日志文件。
