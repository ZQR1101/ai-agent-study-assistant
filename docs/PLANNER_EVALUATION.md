# Planner 评估报告

## 1. 评估目标

本次评估用于对比 AI Study Assistant 当前的三种运行方式：

- Legacy Agent
- LangGraph Rule Planner
- LangGraph LLM Planner

重点观察以下问题：

- Planner 是否触发 fallback。
- 生成的 plan 是否合理。
- `graph_path` 是否清晰。
- answer 是否存在重复堆叠。
- sources 是否一致。
- flashcards 是否正常返回。
- LLM Planner 是否值得继续投入和扩大测试。

## 2. 测试设置

本次评估使用本地 `/chat` 接口，并通过以下脚本运行：

```bash
python scripts/compare_runtimes.py --include-llm-planner
```

三种运行方式分别是：

- Legacy Agent：`use_langgraph=false`，`planner_mode=rule`
- LangGraph Rule Planner：`use_langgraph=true`，`planner_mode=rule`
- LangGraph LLM Planner：`use_langgraph=true`，`planner_mode=llm`

共享请求参数：

- `mode=auto`
- `use_agent=true`
- `model=mimo-v2.5`
- `temperature=0.3`
- `top_k=3`

如果使用 `--save` 保存原始评估结果，输出文件应位于已忽略的 `outputs/` 目录，不应提交到仓库。

## 3. 关键发现

在当前样本中，LangGraph LLM Planner 没有触发 fallback：

```text
planner_fallback=false
```

这说明当前的结构化规划链路可以正常工作：

```text
JSON-only prompt -> AgentPlan schema -> JSON 提取 -> Pydantic 校验
```

也就是说，LLM Planner 在本次样本中成功输出了合法 JSON，并且通过了 schema 校验，LangGraph Runtime 能够继续执行该 plan。

但需要注意：没有触发 fallback 只能说明结构化输出在当前样本中是稳定的，不代表 LLM Planner 的规划质量一定优于 rule planner，也不代表它已经适合作为默认 planner。

## 4. 评估记录

后续需要继续观察以下维度：

- LLM Planner 是否能生成比 rule planner 更完整的 plan。
- LLM Planner 是否会多调用不必要的工具。
- LLM Planner 是否遵守预期工具顺序：`rag -> summarize/explain/chat -> flashcard -> quiz`。
- LLM Planner 是否能正确选择 `rag`、`explain`、`flashcard`、`quiz`。
- LLM Planner 是否比 rule planner 更适合复杂中文自然语言表达。
- 在更多样本中，LLM Planner 是否会触发 fallback。
- 当同时生成 flashcards 和 quiz 时，最终 answer 是否仍能避免重复堆叠。
- LLM Planner 返回的 sources 是否与 rule planner 和 Legacy Agent 保持基本一致。

## 5. 当前结论

暂时不把 `planner_mode` 默认改成 `llm`。

当前仍然保持：

```text
planner_mode=rule
```

作为 LangGraph Runtime 的默认 planner。

LLM Planner 继续作为可选增强路径保留，用于复杂任务测试、本地评估和后续迁移准备。

这个结论比较谨慎，原因是：当前结果只能说明 LLM Planner 的结构化输出链路可用，但样本数量还不够，暂时无法证明它在规划质量、稳定性、错误率和用户体验上已经全面优于 rule planner。

## 6. 下一步

建议继续做以下工作：

- 增加更多真实评估 case。
- 覆盖 failed RAG、flashcard-only、quiz-only、summarize-only、多轮 history 等场景。
- 记录 LLM Planner 在多次测试中的 fallback 率。
- 优化 LLM Planner prompt，加入更多否定表达和复杂组合任务示例。
- 对比 rule planner 和 LLM planner 的不必要工具调用率。
- 在更大样本评估后，再判断是否把 `planner_mode=llm` 设为推荐模式。
