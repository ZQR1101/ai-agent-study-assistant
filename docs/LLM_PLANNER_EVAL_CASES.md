# LLM Planner 评估用例集

## 1. Evaluation Purpose

本用例集用于评估 LangGraph LLM Planner 的规划质量，而不只是验证它是否能输出合法 JSON。

重点关注：

- 是否选择正确工具。
- 是否漏掉用户明确要求的工具。
- 是否多调用不必要工具。
- 是否正确处理否定表达。
- 是否正确处理复合任务。
- 是否正确处理 RAG、flashcard、quiz、summarize 等任务类型。
- 是否触发 fallback。

## 2. Expected Tool Order

推荐工具顺序：

```text
rag -> summarize / explain / chat -> flashcard -> quiz
```

说明：

- 需要知识库、资料、文档或上传内容时，`rag` 应优先执行。
- `summarize` 和 `explain` 通常二选一，除非用户明确要求同时总结和解释。
- `chat` 适合直接闲聊或没有明确学习工具需求的请求。
- `flashcard` 应在主要内容生成后执行。
- `quiz` 通常最后执行。
- 不应重复调用同一个工具。

## 3. Evaluation Table

| ID | User Message | Expected Tools | Should Use RAG | Should Generate Flashcards | Should Generate Quiz | Negation Handling | Expected Notes |
|---|---|---|---|---|---|---|---|
| LLM-01 | 什么是 RAG | `explain` | 否 | 否 | 否 | 无 | 基础概念解释，不应调用 RAG、卡片或出题工具。 |
| LLM-02 | 请总结 prompt engineering 的核心思想 | `summarize` | 否 | 否 | 否 | 无 | 总结意图应优先选择 `summarize`，不强制 explain。 |
| LLM-03 | 根据知识库解释 agentic rag | `rag -> explain` | 是 | 否 | 否 | 无 | 先检索知识库，再基于上下文解释。 |
| LLM-04 | 根据知识库解释 agentic rag，生成记忆卡片，并出 3 道题 | `rag -> explain -> flashcard -> quiz` | 是 | 是 | 是 | 无 | 典型复合任务，应覆盖 RAG、讲解、卡片和练习题。 |
| LLM-05 | 帮我生成 RAG 的记忆卡片 | `explain -> flashcard` 或 `flashcard` | 否 | 是 | 否 | 无 | 如果 flashcard 工具能独立生成卡片，可以只用 `flashcard`；若需要铺垫内容，可先 `explain`。 |
| LLM-06 | 根据刚才内容出 3 道题 | `quiz` | 否 | 否 | 是 | 无 | 应结合 history 或 previous context，避免无意义解释。 |
| LLM-07 | 请解释 RAG，不要出题 | `explain` | 否 | 否 | 否 | `quiz=false` | 必须遵守“不要出题”，不能调用 `quiz`。 |
| LLM-08 | 不要生成卡片，只解释 agentic rag | `explain` | 否 | 否 | 否 | `flashcard=false` | 必须遵守“不要生成卡片”，不能调用 `flashcard`。 |
| LLM-09 | 只根据知识库回答，不要出题 | `rag -> explain` 或 `rag` | 是 | 否 | 否 | `quiz=false` | 必须使用知识库，同时不能调用 `quiz`。 |
| LLM-10 | 请总结这段内容，并生成 3 张复习卡片 | `summarize -> flashcard` | 否 | 是 | 否 | 无 | 先总结，再生成复习卡片。 |
| LLM-11 | 根据知识库总结 prompt engineering，并生成选择题 | `rag -> summarize -> quiz` | 是 | 否 | 是 | 无 | RAG 优先，主要内容是总结，最后生成选择题。 |
| LLM-12 | 请直接聊天，不要用知识库 | `chat` 或 `explain` | 否 | 否 | 否 | `rag=false` | 必须遵守“不要用知识库”，不能调用 `rag`。 |
| LLM-13 | 根据知识库解释一个不存在的概念 | `rag -> explain` 或 `rag` fallback | 是 | 否 | 否 | 无 | 观察 RAG fallback 和回答是否诚实说明知识库未命中。 |
| LLM-14 | 把刚才那个概念做成卡片 | `flashcard` | 否 | 是 | 否 | 无 | 应结合 history，避免重新解释过多。 |
| LLM-15 | 比较传统 RAG 和 Agentic RAG，并出题 | `explain -> quiz` 或 `rag -> explain -> quiz` | 可选 | 否 | 是 | 无 | 如果没有要求知识库，RAG 可选；必须包含比较讲解和 quiz。 |

## 4. Manual Evaluation Criteria

人工评估时可以按以下标签记录结果：

- **Correct**：工具选择完全符合预期，顺序合理，没有漏调或多调。
- **Partial**：整体合理，但多调用或少调用一个工具，或者顺序略有问题但不影响主要结果。
- **Incorrect**：明显不符合用户意图，例如忽略否定表达、漏掉核心工具、调用错误工具。
- **Fallback**：LLM Planner 失败并回退到 rule planner。

## 5. Metrics To Track

后续可以统计这些指标：

- fallback rate
- tool precision
- tool recall
- unnecessary tool call rate
- missing tool rate
- negation accuracy
- average answer length
- source overlap with rule planner
- flashcard success rate
- quiz success rate

## 6. Current Decision

LLM Planner 暂不作为默认 planner。

当前策略保持不变：

```text
Rule Planner 仍是 LangGraph Runtime 默认 planner。
LLM Planner 作为可选增强路径继续评估。
```

原因是：当前已经验证 LLM Planner 的结构化输出链路可用，但规划质量还需要通过更多样本评估，尤其是复杂任务、否定表达、多轮 history 和 RAG fallback 场景。

## 7. Next Step

下一阶段可以把这些用例做成半自动评估脚本，例如：

```text
scripts/evaluate_llm_planner.py
```

脚本可以批量调用 `/chat`，收集 `planner_mode`、`planner_fallback`、`planner_error`、`plan`、`graph_path`、`tool_calls`、sources 和 flashcards，并辅助人工标注 Correct / Partial / Incorrect / Fallback。

本阶段只新增评估用例文档，不实现脚本。
