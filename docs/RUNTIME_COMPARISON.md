# Runtime Comparison Report

## 1. Evaluation Goal

This evaluation compares the current legacy Planner + Executor Agent and the optional LangGraph Runtime under the same `/chat` requests. The goal is to understand differences in routing clarity, output shape, source reuse, flashcard handling, trace readability, and readiness for default runtime usage.

## 2. Test Setup

- Backend: local FastAPI server at `http://127.0.0.1:8000`
- Endpoint: `POST /chat`
- Tool: `scripts/compare_runtimes.py`
- Legacy request: `use_langgraph=false`
- LangGraph request: `use_langgraph=true`
- Shared request settings:
  - `mode=auto`
  - `model=mimo-v2.5`
  - `temperature=0.3`
  - `use_agent=true`
  - `top_k=3`
- Raw comparison JSON files were saved under ignored `outputs/` for local inspection only.

## 3. Test Cases

1. **Agentic RAG compound task**
   - Message: `根据知识库解释 agentic rag，生成记忆卡片，并出 3 道题`
   - Purpose: compare multi-step planning, RAG use, flashcard generation, quiz generation, and final answer composition.

2. **Plain concept explanation**
   - Message: `什么是 RAG`
   - Purpose: compare simple non-RAG explanation behavior.

3. **Knowledge-base RAG answer**
   - Message: `根据知识库解释 prompt engineering`
   - Purpose: compare RAG source retrieval and grounded explanation behavior.

4. **Summarization task**
   - Message: `请总结 prompt engineering 的核心思想`
   - Purpose: compare summarize routing and RAG context usage.

## 4. Results Summary

| Case | Legacy Answer Length | LangGraph Answer Length | Sources Overlap | Legacy Plan | LangGraph Path | Flashcards | Observation |
|---|---:|---:|---:|---|---|---:|---|
| Agentic RAG compound task | 1544 | 2237 | 1 | `rag -> flashcard` | `planner -> rag -> explain -> flashcard -> quiz -> finalizer` | 4 / 4 | LangGraph executed the full intended path, including explain and quiz. Legacy skipped quiz in the returned plan. |
| Plain concept explanation | 926 | 516 | 0 | `explain` | `planner -> explain -> finalizer` | 0 / 0 | Both routed correctly. LangGraph was shorter and easier to inspect structurally. |
| Knowledge-base RAG answer | 1926 | 639 | 1 | `rag -> explain` | `planner -> rag -> explain -> finalizer` | 0 / 0 | Both used the same source count and overlap. LangGraph final answer was much more concise. |
| Summarization task | 1442 | 473 | 1 | `rag -> summarize` | `planner -> rag -> summarize -> finalizer` | 0 / 0 | Both routed correctly. LangGraph path and tool calls were explicit. |

Flashcards are shown as `legacy / langgraph`.

## 5. Observations

- **Routing clarity:** LangGraph exposes a clear `graph_path`, such as `planner -> rag -> explain -> flashcard -> quiz -> finalizer`. This is easier to compare and debug than inferring legacy behavior from grouped trace blocks.
- **Tool-call observability:** LangGraph returns structured `tool_calls`, including the tools executed and whether context was used. This makes runtime-level debugging more machine-readable.
- **Source consistency:** In the RAG cases, sources overlap was `1`, indicating both runtimes generally retrieved the same source set for these small samples.
- **Answer length:** LangGraph produced shorter answers in three of four cases. For the compound task, LangGraph produced a longer response because it completed the requested explain + flashcard + quiz path.
- **Plan completeness:** In the compound task, legacy returned `rag -> flashcard`, while LangGraph returned `rag -> explain -> flashcard -> quiz`. LangGraph better matched the user request in this case.
- **Trace readability:** Legacy trace remains useful for operational details, but it is more verbose and less structured. LangGraph trace plus `runtime_info` gives both human-readable and machine-readable views.
- **Answer repetition:** The comparison script flagged legacy as having repeated flashcard markdown in the compound case. It also flagged LangGraph heuristically because the final answer intentionally contains a short "memory card" section header/prompt. Manual interpretation: LangGraph finalizer appears to avoid dumping full flashcard markdown into the answer and instead relies on structured `flashcards`.
- **Finalizer behavior:** LangGraph `finalizer_used` was true in all LangGraph cases. The finalizer is useful for composing a stable final answer and keeping structured flashcards separate from answer text.

## 6. Current Decision

Do not make LangGraph the default main runtime yet.

Keep the legacy Agent as the default stable path. Continue using LangGraph Runtime as an optional execution path for complex tasks, debugging, runtime comparison, and future migration work.

This is a cautious decision because:

- The legacy Agent is already the stable v0.1 path.
- The current sample size is small.
- LangGraph observability is better, but runtime behavior still depends on rule-based intent detection.
- More regression and quality comparison cases are needed before changing the default user-facing runtime.

## 7. Next Steps

- Add more comparison cases, including chat-only, failed RAG, flashcard-only, quiz-only, and multi-turn history scenarios.
- Improve LangGraph intent detection for Chinese task phrasing and mixed requests.
- Consider upgrading the LangGraph planner from rule-based detection to a schema-validated LLM planner.
- Make runtime comparison metrics more structured, including latency, fallback usage, token/cost estimates if available, and exact source IDs.
- Add manual review notes for answer quality, not just structural metrics.
- Re-evaluate whether LangGraph should become the default runtime after more test coverage and stability evidence.
