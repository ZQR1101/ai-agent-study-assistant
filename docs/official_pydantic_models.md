# Pydantic Models

- source: `https://docs.pydantic.dev/latest/concepts/models/`
- curated_at: `2026-07-06 Asia/Shanghai`
- publisher: `Pydantic official documentation`

Pydantic model 是继承 `BaseModel` 的类，字段由 Python 类型注解定义。它的目标是保证解析和校验后的模型实例符合声明的类型与约束；输入数据可能发生类型转换，因此严格校验需要显式使用 strict mode 或相关配置。

常用方法包括 `model_validate()`、`model_validate_json()`、`model_dump()`、`model_dump_json()`、`model_copy()` 和 `model_json_schema()`。`model_construct()` 会跳过校验，只适合已经可信的数据。extra 字段可以配置为 ignore、forbid 或 allow。

嵌套 model 可表达层级结构。无法转换的数据会产生 `ValidationError`，错误包含字段位置和原因。JSON Schema 可由 model 生成，FastAPI 会将其用于 OpenAPI 与交互式文档。

本项目用 Pydantic 定义 ChatRequest/ChatResponse、AgentPlan、Flashcard、JudgeEvaluationResult 和 Run。LLM planner 与 Judge 的 JSON 输出都需要先解析，再通过 schema 或归一化逻辑进入主流程。

