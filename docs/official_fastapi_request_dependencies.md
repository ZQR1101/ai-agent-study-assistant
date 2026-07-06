# FastAPI Request Body 与依赖注入

- source: `https://fastapi.tiangolo.com/tutorial/body/`
- source: `https://fastapi.tiangolo.com/tutorial/dependencies/`
- curated_at: `2026-07-06 Asia/Shanghai`
- publisher: `FastAPI official documentation`

FastAPI 使用 Pydantic model 声明 request body。路径操作函数参数如果标注为 Pydantic model，FastAPI 会从 JSON body 读取数据、执行类型转换和校验，并把 JSON Schema 写入 OpenAPI。没有默认值的字段是必填字段；显式默认值或 `None` 可表达可选字段。

依赖注入通过 `Depends` 声明。路径操作可以依赖普通函数或异步函数，FastAPI 负责解析依赖树并注入结果。依赖适合承载共享逻辑、数据库会话、安全检查和权限要求；依赖及子依赖的参数也会进入 OpenAPI schema。

与本项目的对应关系：`ChatRequest`、`DebugRagRequest` 等请求结构由 Pydantic 校验；数据库 session、工具权限和上传安全虽然不都写成 `Depends`，但遵循把请求校验与业务执行分离的思路。

