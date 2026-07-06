# SQLAlchemy 会话历史

- source: `backend/database.py`, `backend/db_models.py`, `backend/session_store.py`, `backend/server.py`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

数据库会话历史由 `ENABLE_DB_HISTORY` 和 `DATABASE_URL` 共同决定。`database.py` 懒创建 SQLAlchemy Engine 与 sessionmaker，`get_db_session()` 用 context manager 提交或回滚事务，`init_db()` 创建当前 ORM metadata。

当前 ORM 模型包括 ChatSession、ChatMessage 和 JudgeEvaluation。ChatSession 保存 session_id、title、created_at、updated_at；ChatMessage 保存 role、content、可选 response_json 与创建时间，并关联 session；JudgeEvaluation 保存 Judge 分数、verdict、反馈和 run_id。

`/chat` 在数据库历史启用时先创建或取得 session，读取最近消息作为 history；执行完成后分别保存 user 与 assistant 消息，assistant 还可保存完整 ChatResponse 快照。数据库加载或保存异常会进入 `runtime_info.db_history_error`，不会阻断主聊天结果。

`GET /sessions` 返回最近会话，`GET /sessions/{session_id}/messages` 返回消息。该数据库历史与 RunRepository 分工不同：前者面向会话恢复，后者面向单次执行的计划、工具、审计和产物。

