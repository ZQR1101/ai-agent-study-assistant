# RunRepository 运行记录

- source: `backend/run_repository.py`, `backend/run_metadata.py`, `backend/server.py`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

一次 `/chat` 请求对应一个 `Run`。Run 是 Pydantic 模型，包含 id、status、request、session_id、plan、tools、audit、artifacts、output、error、时间戳和 version。

`RunRepository` 不是 SQLAlchemy Repository。它使用本地 JSON 文件持久化，每个 Run 一个文件，默认目录是 `data/runs/`，可通过 `RUNS_DIR` 调整。写入先落临时文件再原子替换，进程内用 `RLock` 保护。

主要操作包括 create、update、finish、get、list、append_audit 和 soft delete。update 会合并 metadata、artifacts、output 字典并递增 version。delete 不移除文件，而是写入 `deleted` tombstone、deleted_at 和审计事件；列表默认隐藏已删除 Run。

`GET /runs` 和 `GET /runs/{run_id}` 提供读取。Run 聚合执行事实；JudgeEvaluation 独立存储，只通过 run_id 关联，不写进 Run 的核心数据模型。

