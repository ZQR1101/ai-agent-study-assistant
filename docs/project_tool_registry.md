# Tool Registry 实现说明

- source: `backend/tool_registry.py`, `backend/tools.py`, `backend/tool_actions.py`, `backend/server.py`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

`ToolSpec` 定义工具名、描述、执行函数、分类、是否需要确认和是否对 Agent 可见。分类为 `read`、`write`、`dangerous`。危险工具如果没有设置 `requires_confirmation=True`，注册时会直接报错。

所有工具统一经过 `ToolRegistry.execute()`。执行入口负责别名解析、未知工具审计、危险操作确认、耗时记录和结果审计。AuditLog 以 JSONL 追加写入，敏感参数名会被脱敏；如果事件带 run_id，还会同步追加到对应 Run。

危险操作采用两阶段确认：首次调用签发 approval request；独立批准后得到一次性 token；再次调用时 token 必须与工具名、完整参数摘要和请求方身份一致，并且在有效期内。服务端还要求 requester key 和 approver key 不同。

当前 Agent 可见的核心工具包括 `chat`、`rag_search` 和 `study`。`study` 通过 operation 组合解释、总结、题目和卡片。保存数据、删除文件、重建/清空索引、运行受限代码和删除 Run 等工具按 write 或 dangerous 分类。

