# LLM-as-Judge 实现说明

- source: `backend/judge_service.py`, `backend/schemas.py`, `backend/session_store.py`, `backend/server.py`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

Judge 默认由 `ENABLE_LLM_JUDGE` 控制。启用后，`/chat` 在主回答完成后调用 `judge_answer()`，输入问题、回答、sources、trace、runtime_info 和模型名。Judge 失败只写入 runtime_info，不中断主回答。

评估 prompt 要求 JSON 输出，维度包括 correctness、relevance、completeness、clarity 和 citation_quality。服务端会提取 JSON、归一化分数、限制扣分结构，并根据 overall_score 与 citation_quality 计算 verdict。解析异常抛出 `JudgeEvaluationError`。

如果配置了数据库且 `ENABLE_JUDGE_PERSISTENCE=true`，Judge 结果保存到 JudgeEvaluation 表，并带 session_id、run_id、问题、回答、分数、verdict 和原始 JSON。`GET /judge-results/recent` 查询最近结果，反馈接口可记录人工对 Judge 结果的评价。

Judge 是启发式质量信号，不是人工金标准。离线检索评测默认不调用 Judge，避免 API 消耗和把生成质量混入来源排序指标。

