# 静态 Prompt、动态 Prompt 与缓存友好结构

tags: prompt, prompt-caching, static-context, dynamic-context, 2026

## 摘要

生产 agent 的 prompt 不应被看成一整块文本，而应拆成静态前缀和动态尾部。静态部分包括系统规则、工具说明、输出格式、少量稳定示例；动态部分包括用户输入、检索结果、工具观察、当前任务状态。

## 静态内容

静态内容应放在 prompt 前部，并尽量保持字节级稳定。OpenAI 的 prompt caching 对相同前缀更友好，长系统提示、工具定义和固定示例如果频繁变化，会破坏缓存命中。

## 动态内容

动态内容应放在后部，并显式标注来源和时间。检索片段、用户偏好、当前 UI 截图、失败日志等都属于动态内容。动态内容的目标不是越多越好，而是刚好覆盖当前决策需要。

## 设计模式

推荐结构：

1. 固定身份与安全边界。
2. 固定任务协议和输出 schema。
3. 固定工具使用规则。
4. 稳定 few-shot 示例。
5. 当前用户请求。
6. 当前检索结果和工具观察。
7. 当前停止条件。

## 常见错误

把时间戳、随机 ID、临时检索结果插在系统提示最前面，会降低缓存收益。把所有历史消息都作为动态内容追加，会增加成本并污染上下文。把业务规则写在用户消息中，会使规则更难治理。

## Sources

- OpenAI Prompt caching guide: https://developers.openai.com/api/docs/guides/prompt-caching
- OpenAI Prompting guide index: https://developers.openai.com/api/docs/guides/agents
