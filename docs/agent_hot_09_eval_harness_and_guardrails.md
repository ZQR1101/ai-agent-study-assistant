# Agent Harness：评测、追踪与护栏

tags: evals, harness, guardrails, traces, agents-sdk, 2026

## 摘要

Agent harness 是包住 agent 的测试与运行外壳。它不只负责调用模型，还负责记录输入输出、工具调用、状态转移、成本、延迟、错误、人工审批和评测结果。

## Harness 的组成

任务集：一组代表真实用户目标的案例，包含输入、允许工具、预期结果和通过标准。

执行器：把任务交给 agent，提供工具、环境变量、沙箱和超时限制。

观察器：记录 trace，包括模型响应、工具参数、工具结果、重试、异常、token、耗时。

判定器：可以是确定性脚本、单元测试、截图对比、结构化校验、人工标注或 LLM-as-judge。

护栏：在输入、工具调用、输出三个阶段做检查，例如阻止敏感写操作、拦截高风险请求、要求用户确认。

## 为什么 agent 更需要 harness

普通单轮 LLM 应用主要评估答案质量；agent 还要评估过程质量。一个答案正确但调用了不该调用的工具，仍然可能是失败。一个任务最后成功但重试 40 次，成本和可靠性也不可接受。

## 最小可用 harness

先做 20 个黄金任务，每个任务包含输入、可运行检查、最大轮数和失败日志保存。再增加 trace 可视化、回归评测、成本阈值和风险工具审批。

## Sources

- OpenAI Agents SDK overview: https://openai.github.io/openai-agents-python/
- OpenAI New tools for building agents: https://openai.com/index/new-tools-for-building-agents/
- OpenAI Agents SDK docs: https://developers.openai.com/api/docs/guides/agents
