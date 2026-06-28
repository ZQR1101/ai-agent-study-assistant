# Computer Use 与 UI Agent

tags: computer-use, ui-agent, browser-agent, cua, operator, 2026

## 摘要

Computer use 让模型通过鼠标、键盘和屏幕截图操作软件界面，适合没有 API 或 API 不完整的系统，例如遗留后台、网页表单、运营工具、桌面应用。

## 工作方式

宿主应用向模型提供任务和可用的 computer 工具。模型返回动作数组，例如点击、输入、滚动。宿主执行动作，截取新屏幕，再把屏幕作为观察结果发回模型。这个循环持续到模型不再请求 computer action。

## 适用场景

UI 自动化、网页 QA、数据录入、后台核验、跨系统操作、地图或图像界面查询。它尤其适合“人类能做但没有稳定 API”的流程。

## 风险

UI agent 能触达与人类相同的页面和按钮，因此安全边界更高。它可能误点、误读页面、受 prompt injection 影响，或在不可逆流程中提交错误操作。敏感动作应要求确认，环境应隔离，权限应最小化。

## 与 RPA 的区别

传统 RPA 依赖固定选择器和脚本，稳定但脆弱；UI agent 能用视觉和语言理解动态界面，适应性更强，但确定性更弱。生产系统常把两者结合：确定性步骤用脚本，开放性判断用模型。

## Sources

- OpenAI Computer use guide: https://developers.openai.com/api/docs/guides/tools-computer-use
- OpenAI New tools for building agents: https://openai.com/index/new-tools-for-building-agents/
- OpenAI Introducing Operator: https://openai.com/index/introducing-operator/
