# AI 黑客松获奖研究与方向决议：从 RAG Failure Lab 到尽调问卷自动应答

调研日期：2026-09-11（同日修订：摒弃 RAG Failure Lab，确立新方向）  
范围：2026 年 5–9 月已公布结果的 AI、Agent 与开发者工具类黑客松。优先采用主办方公告、官方赛页和获奖项目页。

## 观察到的获奖模式

### 1. 赢家解决的是一个具体闭环，不是展示能力列表

Agent Academy Hackathon 四个赛道的头名全部是"一种输入 → 一种可执行输出"的转换器，无一例外：

| 赛道 | 头名项目 | 输入 → 输出 |
|---|---|---|
| Recruit | Performance Development Assistant | 绩效反馈 → 发展路线图 |
| Operative | VendorGuard | 合同 PDF → 合规记分卡报告 |
| Special Ops | SprintForge | 会议转录/笔记 → Sprint 计划与 Jira 工作项 |
| Cowork Collective | Copilot Cowork Autonomous ITSM | 工单 → 服务台处理流程 |

评委明确强调了场景实用性、清晰流程和可演示结果。[微软官方获奖公告](https://devblogs.microsoft.com/powerplatform/agent-academy-hackathon-winners/)

### 2. 证据、验证和失败保护本身可以成为产品体验

DataHub Agent Hackathon 总冠军 Project Blackbox 使用 `READ → PROVE → ACT → VERIFY → WRITE` 的闭环：模型选择调查方向，确定性工具验证事实，人在写操作前授权。项目还主动演示健康数据、错误修复和评测污染等失败场景。[官方获奖项目页](https://devpost.com/software/project-blackbox)

同场获奖项目 Hindsight 将每次事故保存为下一次可检索的事后记录，并用冷启动 20 次工具调用、复用记忆后 15 次这一具体数字展示价值；它还提供离线 replay，让评委不必相信一次现场演示。[Hindsight 项目页](https://devpost.com/software/hindsight-cpxymj)

### 3. 强项目通常有一句能立刻演示的问题

Gemini Live Agent Challenge 的冠军 ORION 聚焦"手术医生不离开无菌操作即可用语音获得信息和视觉协助"；Drone Copilot 用自然对话替代复杂遥控；Wand 让用户用语音和指向直接操作网页。它们不是先解释技术，而是先呈现一个用户立刻理解的动作。[Google Cloud 官方获奖公告](https://cloud.google.com/blog/topics/developers-practitioners/winners-and-highlights-of-the-gemini-live-agent-challenge)

### 4. 可玩的测试环境比静态教程更有辨识度

NandaHack 先让参赛者在 NANDA Town 沙盒修复一个真实模块、增加失败前后测试，再构建可被普通 Agent 独立调用的服务。评审标准包括正确性、测试、文档、实用性、创意和低设置成本；第三名 Litmus 本身就是用于观察 Agent 在欺骗性工具面前如何行动的安全蜜罐。[NandaHack 官方结果](https://nandahack.media.mit.edu/)

## Operative Track 冠亚军拆解（本次方向变更的直接依据）

[官方获奖公告](https://devblogs.microsoft.com/powerplatform/agent-academy-hackathon-winners/)：**VendorGuard 第一名，且为全场所有作品最高分**；Engagement Hub 第二名；FrostByte AI Advisor 第三名。

### VendorGuard 的头名公式（五个可复用得分点）

仓库：[vendorguard-copilot-studio](https://github.com/experienceswithanishh/vendorguard-copilot-studio)。邮件触发的合同合规流水线：hub & spoke 编排器 + 四个专业代理，15 条规则 × 5 个维度的红黄绿记分卡，规则手册外置 Dataverse。

1. 可转述的量化对比：人工 2–4 小时 → 不到 2 分钟。
2. 产物 3 秒可懂：红黄绿记分卡。
3. 零接触叙事：从邮件进邮箱到报告进 Teams，全程无人干预。
4. 一句话架构优点：规则手册在 Dataverse，改规则不动代理。
5. 结果可复用：Q&A 代理让每份报告可自然语言查询。

弱点：合规判定不要求条款级引用，也没有人工审批门——这正是新方向要补的两个位置。

### Engagement Hub 的工程参考（第二名）

仓库：[engagement-hub-agent](https://github.com/leila-marspooner/engagement-hub-agent)。值得直接借鉴的工程决策：刻意混用模式（需要确定性的地方用受控话题流而非生成式代理）；AI 未经人工批准绝不创建 Engagement Request；幂等 Teams 通知（事件名 + 成功标志 + 行 ID 三者一起匹配才算幂等）；结构化日志分类法与 correlation ID；显式测试 happy / duplicate / failure 三条路径。

### 修正后的评委偏好结论

公约数是具体闭环 + 可量化结果 + 可演示链路。治理与人工审批是强加分项（第二名证明），但头名公式是量化最锋利的自主闭环（全场最高分证明）。

## 方向决议：摒弃 RAG Failure Lab（2026-09-11）

决定：不再投入"用户亲手修复一次 RAG 失败"的学习型产品。

理由：

1. 它销售的是教学价值，不是企业每天付费的闭环；四个赛道的头名全部是输入 → 输出的业务转换器。
2. 核心资产不浪费——失败案例、分层指标、拒答与回放，在新方向里以"证据引用 + 拒答 + 覆盖率指标"的形态全部复用（见下表）。

## 新方向：尽调问卷自动应答（Due Diligence Questionnaire Responder）

一句话：客户发来一份 300 题的安全/尽调问卷，系统一小时内交付**每题带引用的答案草稿、红黄绿覆盖记分卡和缺口清单**；知识库覆盖不了的题，宁可留空标记给专家，也不编。

### 闭环（对标 VendorGuard 结构）

1. 问卷（Excel/Word/PDF）随邮件到达 → 触发器建 Dataverse/数据库记录。
2. 解析问卷，逐题拆分。
3. 检索公司知识库（安全策略、SOC 2/ISO 27001 报告、架构文档、历史问卷），生成带引用草稿。
4. 覆盖门：无强证据 → 留空 + 标记 + 进专家队列，人工批准后才定稿。
5. 生成覆盖记分卡与交付文档（Word/SharePoint），Teams 通知。
6. 已交付问卷全部可自然语言追问。

### 量化指标（对齐"2–4 小时 → 2 分钟"式表达）

- 周期：行业惯例 1–2 周 → 1 小时内。
- 三个可展示指标：覆盖率（带引用答案占比）、拒答正确率（知识库外的问题被正确留空）、引用命中率。

### 为什么是它：补 VendorGuard 的缺口，且用满现有资产

VendorGuard 全场最高分的最大缺口是"判定无引用、无人工门"。本方向把这两点做成硬约束，同时现有模块直接映射：

| 现有模块 | 产品角色 |
|---|---|
| Hybrid + CrossEncoder Reranker | 逐题证据检索 |
| Source Pollution 修复 / Negative-case Fallback | "知识库没有就不答"的覆盖门 |
| 结构化 Chunk + 元数据 | 引用定位到文档/章节 |
| Eval Harness（Top-k / MRR / Fallback / Pollution） | 覆盖率、拒答正确率、引用质量指标 |
| Tool Safety（requester/approver、pending action） | 专家审批队列 |
| Run Observability | 审计与交付回放 |

### 市场验证

该痛点已有成熟付费市场：Conveyor（Business 档约 $9,600/年）、SafeBase、Vanta、Drata、Loopio、Responsive、Whistic 等。[Conveyor 2026 盘点](https://www.conveyor.com/blog/the-best-security-questionnaire-automation-software-in-2026)、[Vanta 问卷自动化产品页](https://www.vanta.com/products/questionnaire-automation)。市场的存在证明预算真实；差异化在于引用硬约束与拒答门的可演示性，而不是"又一个问卷自动应答"。

### 两种交付形态

- **Microsoft 赛道**：Copilot Studio 编排 + Dataverse（问卷 / 答案 / 缺口 / 审批四张表）+ 自定义 MCP Server 承载检索/引用/拒答引擎——Special Ops 头名同款"低代码编排 + 代码 MCP"模式（SprintForge、Calamity、Warehouse Picking 均用自定义 MCP Server）。
- **独立产品**：现有 FastAPI/React 栈直接承载，邮箱进 → 记分卡出。

## 备选方向（未采用，留档）

- **RFP Bid/No-Bid 分析**：招标书 → 需求追溯矩阵 + 出价/弃标记分卡。记分卡形态最像 VendorGuard，但主难点是抽取与规则匹配，现有检索优势用不满。
- **ESG 披露核查**：供应商 ESG 报告 → 主张提取 → 证据核对 → 漂绿红黄绿。题材热，但演示数据与量化基线更难构造。

## 明确不建议（新方向下）

- 不克隆 VendorGuard 的采购合同域——同赛道已拿过第一名。
- 不交付任何不带引用的答案或判定。
- 不做通用文档问答聊天框。
- 不让模型回答知识库之外的问题——拒答是功能，不是缺陷。
- 一个产品只做一个闭环。

## 最小产品主张

> 知识库有的，带引用秒答；知识库没有的，明说没有，交给专家。
