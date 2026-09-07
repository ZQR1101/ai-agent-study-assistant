# DeepTutor（HKUDS）项目调研报告

- **调研日期**：2026-08-21
- **调研对象**：[HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)（DeepTutor: Lifelong Personalized Tutoring）、官方文档站 [deeptutor.info](https://deeptutor.info/)、arXiv 论文 2604.26962《DeepTutor: Towards Agentic Personalized Tutoring》
- **报告性质**：全部结论基于一手来源（GitHub README 与源码结构、GitHub Releases API、官方 Mintlify 文档站、arXiv 论文原文、GitHub 仓库元数据）。凡在一手来源中找不到的信息，均明确标注"未找到可靠来源"。

## 主要来源列表

| # | 来源 | URL | 用途 |
|---|------|-----|------|
| S1 | GitHub 仓库 README（main 分支） | https://github.com/HKUDS/DeepTutor（raw: https://raw.githubusercontent.com/HKUDS/DeepTutor/main/README.md ） | 定位、功能、安装、部署、版本演进 |
| S2 | arXiv 论文（摘要页/HTML 全文） | https://arxiv.org/abs/2604.26962 、 https://ar5iv.labs.arxiv.org/html/2604.26962 | 核心理念、架构设计、评测（TutorBench） |
| S3 | CITATION.cff | https://github.com/HKUDS/DeepTutor/blob/main/CITATION.cff | 作者名单、论文引用信息 |
| S4 | GitHub Releases（API） | https://api.github.com/repos/HKUDS/DeepTutor/releases 、 https://github.com/HKUDS/DeepTutor/releases | 版本时间线（69 个 release） |
| S5 | GitHub 仓库元数据（API） | https://api.github.com/repos/HKUDS/DeepTutor | stars、创建时间、license、topics |
| S6 | 源码结构（git tree API） | https://api.github.com/repos/HKUDS/DeepTutor/git/trees/main?recursive=1 | 后端/前端/CLI 组件盘点 |
| S7 | pyproject.toml | https://github.com/HKUDS/DeepTutor/blob/main/pyproject.toml | 依赖与技术栈 |
| S8 | CONTAINERIZATION.md | https://github.com/HKUDS/DeepTutor/blob/main/CONTAINERIZATION.md | Docker/Podman 部署细节 |
| S9 | 官方文档站（Mintlify） | https://deeptutor.info/ ；quickstart: https://mintlify.wiki/HKUDS/DeepTutor/quickstart ；introduction: https://mintlify.wiki/HKUDS/DeepTutor/introduction ；Docker: https://mintlify.wiki/HKUDS/DeepTutor/deployment/docker ；环境变量: https://mintlify.wiki/HKUDS/DeepTutor/deployment/environment-variables ；FAQ: https://mintlify.wiki/HKUDS/DeepTutor/troubleshooting/faq ；页面索引: https://mintlify.com/HKUDS/DeepTutor/llms.txt | 上手/部署文档（注意：内容滞后于 README，见 §10） |
| S10 | PyPI 包 `deeptutor` | https://pypi.org/project/deeptutor/ | 发布渠道 |
| S11 | 论文评测代码分支 | https://github.com/HKUDS/DeepTutor/tree/eval | 评测复现 |
| S12 | 外部佐证（DSML） | https://github.com/anomalyco/opencode/issues/24566 、 https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/discussions/209 | DSML = DeepSeek 模型的工具调用标记 |

---

## 1. 项目定位与核心理念

**一句话定位**：DeepTutor 是香港大学数据科学实验室（HKUDS）主导的开源"终身个性化辅导"（Lifelong Personalized Tutoring）系统，自我定位为 **agent-native learning workspace**——一个把辅导解题、出题、研究、可视化、精熟练习等串在同一个 agent 运行时上的可扩展学习工作区。仓库描述为 "DeepTutor: Lifelong Personalized Tutoring. https://deeptutor.info/."（[S1][S5]）。

**"Agentic Personalized Tutoring" 的含义**（论文 §1、Abstract，[S2]）：

- 论文把现有 LLM 辅导系统概括为"**会话有界的助手**"（session-bounded assistants）：它们只针对当下 prompt 做自适应，没有持久的 learner model；各功能是互相不共享状态的孤立模块（会解题的模块不知道出题模块昨天发现的薄弱点）。
- DeepTutor 的核心主张是：**个性化辅导需要统一架构而非孤立 prompt**——必须在多模态会话间同步"学习者记忆、任务分解、工具使用"，并且从"被动响应"平滑演进到"主动陪伴"而不重写核心运行时。
- 论文给出的两层设计维度：
  1. **Personalized Agentic Tutoring（个性化智能体辅导）**：混合个性化引擎（静态知识锚定 + 动态多分辨率记忆 trace forest），把交互历史蒸馏成持续演进的 learner profile；解题（citation-grounded problem solving）与出题（difficulty-calibrated question generation）构成闭环。
  2. **Proactive Autonomous Companionship（主动自主陪伴）**：通过 TutorBot 把辅导核心部署成具备可扩展 skills、多 bot 协同、跨渠道统一上下文的自主 agent。

**"终身个性化"（Lifelong）的含义**（论文 §2 Related Work、[S2]）：辅导本质上是持续数周/数月（longitudinal）的活动，需要随个体学习者演进的模型，而不是单次会话内的 bounded episode。DeepTutor 用"trace forest + 三维护 learner profile"实现这一点（详见 §3）。

**社区热度数据**（[S5][S1]）：仓库创建于 2025-12-28，2025-12-29 官方发布；README News 记录 10k stars 用时 39 天（2026-02-06）、20k stars 用时 111 天（2026-04-19）；调研当日（2026-08-21）GitHub API 显示 **36,828 stars、4,627 forks**，Apache-2.0 许可，主语言 Python。

### 引用来源
- [S1] GitHub README（标题、News、Key Features、Community）
- [S2] arXiv 2604.26962（Abstract、§1、§2）
- [S5] GitHub API 仓库元数据

---

## 2. 主要功能/能力

### 2.1 README（当前 v1.5.x）视角的关键能力

按 README "Key Features" 与功能导览（[S1]）：

- **单一运行时覆盖所有模式（One runtime for every mode）**：Chat、Quiz、Research、Visualize、Solve、Mastery Path、Immersive Reading 都跑在同一个 agent loop 上，"切换的是目标而不是引擎"，上下文随学习者流转。
- **互联的学习上下文（Connected learning context）**：知识库、书（Book）、Co-Writer 草稿、notebook、题库、persona、Memory 在所有工作流间可用，而非孤立工具。
- **Subagents 与 Partners**：可在任意一轮对话中咨询本机编码 CLI（Claude Code、Codex、Gemini、Kimi、opencode、MiMo）或 Partner，也能导入它们的过往对话；Partner 是"有性格和电话号码的聊天"——同一套 ChatOrchestrator 上的持久 IM 陪伴体，支持 Feishu、Telegram、Slack、Discord、DingTalk、QQ/NapCat、WeCom、WhatsApp、Zulip、Mattermost、Matrix、Mochat、Microsoft Teams 等渠道。
- **多引擎知识（Multi-engine knowledge）**：知识库可选择检索引擎——LlamaIndex（默认，本地向量 + BM25）、PageIndex（带页级引用的推理式检索，可托管也可自托管 OSS）、GraphRAG 与 LightRAG（知识图谱检索）、LightRAG Server（外部 LightRAG 实例）、腾讯 IMA 库、Obsidian 库；文档解析引擎可选 Text-only、MinerU、Docling、Tika、markitdown、PyMuPDF4LLM、LiteParse。
- **可扩展工具与技能**：内置工具、MCP 服务器、CLI Apps（来自 [CLI-Anything](https://github.com/HKUDS/CLI-Anything) 目录）、图像/视频/语音生成模型、可从 EduHub 安装社区技能（Agent-Skills 格式，`SKILL.md`）。
- **可审计的记忆（Inspectable memory）**：L1 轨迹、L2 表面摘要、L3 综合，配合 Memory Graph 把每条综合结论追溯到证据。
- **Chat 工具清单**：用户可开关工具 brainstorm / web_search / paper_search / reason / geogebra_analysis，以及 imagegen / videogen（配置生成模型后出现）；上下文相关工具（有对应上下文时自动挂载）包括 rag、kb_files、read_source、read_memory、write_memory、read_skill、load_tools、exec、web_fetch、ask_user、list_notebook、write_note、question_bank、github、consult_subagent 等。`ask_user` 是特殊工具：agent 可以暂停回合、提出结构化澄清问题、等用户回答后续跑。
- **代码执行沙箱**：office skills（docx/pdf/pptx/xlsx）由模型写 Python 脚本经 `exec`/`code_execution` 工具执行；本地/Docker 用受限子进程沙箱，docker-compose 形态下路由到加固的 runner sidecar（`Dockerfile.runner`）。
- **CLI 与 agent 对接**：`deeptutor chat`（REPL）与 `deeptutor run <capability> <msg> --format json`（NDJSON 事件流，供另一个 agent 驱动）；仓库还带一份约 150 行的根级 [SKILL.md](https://github.com/HKUDS/DeepTutor/blob/main/SKILL.md) 供工具型 LLM 快速接管。

### 2.2 官方文档站视角的模块（较旧，见 §10 注意事项）

官方 introduction/quickstart 文档（[S9]）把功能组织为八个模块：Smart solver（dual-loop 推理：Analysis Loop + Solve Loop）、Question generator（Custom/Mimic 两种模式）、Guided learning（由 notebook 记录生成个性化学习路径）、Deep research（DR-in-KG 三阶段：规划→研究→报告）、Idea generation（IdeaGen）、Co-writer（改写/缩短/扩写 + TTS 旁白）、Knowledge base、Notebook。**注意**：这些描述对应 v0.x–v1.0 时代的能力组织（如 "/solver" 路由、`scripts/start_web.py`），与当前 README 的能力组织（capability: chat/deep_solve/deep_question/deep_research/visualize/math_animator/mastery_path）不一致。

### 引用来源
- [S1] README（Key Features、Explore DeepTutor 各小节、Chat/Partners/Knowledge Center/Memory/Book/CLI）
- [S9] introduction.md、quickstart.md

---

## 3. 系统架构与关键组件

### 3.1 论文给出的设计原则与统一运行时（[S2] §3）

三条设计原则：

1. **共享个性化作为统一基座（Shared Personalization as the Unifying Substrate）**：所有工作流（解题、出题、写作、研究、引导学习）都路由到一个集中的个性化引擎——共享知识库 + trace forest + 统一 learner profile 𝒟。解题中发现的薄弱点直接塑造后续引导会话。
2. **可复用 agentic 工作流而非单体功能（Reusable Agentic Workflows over Monolithic Features）**：辅导功能是建立在共享运行时（RAG、agentic reasoning、沙箱代码执行、记忆访问、多 agent 编排）之上的可组合工作流，继承统一的上下文传播与流式协议。
3. **统一上下文的主动智能体（Proactive Agency with Unified Context）**：主动行为（TutorBot）是辅导基座的结构性扩展，学生上午用 Web、下午用 CLI、晚上用 Telegram，面对的是同一个 context-unified 的 tutor。

关键基础设施设计：**系统级统一上下文结构**（封装 session 元数据、对话历史、工具注册表、知识库引用、个性化信号 C_mem）；**强类型事件 + 异步事件总线**（把 agent 逻辑与投递层解耦）；**收敛式入口**（Web、CLI、SDK、TutorBot 全部汇入同一个 orchestrator，共享同一 learner model）。

### 3.2 论文的个性化引擎（[S2] §4.1）

- **静态知识锚定（Static Knowledge Grounding）**：课程材料拆成原子内容单元，用"知识图谱 𝒢 + 稠密 embedding 索引 ℬ"双结构索引；图谱遍历擅长结构化问题（"Stokes 定理的前置条件是什么"），稠密检索擅长语义模糊问题（"我不理解环量与通量的关系"）；两路候选经 reciprocal rank fusion 融合后截断为 C_rag。
- **动态个人记忆：Trace Forest（追踪森林）**：多分辨率层次数据结构，每棵树记录一次完整辅导交互，节点分三级——L1 会话级元数据与全局摘要、L2 任务分解产生的中间规划单元（子目标/状态/理由）、L3 细粒度执行记录（工具输出、检索证据、验证结果）；每个节点带稠密 embedding 支持跨森林相似检索。会话结束时生成一棵新 trace tree（planner 造 L2、executor 追加 L3、post-session summarizer 生成 L1 根）。
- **TraceToolkit**：程序化接口，三个操作——`SearchTrace`（语义检索 + 返回祖先路径）、`ListTraces`（按时间/主题/结果过滤枚举）、`ReadNodes`（读节点全文并重建祖先路径）。对系统内所有 agent 开放。
- **三维护 learner profile 𝒟 = (𝒟_s, 𝒟_w, 𝒟_r)**：三个专职 memory agent 并行处理每条新 trace：
  - 𝒟_s **Session History**：主题覆盖与表现趋势（发现反复出现的主题、平台期）；
  - 𝒟_w **Weakness Diagnosis**：知识缺口的有序清单，标注 active/resolved（连续两次会话正确应用则标记 resolved，复现则回退 active）；
  - 𝒟_r **Self-Reflection**：系统自身的教学自我批评（脚手架过密/过疏、类比是否奏效、节奏是否匹配）。
- **注入机制**：每个 agent 步骤前组装个性化上下文 C_mem（活跃 trace 检索 + 记忆摘要），避免上下文饱和，保证每个 agent 拿到恰好的个性化信号。

### 3.3 论文的闭环辅导循环与扩展（[S2] §4.2–4.3）

- **个性化解题**：Stage① 个性化调研（个性化最早进入管线）→ 多阶段 draft–refine 生成最终答案。
- **个性化出题**：围绕主题在个体学习者视角下生成候选题目与个性化 rationale，锚定 𝒟_w 中的缺口；出题结果与解题结果都回写 learner profile，形成双向闭环。
- **扩展模态**：协作写作（Co-Writer）、多 agent 深度研究（deep research）、交互式引导学习（guided learning）共享同一个性化基座，跨模态互哺（写作产生的 trace 丰富出题参数，研究洞察精化出题）。

### 3.4 论文的 TutorBot 主动层（[S2] §5）

- 四层架构：多通道接口（12 个消息平台适配器经统一消息总线）→ 自主 agent 核心（Soul persona + 可扩展 skills）→ 持久记忆系统（自动 consolidation）→ 共享 DeepTutor 运行时（把完整辅导栈暴露为可复用服务）。
- 复用运行时：TutorBot 与 Web 走同一 orchestrator 与个性化基座；每个实例在 DeepTutor 服务进程内运行，无需单独部署。
- 与标准 ReAct 的两点区别：**持久双层记忆**（长期用户画像 + 可检索的会话历史日志；自动监测上下文窗口压力、在溢出前把最旧消息蒸馏进两层记忆）；**高层辅导动作**（RAG 解释、深度推理、沙箱代码执行、学术论文检索作为一等动作）。
- **Skills**：声明式模块（triggers + 分步 instructions + 所需 tools + 可选脚本）；`skill-creator` 元技能允许 bot 在运行时自己创作并安装新技能。
- **多 bot 并行**：单部署可承载多个独立 TutorBot，各自有 persona（Soul 模板）、技能配置、主动调度、会话历史，共享知识库与运行时；heartbeat 服务周期性唤醒 agent 判断是否需要主动动作（如生成每日练习），LLM 自己决定做或不做。
- **跨渠道统一上下文**：Web 上午、CLI 下午、Telegram 晚上，同一 trace forest / learner profile / 个性化上下文贯穿所有入口。

### 3.5 代码层面的组件盘点（[S6] 源码结构，README [S1] 佐证）

- **后端（Python 包 `deeptutor/`）**：`deeptutor/api/`（FastAPI：main.py、run_server.py、30+ 个 routers，含 chat、knowledge、memory、partners、book、co_writer、mastery_path、unified_ws 等）；`deeptutor/agents/`（chat 的 agent_loop.py / agentic_pipeline.py / session_manager.py / context_budget.py / dsml_tool_calls.py，以及 research、question、visualize、math_animator、vision_solver、notebook 等能力管线，prompt 全部为 YAML）；`deeptutor/book/`（Book 引擎：ideation/page_planner/source_explorer/spine 等 agent + 12 种 typed blocks）；`deeptutor/services/`（RAG 多引擎 pipelines：graphrag/lightrag 等、parsing 层）。
- **前端（`web/`，Next.js 16）**：App Router 多页面（home/chat、knowledge、memory（L1/L2/L3/graph）、space（learning/notebooks/skills/mcp/cli-apps/personas/questions）、book、co-writer、partners、settings 各分页）；`web/proxy.ts` 是 Next.js 中间件，把 `/api/*`、`/ws/*` 请求在服务端转发给 FastAPI（这是"只发布 3782 一个端口"架构的关键，见 §6）。
- **CLI**：`deeptutor_cli`（`deeptutor` 入口命令），支持 `init/start/serve/run/chat/partner/kb/skill/memory/session/notebook/book/plugin/config/provider login`。
- **配置体系**：`data/user/settings/*.json|yaml`（model_catalog.json、system.json、auth.json、integrations.json、interface.json、main.yaml、agents.yaml）；README 明确"项目根 .env **不是**应用配置文件"。

### 引用来源
- [S2] arXiv 论文 §3、§4、§5、附录 A
- [S1] README（Chat 的 agent loop 说明、Memory、Settings、CLI）
- [S6] 仓库 git tree

---

## 4. 技术栈

（来源：[S7] pyproject.toml、[S1] README、[S6]）

- **语言/运行时**：Python 3.11–3.13（pyproject 强制 `>=3.11,<3.14`；README 标注 Python 3.11+）；前端 TypeScript + Next.js 16 / React 19（Node 20+ 运行打包后的 standalone server）。
- **后端框架**：FastAPI + uvicorn + websockets（WebSocket 流式）；Pydantic v2；PyYAML + Jinja2（prompt 管理）；typer + rich + prompt_toolkit（CLI）；loguru（日志）。
- **LLM SDK**：openai、anthropic、dashscope（阿里通义）、perplexityai、oauth-cli-kit（Codex OAuth）；"原生 OpenAI/Anthropic SDK（v1.0.0-beta.3 起弃用 litellm）"。
- **RAG/知识**：llama-index（>=0.14.12）、llama-index-retrievers-bm25、llama-index-vector-stores-faiss + faiss-cpu（大知识库 ANN 检索）；可选引擎 extras：`graphrag`（microsoft/graphrag 3.x）、`rag-lightrag`（HKUDS/RAG-Anything ≥1.2.5，会传递引入 MinerU）、pageindex（PageIndex 客户端）；PyMuPDF / pdfplumber / pypdf 等解析；可选解析 extras：markitdown、docling、pymupdf4llm、liteparse。
- **Agent 生态**：mcp（Python MCP client，>=1.26,<2.0，核心依赖）；Agent-Skills 技能格式（SKILL.md）；CLI-Anything（CLI Apps 目录）。
- **Partner/IM**：python-telegram-bot、lark-oapi（飞书）、wecom-aibot-sdk（企微）、dingtalk-stream、slack-sdk、qq-botpy、zulip、matrix-nio（Matrix，可选 E2EE）、msteams（PyJWT）、微信二维码登录（qrcode）等，见 pyproject `[partners]` extra。
- **其他**：pocketbase（可选鉴权/存储 sidecar）、bcrypt + python-jose（鉴权）、json-repair（LLM JSON 修复）、psutil（内存探测）、manim（Math Animator 数学动画，可选）。
- **支持的模型服务**：docs（[S9]）列出的 LLM binding 包括 openai / azure_openai / anthropic / **deepseek** / openrouter / groq / together / mistral / ollama / lm_studio / vllm / llama_cpp；README 另提及 NIM（NVIDIA）、Lemonade、Atlas、Eden AI、Novita AI、Gemini、Kimi、MiMo、CodeBuddy、OpenAI Codex OAuth、GitHub Copilot（`deeptutor provider login`）。**注意**：docs 的环境变量表是 v0.x 时代的（LLM_BINDING/EMBEDDING_*），当前版本改为 `data/user/settings/model_catalog.json` 的 UI 配置。

### 引用来源
- [S7] pyproject.toml（dependencies 与 optional-dependencies）
- [S1] README（技术徽章：Python 3.11+、Next.js 16；Settings 小节；Open Source Partners 表）
- [S9] introduction.md / environment-variables.md

---

## 5. 快速上手/安装方式

README 提供四条安装路径（[S1]），共用同一工作区布局（设置位于启动目录下的 `data/user/settings/`，或用 `DEEPTUTOR_HOME` / `deeptutor start --home` 指定）：

**Option 1 — PyPI（完整 Web 应用 + CLI，无需 clone）**（需要 Python 3.11–3.13 与 PATH 上的 Node.js 20+）：
```bash
mkdir -p my-deeptutor && cd my-deeptutor
pip install -U deeptutor
deeptutor init     # 交互式：后端端口(默认8001)、前端端口(默认3782)、LLM provider、可选 embedding
deeptutor start    # 同时启动 backend + frontend
```
默认前端地址 http://127.0.0.1:3782 ；`Ctrl+C` 同时停止前后端。

**Option 2 — 源码安装（开发用）**：`git clone` → venv（Python 3.11–3.13）→ `python -m pip install -e .` → `( cd web && npm ci --legacy-peer-deps )` → `deeptutor init` → `deeptutor start --dev`（--dev 开启 Next.js HMR）。可选 extras：`.[dev]`、`.[partners]`、`.[matrix]`、`.[matrix-e2e]`、`.[math-animator]`。

**Option 3 — Docker**（见 §6）。

**Option 4 — 仅 CLI**（源码安装 `python -m pip install -e ./packaging/deeptutor-cli`，尚未发布到 PyPI；`deeptutor init --cli`、`deeptutor chat`）。常用命令：`deeptutor run chat "..."`、`deeptutor run deep_solve "..." --tool rag --kb my-kb`、`deeptutor kb create my-kb --doc textbook.pdf`、`deeptutor memory show`。

**旧文档路径（已过时，见 §10）**：官方 quickstart（[S9]）目前仍写着 `cp .env.example .env`（填 LLM_BINDING / LLM_MODEL / LLM_API_KEY / LLM_HOST / EMBEDDING_*）→ `docker compose up` 或 `python scripts/install_all.py` + `python scripts/start_web.py`，对应 v0.x 版本。

### 引用来源
- [S1] README（Get Started 四选项、CLI 命令参考）
- [S7] pyproject.toml（extras）
- [S9] quickstart.md

---

## 6. 部署方式

### 6.1 Docker 单容器（当前推荐，[S1][S8]）

镜像在 GitHub Container Registry：`ghcr.io/hkuds/deeptutor:latest`（稳定版）/ `:pre`（预发布）。单容器内用 supervisord 同时跑 FastAPI 后端（:8001）与 Next.js 前端（:3782），基础镜像 `python:3.11-slim`：

```bash
docker run --rm --name deeptutor \
  -p 127.0.0.1:3782:3782 \
  -v deeptutor-data:/app/data \
  ghcr.io/hkuds/deeptutor:latest
```

关键架构点：**只需发布 3782**——浏览器只跟前端 origin 通信，Next.js 中间件 `web/proxy.ts` 在容器内把 `/api/*`、`/ws/*` 转发给后端（运行时读 `DEEPTUTOR_API_BASE_URL`）；发布 8001 只是方便 curl/脚本直连 API。`/app/data` 卷持久化配置、API key、日志、工作区、记忆、知识库。

- 连宿主机模型服务：`--add-host=host.docker.internal:host-gateway`，然后 Settings → Models 里把 Base URL 指向 `host.docker.internal`（Ollama :11434、LM Studio :1234、llama.cpp :8080、Lemonade :13305）；Linux 也可 `--network=host`。
- 拆分部署（前后端分离容器）：在 `data/user/settings/system.json` 设 `next_public_api_base: "http://backend:8001"`；该值是服务端读取，浏览器不可见。
- 鉴权：`auth.json` 打开 `auth_enabled`，通过 `dt_token` cookie 门控 `/api/*` 与 `/ws/*`。

### 6.2 Compose / Podman / 加固形态（[S1][S8]）

- `docker-compose.yml`（源码构建 + sidecar）：包含 PocketBase sidecar（可选鉴权/存储，`integrations.pocketbase_url` 指向 `http://pocketbase:8090`）与 **sandbox-runner sidecar**（`Dockerfile.runner`，最小权限容器执行模型生成的代码，经 `DEEPTUTOR_SANDBOX_RUNNER_URL` 路由）。
- `compose.yaml`（rootless Podman 加固路径）：`read_only: true` 只读 rootfs + tmpfs + `userns_mode: keep-id` + 仅 loopback 端口；无 runner sidecar，主应用回退到 `bwrap` 或受限子进程后端（由 `sandbox_allow_subprocess` 控制，默认 true）。
- 安全姿势：supervisord 以 root 为 PID 1 但把 backend/frontend 进程降权到非 root 的 `deeptutor` 用户（UID 1000）；模型生成的代码默认在受限子进程沙箱中执行——README 明确提示"在宿主机上运行模型生成的代码是一个真实的信任决策"，可设 `sandbox_allow_subprocess=false` 关闭。

### 6.3 多用户部署（[S1]）

鉴权默认关闭（单用户）；开启后单 `data/` 树同时容纳 admin 工作区、`data/users/<uid>/` 隔离的用户工作区与 `data/partners/<id>/workspace/`。第一个注册用户成为 admin，拥有模型目录、provider 凭据、共享知识库、技能与 per-user grants；其他用户的 Settings 被脱敏（只能看到 admin 指派、只读的作用域选项）。

### 6.4 旧文档中的部署内容（[S9]，已过时见 §10）

官方 Docker 部署页：`docker compose up` 源码构建（首次约 10–15 分钟）或 `docker run ghcr.io/hkuds/deeptutor:latest --env-file .env`；云部署需设 `NEXT_PUBLIC_API_BASE_EXTERNAL`；自定义端口需 `BACKEND_PORT`/`FRONTEND_PORT` 与 `-p` 映射一致；给出 nginx HTTPS 反向代理配置（含 WebSocket `Upgrade` 头与 `X-Forwarded-Proto`）。**HuggingFace / MinerU 设置**（该文档站把此小节放在 environment-variables 页面，"Docker deployment"页面的同名锚点已失效）：`HF_ENDPOINT`（HF 镜像端点）、`HF_HOME`（缓存目录，文档建议 `/app/data/hf` 配合 Docker 卷复用模型）、`HF_HUB_OFFLINE=1`（离线模式）——这些是给 MinerU PDF 解析器下载模型用的。当前 README 中 MinerU 只是 Settings → Knowledge Base 的可选解析引擎之一（本地模型下载默认关闭），Docling 可配 remote 模式（Docling Serve），Tika 需 `TIKA_SERVER_URL`。

### 引用来源
- [S1] README（Option 3、Code Execution Sandbox、Multi-User、OpenAI Codex OAuth）
- [S8] CONTAINERIZATION.md
- [S9] deployment/docker.md、deployment/environment-variables.md、quickstart.md

---

## 7. 版本演进

（来源：[S4] Releases API：69 个 release，最早 v0.2.0（2026-01-02），最新 v1.5.15（2026-08-20）；[S1] README 的 Release 列表；仓库创建 2025-12-28、官方发布 2025-12-29）

**早期 v0.x（2026-01，功能化阶段）**：
- v0.2.0（01-02）：Docker 部署、Next.js 16 & React 19 升级、WebSocket 安全加固。
- v0.3.0（01-05）：统一 PromptManager 架构、GitHub Actions CI/CD、GHCR 预构建镜像。
- v0.4.0（01-09）：多 provider LLM 与 embedding、RAG 模块解耦。
- v0.5.0（01-15）：统一服务配置、按知识库选择 RAG 管线、出题重构；v0.5.2（01-18）加入 Docling。
- v0.6.0（01-23）：会话持久化、增量文档上传、完整中文本地化。

**v1.0 时代（2026-04，agent-native 重写）**：
- v1.0.0-beta.1（04-04）：**"Agent-native architecture rewrite（约 20 万行）"**——Tools + Capabilities 插件模型、CLI & SDK、TutorBot、Co-Writer、Guided Learning、持久记忆。
- v1.0.0-beta.2（04-07）：Python 3.11+ 最低版本、MinerU 嵌套输出、WebSocket 修复。
- v1.0.0-beta.3（04-08）：原生 OpenAI/Anthropic SDK（弃用 litellm）、Windows Math Animator 支持、完整中文 i18n。
- v1.0.x / v1.1.x / v1.2.x（04 中下旬）：Visualize 能力、附件（PDF/DOCX/XLSX/PPTX）、用户自定义 Skills、Book Engine"活书"编译器、多文档 Co-Writer、题库 @ 提及、数学 LaTeX 渲染。

**v1.3.x（2026-04 末–05，多用户与稳定性）**：版本化 KB 索引与重建工作流、NVIDIA NIM + Gemini embedding、可选多用户部署（隔离工作区/admin grants）、TutorBot Zulip、`deeptutor start`、远程 Docker CORS 恢复。

**v1.4.x（2026-05–06，能力大版本）**：
- v1.4.0（05-22，GA cut）：Auto Mode、**三层 Memory（L1/L2/L3）**、agentic Deep Research / Solve / Question、LlamaIndex RAG 重构、重启安全的 turn 运行时。
- v1.4.3（06-12）：TutorBot 升级为 **Partners**（生产级 IM 管线，15 渠道，实时流式）；聊天统一为单一 agent loop；真实 per-user 隔离。
- v1.4.5（06-14）：Guided Learning 重建在 chat agent loop 上，/learning 仪表盘、loop-plugin 框架。
- v1.4.6（06-17）：**四面合并**——Space 学习仪表盘（导入 My Agents、顶层 Memory）、Knowledge Center（GraphRAG / PageIndex / LightRAG / linked-KB / Obsidian）、Settings 全开放、per-model 能力门控。
- v1.4.7/8（06-18）：连接本机 Claude Code / Codex 并实时咨询（consult_subagent）；Partners 可挂到 My Agents 下。
- v1.4.12（06-24）：**LightRAG Server** 检索引擎、PyMuPDF4LLM 轻量解析、FAISS 向量后端（大知识库检索大幅加速）。

**v1.5.x（2026-07–08，连接与打磨）**：
- v1.5.0（07-04）：LlamaIndex 摄取遵循 Document Parsing 引擎（多模态图像抽取）；Python 3.14+ 上可选 RAG extras 可干净安装。
- v1.5.1（07-09）：支持从知识库移除单个失败文档（含 error 状态），不必整库删除重建。
- v1.5.5（07-26）：OpenAI Codex OAuth 登录（用自己的 ChatGPT 套餐）；Eden AI provider；可追溯的 `rag` 引用；GraphRAG 索引。
- v1.5.7（07-31）：per-account MCP Services 商店、101 个 CLI Apps、凭据移出沙箱可触及范围、移动端布局。
- v1.5.12（08-13）：Web 搜索六个新 provider（Doubao、Bocha、Zhipu、Firecrawl、Qianfan、Aliyun IQS）、LiteParse 解析引擎、凭据变更后 MCP 服务器自动重连、CodeBuddy + OrcaRouter。
- v1.5.14（08-19）：**Immersive Reading**（文档与对话并列、逐页引用）、对话内配置 DeepTutor、IMA 库读写、notebook 控制台。
- v1.5.15（08-20，最新）：**PageIndex OSS 自托管 + 推理式检索**、可归档的题库、第三方工具/能力插件、**Apache Tika** 解析。

**演进主线**：v0.x 功能单体 → v1.0.0-beta.1 的 agent-native 重写（统一 agent loop + capability 插件化）→ v1.4 的三层记忆与全能力 agent 化 → v1.4.6 起的四面/工作区整合（Space/Knowledge Center/My Agents）→ v1.5 的连接生态（MCP、CLI Apps、Codex OAuth、外部检索引擎、多解析引擎）。

### 引用来源
- [S4] GitHub Releases API / Releases 页面
- [S1] README Releases 列表与 News

---

## 8. 相关项目/论文背景

### 8.1 HKUDS 实验室与作者

- DeepTutor 由 [Bingxi Zhao](https://github.com/pancacake)（赵丙玺）在 **HKUDS Group** 内主导，完全开源、无付费产品（[S1] Community 小节）。
- 论文致谢 HKU 数据智能实验室主任 **Chao Huang**（黄超）以及 Jiahao Zhang、Zirui Guo、Xubin Ren 等 labmates（[S1]）。CITATION.cff 列出的作者：Bingxi Zhao、Jiahao Zhang、Xubin Ren、Zirui Guo、Tianzhe Chu、Yi Ma、Chao Huang（[S3]）。
- 论文主页链接指向 arXiv 2604.26962（[S5]），标注 "Tech Report, work in progress"，2026-04-10 上线 arXiv（[S1] News）。

### 8.2 论文（arXiv 2604.26962）要点

《DeepTutor: Towards Agentic Personalized Tutoring》（[S2]）：
- 提出混合个性化引擎（静态知识锚定 + trace forest 动态记忆）、解题–出题闭环、TutorBot 主动层。
- 提出 **TutorBench**：学生中心基准，30 个知识库、5 大学科（人文/科学/工程/商科/前沿研究）、90 个 learner profile、270 个交互任务；采用"第一人称交互式评测协议"——由 profile 驱动的 LLM 学生模拟器（Gemini-3-Flash）进行多轮对话，独立 LLM judge（Claude Sonnet 4.6，零温度）按个性化 rubric 打分（10 个维度：solve-side 的 Source Faithfulness/Personalization/Applicability/Vividness/Logical Depth，practice-side 的 Fitness/Groundedness/Diversity/Answer Quality/Cross Concept）。
- 基线：Naive Tutor / CoT Tutor / Self-Refine Tutor / ReAct Tutor。
- 结果：个性化辅导指标平均提升 **10.8%**（相对最强基线）；关闭个性化引擎后，解题管线在五个通用基准（HLE、GPQA-Diamond、LiveBench、GAIA、AA-LCR）上使五个 backbone 模型的通用推理平均提升 **28.6%**（正文/结论数字；摘要处写 29.4%，见 §10）。消融显示 RAG（事实锚定）与 Memory（个性化深度）沿互补轴贡献。
- 评测代码在 [tree/eval](https://github.com/HKUDS/DeepTutor/tree/eval)（[S11]）。

### 8.3 与相关开源项目的关系

README 的 Appreciation 表（[S1]）明确列出依赖/灵感来源：
- **[LlamaIndex](https://github.com/run-llama/llama_index)**：RAG 管线与文档索引主干（当前默认 KB 引擎，[S7] 佐证）。
- **[nanobot](https://github.com/HKUDS/nanobot)**（HKUDS）：超轻量 agent 引擎，驱动最初的 TutorBot。
- **[LightRAG](https://github.com/HKUDS/LightRAG)**（HKUDS）：作为可选知识图谱检索引擎集成（[S7] `rag-lightrag` extra 走 HKUDS/RAG-Anything，并传递引入 MinerU）。
- **[AutoAgent](https://github.com/HKUDS/AutoAgent)**（HKUDS）：零代码 agent 框架（灵感来源）。
- **[AI-Researcher](https://github.com/HKUDS/AI-Researcher)**（HKUDS）：自动研究管线（灵感来源）。
- **[OpenClaw](https://github.com/openclaw/openclaw)**：ClawHub 背后的开放 agent 网关与技能生态。
- **OpenAI Codex / Claude Code**：agent-native 编码 CLI，启发了 CLI 工作流与 agent loop。
- **[ManimCat](https://github.com/Wing900/ManimCat)**：Math Animator 的 AI 数学动画生成。
- 另有合作伙伴 **[PageIndex](https://github.com/VectifyAI/PageIndex)**（VectifyAI，README 提供优惠码 DEEPTUTOR20）与 HKUDS 的 **[CLI-Anything](https://github.com/HKUDS/CLI-Anything)**（CLI Apps 目录）、**EduHub** 技能注册表（[S1]）。

### 8.4 与 DeepSeek-R1 的关系：**未找到可靠的一手来源**

- README 全文未提及 DeepSeek-R1；Appreciation 表、依赖表与源码清单中也没有 DeepSeek-R1 相关的直接依赖（[S1][S7]）。
- 官方文档（旧版）把 `deepseek` 列为受支持的 LLM provider binding 之一（`LLM_MODEL` 示例 `deepseek-chat`）（[S9]），这是"DeepSeek 系列模型可作为 DeepTutor 的模型后端"的证据，但**不是**与 DeepSeek-R1 项目本身的合作/依赖关系。
- 一个间接线索：当前源码 `deeptutor/agents/chat/dsml_tool_calls.py`（[S6]）与 v1.5.11 release note"DSML tool call 周围的正文不再消失"（[S1]）表明 chat agent loop 处理 **DSML** 工具调用标记——外部来源显示 DSML 是 DeepSeek 模型的工具调用标记格式（[S12]），即 DeepTutor 对 DeepSeek 模型的工具调用输出做了专门适配。这一点仅有外部佐证，官方 README 未明确说明。

### 引用来源
- [S1] README（Community、Open Source Partners、Ecosystem）
- [S2] arXiv 论文
- [S3] CITATION.cff
- [S5] GitHub API 元数据（homepage 指向 arXiv）
- [S6] 源码 tree（dsml_tool_calls.py）
- [S7] pyproject.toml（graphrag/rag-lightrag extras）
- [S9] environment-variables.md（deepseek binding）
- [S12] DSML 外部佐证

---

## 9. 与同类教育 AI 系统的简要对比

**论文内的对比（一手来源，[S2]）**：
- Related Work 把自身与三类工作区分：①工具增强 LLM agent（ReAct 等，局限于 bounded episodes）；②LLM 记忆/个性化（MemGPT、HiMem、G-Memory、知识追踪）；③AI4Edu（对话式辅导、多 agent 教学架构、开源辅导平台、MathTutorBench 等基准）。DeepTutor 的差异点：用 trace forest 统一多模态辅导 + 从学生视角（第一人称模拟对话）评测。
- TutorBench 上与四个基线（Naive/CoT/Self-Refine/ReAct）的定量对比：个性化指标平均 +10.8%；且论文指出四个基线"尽管推理策略不同却聚在很窄的性能带上"——即单纯更聪明的推理并不等于更好的个性化。

**与具体商业/知名产品的对比**：**未找到可靠的一手来源**。论文与 README 均未点名对比 Khanmigo、ChatGPT 教育模式、Duolingo 等具体产品（论文只把"现有 LLM 辅导产品是 session-bounded"作为一般性批评）。中文/英文二手报道（如 [unwire.hk 报道](https://unwire.hk/2026/04/10/deeptutor-hku-open-source-ai-tutor/software/)、[BAAI 在线教程](https://hub.baai.ac.cn/view/53930)）把 DeepTutor 描述为"港大开源 AI 个人补习老师/家教"，可作为社区反响参考，但属于二手转述，本报告不以此立论。

### 引用来源
- [S2] arXiv 论文 §2、§6
- 二手背景（仅参考）：https://unwire.hk/2026/04/10/deeptutor-hku-open-source-ai-tutor/software/ 、 https://hub.baai.ac.cn/view/53930

---

## 10. 未能核实或存疑的信息

1. **官方文档站内容滞后于 README**（已核实，重要）：deeptutor.info 的 Mintlify 文档（quickstart、introduction、Docker deployment、FAQ、environment-variables）整体描述的是 **v0.x 时代**的工作方式——`.env` + `LLM_BINDING`/`EMBEDDING_*` 环境变量、`scripts/install_all.py`、`scripts/start_web.py`、`/solver` 页面、LightRAG 知识图谱、Node 18/Python 3.10 门槛等；而当前 README（v1.5.x）是 `deeptutor` CLI + PyPI + `data/user/settings/*.json`、明确忽略项目根 `.env`、Python 3.11+。文档站 "Docker deployment" 页的 `#huggingface-/-mineru-settings` 锚点在页内已不存在（相关表格实际位于 environment-variables 页面）。写报告时以 README 为准、文档站内容均标注"旧版文档"。
2. **论文摘要与正文的指标数字不一致**（已核实）：通用推理增益摘要写 29.4%，正文（§1、§6、Conclusion）写 28.6%；论文自标 "Tech Report, work in progress"，两者并存。
3. **DSML 缩写**：源码与 release note 出现 "DSML tool call"，官方未给出展开；外部来源显示其为 DeepSeek 模型的工具调用标记格式（拼写有 "Makeup Language" 等差异），本报告按外部佐证转述并标注。
4. **TutorBench 数据/评测细节**：论文给出了 30 KB / 90 profile / 270 task 等统计与评测分支链接（tree/eval），但本调研未 clone 仓库逐行复核评测代码；TutorBench 数据集是否公开下载、license 等未在一手来源中确认。
5. **"20 万行重写"等规模数字**：v1.0.0-beta.1 release note 称 "~200k lines"（[S1]），未另行核实。
6. **内部实现细节**：内存 consolidator 的 Update/Audit/Dedup 预算、DSML 解析、proxy 转发等具体参数与实现，仅依据 README 描述与源码目录结构，未逐行阅读源码。
7. **与 DeepSeek-R1 / 具体商业产品的对比关系**：未找到可靠一手来源（见 §8.4、§9）。

---

*调研方法说明：本报告基于一手在线来源（GitHub README raw 文件、GitHub API、arXiv 论文 HTML、Mintlify 文档站原始 Markdown/HTML）逐条核实；未 clone 仓库到本地。报告中对每个关键论断都标注了来源编号（S1–S12）与链接。*
