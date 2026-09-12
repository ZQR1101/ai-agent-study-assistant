# Stitch 前端生成提示词包 v2 · Rulebook 文档审查平台

> **v2 修订原因**：v1 主提示词把全部产品概念倒给 Stitch 后只说"生成工作台（见 §2）"，但 Stitch 看不到 §2，于是把所有概念画进了一屏，产出信息过载的混血界面。v2 的修正：**主提示词自带完整首屏规格、每屏数量封顶、语言锁死、列表行只许一行标题。**

使用方法：打开 [stitch.withgoogle.com](https://stitch.withgoogle.com)，新建 **Web / Desktop** 项目。先粘贴 §1 主提示词（自带首屏规格，一次生成），然后逐屏粘贴 §3 跟进提示词。已有会话混乱时先粘贴 §2 抢救提示词。生成后可 Export to Figma 或导出代码。

---

## §2 抢救提示词（粘进已经画乱的会话，先救当前屏）

```text
Simplify this screen drastically. Keep ONLY these four zones, nothing else:
1) Header: page title "工作台" + one primary button "上传文档".
2) A slim segmented control with two tabs: "供应商合同合规" and "客户交付件风险分析".
3) One single-line text stats row (plain text with dividers, NOT cards):
   "文档 24 · 待签字 6 · 红灯判定 11 · 平均覆盖率 82%".
4) One clean data table, 6 rows, columns exactly:
   编号 | 标题 | 状态 | 红黄绿 | 覆盖率 | 更新时间.
   Each row: ONE title line + small chips only. Remove ALL inline detail cards,
   rule panels, verdict rationales, quote previews, progress rings, banners,
   footers and pagination from the rows.
Additional rules:
- Translate EVERY remaining English word into Simplified Chinese. English is
  allowed ONLY inside document IDs (CG-0007, DI-0003).
- Generous whitespace: 16px+ row spacing, max 2 buttons visible on this screen.
- Only ONE filled indigo primary button on the entire screen ("上传文档").
```

---

## §1 主提示词 v2（新会话第一次粘贴，自带首屏规格）

```text
Design a desktop web application called "Rulebook" (规则手册): a rulebook-driven
document review platform for small procurement/legal/consulting teams. Chinese
UI (Simplified Chinese). Light theme, professional and calm, whitespace-first.

PRODUCT CONTEXT (background only — do NOT visualize all of it now):
Documents (vendor contracts, client SOWs) are reviewed by an AI engine against
an editable rulebook, producing red/amber/green verdicts that cite the document
text; red/amber verdicts need expert sign-off before export. The visual language
is built on three semantic chips: 红 #E03131, 黄 #F08C00, 绿 #2F9E44.

STYLE SYSTEM
- Background #F8F9FB; white cards with 1px #E9ECEF border, 8px radius, subtle shadow.
- Primary accent: deep indigo #3B5BDB — used for exactly ONE primary button per screen.
- Typography: Inter for Latin/digits, Noto Sans SC for Chinese; monospace for IDs (CG-0007).
- Left sidebar 200px: logo "Rulebook", nav items 工作台 / 审批队列 / 规则手册 / 审计日志,
  user "王·管理员" with role tag at the bottom. Lucide-style thin line icons.
- Layout grammar: max 3 zones per screen; 24px page padding; 16px+ card gaps.

HARD RULES — apply to every screen, no exceptions:
1. LANGUAGE LOCK: every visible label, title, button, column header, chip and
   empty-state in Simplified Chinese. English appears ONLY inside document IDs
   (CG-0007, DI-0003). Never render English words inside Chinese labels.
2. ONE SCREEN = ONE JOB. List screens never show quotes, citations, rationales
   or rule details — those live on the detail screen only.
3. COUNT CAPS per screen: at most 1 primary button, 1 table OR 1 card stack,
   2 secondary buttons, 1 slim stats row. No decorative banners, no footer,
   no pagination unless requested.
4. HIERARCHY: page title > table > chips. Metrics are small text or chips,
   never oversized dashboard cards.
5. Empty space is part of the design: when in doubt, remove elements.

NOW GENERATE THE FIRST SCREEN — 工作台 (document list), with EXACTLY these zones:
Zone 1 — Header row: title "工作台" (20px, semibold) + right-aligned primary
button "上传文档".
Zone 2 — TWO playbook selector cards side by side (the product's core concept
"one engine, two playbooks" — make them prominent, but each card is exactly
3 lines: title / one gray description line / one small inline stats line):
  Card 1 (active: 2px indigo border + small check icon):
    供应商合同合规 · 15 条规则 × 5 维度 · 审查供应商合同 · 文档 18 · 待签字 4 · 红灯 9
  Card 2 (inactive, muted 1px border):
    客户交付件风险分析 · 12 条规则 × 4 维度 · 分析客户 SOW 与服务请求 · 文档 6 · 待签字 2 · 红灯 2
  No icons besides the check, no charts, no oversized numbers.
Zone 3 — One data table with exactly 6 columns:
  编号 (monospace: CG-0007) | 标题 | 状态 (chip) | 红黄绿 (three tiny colored
  count chips r/a/g) | 覆盖率 (small % text) | 更新时间.
  6 realistic rows, statuses as colored chips: 待签字 (amber), 已定稿 (green),
  评分中 (gray), 失败 (red). Example row: CG-0007 · 华东智造数据服务合同 ·
  待签字 · 红2 黄3 绿10 · 87% · 09-11 14:20.
  Below the table: one muted text "共 24 份文档".
Nothing else. No stat cards, no charts, no banners, no detail panels, no footer.
```

---

## §3 逐屏跟进提示词（每屏一次，主提示词的 HARD RULES 继续生效）

### Screen · 剧本选择卡强化（对已生成的分段 tab 工作台使用）

```text
Replace the small segmented tabs under the page title with two playbook
selector cards, side by side, occupying the full content width. This is the
product's core concept — "one engine, two playbooks" — so make it prominent.

Card 1 (active, 2px indigo #3B5BDB border, white background, small check icon):
  Title: 供应商合同合规
  One gray line: 15 条规则 × 5 维度 · 审查供应商合同
  Small inline stats (text, not badges): 文档 18 · 待签字 4 · 红灯 9

Card 2 (inactive, normal 1px #E9ECEF border, slightly muted):
  Title: 客户交付件风险分析
  One gray line: 12 条规则 × 4 维度 · 分析客户 SOW 与服务请求
  Small inline stats: 文档 6 · 待签字 2 · 红灯 2

Rules: each card is exactly 3 lines tall (title / description / stats), no
icons except the check, no charts, no large numbers. Clicking a card switches
the document table below. The table, header and upload button stay unchanged.
```

### Screen · 文档详情（核心屏）

```text
Now design the document detail screen 文档详情 — still exactly 3 zones.

Zone 1 — Header: back arrow, "CG-0007 · 华东智造数据服务合同", status chip
"待签字"; right side: secondary button "导出报告" (disabled, tooltip "红/黄判定
需专家签字后才能导出") and primary button "完成定稿".

Zone 2 — Scorecard strip: ONE white card, single row inside:
left: "风险指数" with number 34/100; middle: "覆盖率 87%"; right: three count
chips 红 2 · 黄 3 · 绿 10. Keep it one line tall — no dials, no dimension bars.

Zone 3 — Verdict list grouped by dimension (group header: "法律 · 6 条").
Each verdict row is a quiet card: left rating badge 红/黄/绿, rule name +
review-state tag (待签字 amber / 已批准 gray-green / 已改判 blue / 已驳回 red),
one line of rationale, and a collapsible "引用原文 ▾" revealing ONE quote block
(left 3px indigo border, verbatim Chinese contract text, e.g. "甲方于收到发票后
45个工作日内付款"). For a red verdict missing content show a small amber callout
"缺口：合同未约定数据泄露通知时限".
Rows in 待签字 state show three compact text buttons: 批准 · 改判 · 驳回.
Show exactly one row with the approve popover open: title "确认判定", expert
note input "专家意见（选填）", buttons 取消 / 确认批准.

Nothing else: no Q&A panel, no audit, no tabs on this screen.
```

### Screen · 审批队列

```text
Design the approval queue screen 审批队列. Sidebar item shows a red badge "6".

Zone 1 — Header: title "审批队列" + subtitle "红灯置顶 · 全部处理后才能定稿导出".
Zone 2 — Filter chips: 全部 / 红 / 黄.
Zone 3 — Vertical stack of 3 action cards (2 red, 1 amber), each card ONE row
layout: rating badge + rule name + document link chip "CG-0007 →" on line 1;
rationale one line; right side: three text buttons 批准 · 改判 · 驳回.
Bottom: one faded example card "已批准 ✓ 付款账期 · CG-0007".
Empty state variant: green-tinted card "队列已清空 — 可以去工作台定稿导出".
No quotes, no citations, no banners. Citation details live in 文档详情.
```

### Screen · 规则手册（管理员）

```text
Design the rulebook editor 规则手册.

Zone 1 — Header: title "规则手册" + right primary button "＋ 新增规则".
Zone 2 — One muted info line (text, not banner): "规则保存在数据库，保存后下一份
文档立即生效。".
Zone 3 — Rules grouped by dimension (商业 / 法律 / 数据隐私 / SLA与绩效 / 监管),
section header "法律 · 3 条". Each rule row: name (付款账期) · weight tag ×1/×2 ·
active toggle · edit icon; guidance as ONE gray line clamped: "账期不超过 60 天
为绿；61–90 天为黄；超过 90 天为红。". One row shown muted with toggle off.
Show one edit modal: 规则名称 / 所属维度 (select) / 判定标准 (textarea) / 权重,
buttons 取消 / 保存.
```

### Screen · 审计日志

```text
Design the audit log screen 审计日志.

Zone 1 — Header: title "审计日志" + segmented control "全局 / 单文档".
Zone 2 — One table: 时间 | 关联编号 (monospace chip CG-0007) | 事件 (monospace
chip: document.created / review.scored / verdict.approve / document.finalized) |
操作人. 6 rows, alternating subtle row backgrounds.
Zone 3 — For the 单文档 mode show a vertical timeline for CG-0007 with 4 nodes:
文档创建 → 解析完成 · 15 条款 → 评分完成 · 15 判定 → 文档定稿, each with timestamp.
Monospace for IDs and event names; nothing else.
```

### Screen · 登录页

```text
Design a minimal login screen 登录.

One centered card (max 400px) on #F8F9FB: wordmark "Rulebook", tagline "规则手册
驱动的文档审查平台", fields 用户名 / 密码 (eye icon), one primary full-width
button "登录", muted hint "首次启动自动创建管理员账号 admin，初始密码见服务器
data/bootstrap_admin_password.txt". Error variant: red inline "用户名或密码错误".
One card, zero decoration, no footer.
```

---

## §4 与后端对接速查（给实现者，不粘给 Stitch）

| Stitch 元素 | 真实 API |
|---|---|
| 登录表单 | `POST /auth/login` → `{token}`，后续带 `Authorization: Bearer` 或依赖 cookie |
| 工作台列表 | `GET /documents?status=` + `GET /documents/playbooks` |
| 上传弹窗 | `POST /documents/upload`（multipart: `file` + `playbook_id`）；409 = 重复，`detail.friendly_id` 跳转已有文档 |
| 文档详情 | `GET /documents/{id}`（`document.scorecard` + `verdicts[]` + `clauses[]`）；处理中轮询至 `status` 离开 `parsing/scoring` |
| 签字按钮 | `PATCH /documents/{id}/verdicts/{vid}` body `{decision, expert_note, new_rating?}`；422 = 非法迁移，409 = 已定稿 |
| 完成定稿 | `POST /documents/{id}/finalize`；409 = 仍有待签字 |
| 导出 | `GET /documents/{id}/export?format=docx|xlsx`（仅 finalized） |
| 追问面板 | `POST /documents/{id}/ask` body `{question}` |
| 审批队列 | `GET /documents/review/queue` |
| 规则手册 | 规则 CRUD 尚未开放端点（阶段内为 DB 数据）；UI 可先按 §3 Screen 静态设计 |
| 审计 | `GET /documents/{id}/audit`、`GET /documents/audit/recent`、`GET /documents/{id}/review-events` |

状态 → 中文标签映射：`uploaded 待处理 / parsing 解析中 / scoring 评分中 / awaiting_review 待签字 / finalized 已定稿 / failed 失败`；`drafted 初判 / awaiting_review 待签字 / approved 已批准 / edited 已改判 / rejected 已驳回`。
