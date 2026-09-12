"""Playbook 3: 数据处理协议（DPA）审查 — the third vertical, engine untouched.

Landing this playbook as pure data (spec + rulebook seeds + deliverable key)
is the engine-generality proof for phase 2: no engine module may change.
"""

from __future__ import annotations

from backend.playbooks.base import PlaybookSpec

_DPA_RULES = (
    # —— 处理范围与目的 ——
    dict(
        dimension="处理范围与目的",
        name="处理目的限定",
        guidance="检查处理目的条款。目的明确列举且限定为该协议为绿；使用宽泛表述为黄；允许受托方为自身目的处理或无目的限定为红。引用目的条款原文。",
        weight=2,
    ),
    dict(
        dimension="处理范围与目的",
        name="数据最小化",
        guidance="检查数据类别与范围条款。明确列举必要数据类型为绿；部分列举为黄；数据类型与范围不受限为红。",
    ),
    dict(
        dimension="处理范围与目的",
        name="处理指令合规",
        guidance="检查控制方指令机制。约定仅按控制方书面指令处理并承诺合规为绿；指令机制笼统为黄；无指令条款为红。",
    ),
    # —— 数据主体权利 ——
    dict(
        dimension="数据主体权利",
        name="数据主体权利协助",
        guidance="检查协助响应访问/更正/删除请求的义务与时限。义务完整且有时限为绿；有义务无时限为黄；无协助条款为红。",
    ),
    dict(
        dimension="数据主体权利",
        name="透明度与告知",
        guidance="检查隐私告知与数据主体联系机制条款。完整定义为绿；部分定义为黄；未定义透明度义务为红。",
    ),
    # —— 安全措施 ——
    dict(
        dimension="安全措施",
        name="技术与组织措施",
        guidance="检查安全措施条款。加密、访问控制、假名化等具体措施被列举为绿；仅有笼统安全承诺为黄；无安全措施条款为红。",
        weight=2,
    ),
    dict(
        dimension="安全措施",
        name="安全事件通知",
        guidance="检查向控制方通报安全事件的时限。不超过 24 小时为绿；24–72 小时为黄；超 72 小时或无通报义务为红。",
    ),
    # —— 分包与跨境 ——
    dict(
        dimension="分包与跨境",
        name="分包方同意与披露",
        guidance="检查分包（子处理者）条款。需事先具体授权并披露清单为绿；仅事后告知为黄；允许自由分包为红。",
        weight=2,
    ),
    dict(
        dimension="分包与跨境",
        name="跨境传输保障",
        guidance="检查跨境传输条款。禁止跨境或经批准机制（SCC/认证）为绿；可跨境但保障机制模糊为黄；跨境不受限为红。",
        weight=2,
    ),
    # —— 保留删除与审计 ——
    dict(
        dimension="保留删除与审计",
        name="删除与返还义务",
        guidance="检查终止后的删除/返还条款。由控制方选择且需删除认证为绿；承诺删除但无认证为黄；无删除返还条款为红。",
        weight=2,
    ),
    dict(
        dimension="保留删除与审计",
        name="审计与记录义务",
        guidance="检查控制方审计权与处理记录义务。审计权完整且要求留存处理记录为绿；审计权受限为黄；无审计条款为红。",
    ),
)

DPA_REVIEW_PLAYBOOK = PlaybookSpec(
    id="dpa-review",
    name="数据处理协议审查",
    description="对照数据保护基线逐条审查数据处理协议（DPA），产出带引用的合规记分卡与审查报告。",
    friendly_id_prefix="DP",
    dimensions=("处理范围与目的", "数据主体权利", "安全措施", "分包与跨境", "保留删除与审计"),
    rule_seeds=_DPA_RULES,
    deliverables=("dpa_report_docx", "scorecard_xlsx"),
    scoring_instructions=(
        "你是数据处理协议（DPA）评审代理。对每条规则：在协议条款中寻找对应内容，"
        "按规则 guidance 给出 red/amber/green 判定；判定必须引用协议原文片段；"
        "协议中找不到对应内容时判 red 并说明缺失（gap_reason），不得臆造绿色判定。"
    ),
)
