"""Deliverable exporters, one per playbook deliverable kind.

Contract compliance → compliance_report_docx + scorecard_xlsx.
Delivery intake    → discovery_brief_docx + handover_docx (+ shared xlsx).

Export is gated at the API layer: only finalized documents export, so every
deliverable carries the expert sign-off.
"""

from __future__ import annotations

from io import BytesIO

from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from openpyxl import Workbook
from sqlalchemy.orm import Session

from backend.documents.models import Clause, Document, Rule, Verdict

RATING_LABELS = {"red": "红", "amber": "黄", "green": "绿"}


def _load_context(session: Session, document: Document) -> tuple[list[Verdict], dict[str, Rule]]:
    rules = {r.id: r for r in session.query(Rule).filter(Rule.playbook_id == document.playbook_id).all()}
    verdicts = (
        session.query(Verdict)
        .filter(Verdict.document_id == document.id)
        .order_by(Verdict.created_at)
        .all()
    )
    return verdicts, rules


def _base_docx(title: str, document: Document) -> DocxDocument:
    doc = DocxDocument()
    heading = doc.add_heading(title, level=0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta = doc.add_paragraph()
    meta.add_run(
        f"文档编号：{document.friendly_id}    文件：{document.source_filename}\n"
        f"剧本：{document.playbook_id}    状态：已定稿（含专家签字）\n"
    ).font.size = Pt(10)
    return doc


def _add_verdict_table(doc: DocxDocument, verdicts: list[Verdict], rules: dict[str, Rule]) -> None:
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    header = table.rows[0].cells
    header[0].text = "规则"
    header[1].text = "维度"
    header[2].text = "判定"
    header[3].text = "评审状态"
    for verdict in verdicts:
        rule = rules.get(verdict.rule_id)
        row = table.add_row().cells
        row[0].text = rule.name if rule else verdict.rule_id
        row[1].text = rule.dimension if rule else ""
        row[2].text = RATING_LABELS.get(verdict.rating, verdict.rating)
        row[3].text = verdict.review_state
    doc.add_paragraph()


def _render_scorecard_paragraph(doc: DocxDocument, scorecard: dict) -> None:
    counts = scorecard.get("counts", {})
    doc.add_paragraph(
        f"红 {counts.get('red', 0)} 项 · 黄 {counts.get('amber', 0)} 项 · 绿 {counts.get('green', 0)} 项"
        f"    风险指数 {scorecard.get('risk_index', 0)} / 100    覆盖率 {scorecard.get('coverage_pct', 0)}%"
    )


def build_compliance_report(session: Session, document: Document) -> bytes:
    verdicts, rules = _load_context(session, document)
    doc = _base_docx("供应商合同合规审查报告", document)
    _render_scorecard_paragraph(doc, document.scorecard or {})
    _add_verdict_table(doc, verdicts, rules)

    doc.add_heading("逐条判定与引用", level=1)
    for verdict in verdicts:
        rule = rules.get(verdict.rule_id)
        doc.add_heading(f"{rule.name if rule else verdict.rule_id} — {RATING_LABELS.get(verdict.rating, verdict.rating)}", level=2)
        doc.add_paragraph(verdict.rationale)
        for index, citation in enumerate(verdict.citations or [], start=1):
            doc.add_paragraph(f"引用{index}：{citation.get('quote', '')}", style="Intense Quote")
        if verdict.gap_reason:
            doc.add_paragraph(f"缺口说明：{verdict.gap_reason}")
        if verdict.expert_note:
            doc.add_paragraph(f"专家意见（{verdict.reviewed_by}）：{verdict.expert_note}")

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def build_discovery_brief(session: Session, document: Document) -> bytes:
    verdicts, rules = _load_context(session, document)
    gaps = [v for v in verdicts if v.rating == "red"]
    risks = [v for v in verdicts if v.rating == "amber"]
    doc = _base_docx("需求调研简报", document)
    doc.add_heading("缺口清单（需向客户追问）", level=1)
    if not gaps:
        doc.add_paragraph("无缺口。")
    for verdict in gaps:
        rule = rules.get(verdict.rule_id)
        doc.add_paragraph(
            f"[{rule.name if rule else verdict.rule_id}] {verdict.gap_reason or verdict.rationale}",
            style="List Bullet",
        )
    doc.add_heading("风险项", level=1)
    if not risks:
        doc.add_paragraph("无风险项。")
    for verdict in risks:
        rule = rules.get(verdict.rule_id)
        doc.add_paragraph(
            f"[{rule.name if rule else verdict.rule_id}] {verdict.rationale}",
            style="List Bullet",
        )
    doc.add_heading("判定汇总", level=1)
    _add_verdict_table(doc, verdicts, rules)
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def build_handover_doc(session: Session, document: Document) -> bytes:
    verdicts, rules = _load_context(session, document)
    doc = _base_docx("交付交接文档", document)
    doc.add_heading("背景", level=1)
    doc.add_paragraph(
        f"客户提交件《{document.title}》已完成风险体检与专家签字。"
        "以下判定为交付团队接手时的基线认知。"
    )
    doc.add_heading("体检结论", level=1)
    _render_scorecard_paragraph(doc, document.scorecard or {})
    doc.add_heading("逐条判定", level=1)
    _add_verdict_table(doc, verdicts, rules)
    doc.add_heading("风险与缺口明细", level=1)
    for verdict in verdicts:
        if verdict.rating == "green":
            continue
        rule = rules.get(verdict.rule_id)
        doc.add_heading(f"{rule.name if rule else verdict.rule_id}", level=2)
        doc.add_paragraph(verdict.rationale)
        if verdict.gap_reason:
            doc.add_paragraph(f"缺口：{verdict.gap_reason}")
        if verdict.expert_note:
            doc.add_paragraph(f"专家意见（{verdict.reviewed_by}）：{verdict.expert_note}")
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def build_scorecard_xlsx(session: Session, document: Document) -> bytes:
    verdicts, rules = _load_context(session, document)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "记分矩阵"
    sheet.append(["规则", "维度", "判定", "评审状态", "判定理由", "引用数", "缺口说明", "专家"])
    for verdict in verdicts:
        rule = rules.get(verdict.rule_id)
        sheet.append(
            [
                rule.name if rule else verdict.rule_id,
                rule.dimension if rule else "",
                RATING_LABELS.get(verdict.rating, verdict.rating),
                verdict.review_state,
                verdict.rationale,
                len(verdict.citations or []),
                verdict.gap_reason or "",
                verdict.reviewed_by or "",
            ]
        )
    summary = workbook.create_sheet("记分卡")
    scorecard = document.scorecard or {}
    summary.append(["指标", "值"])
    summary.append(["红", scorecard.get("counts", {}).get("red", 0)])
    summary.append(["黄", scorecard.get("counts", {}).get("amber", 0)])
    summary.append(["绿", scorecard.get("counts", {}).get("green", 0)])
    summary.append(["风险指数", scorecard.get("risk_index", 0)])
    summary.append(["覆盖率%", scorecard.get("coverage_pct", 0)])

    clauses_sheet = workbook.create_sheet("条款原文")
    clauses_sheet.append(["序号", "章节", "原文"])
    for clause in (
        session.query(Clause).filter(Clause.document_id == document.id).order_by(Clause.ordinal).all()
    ):
        clauses_sheet.append([clause.ordinal, clause.heading or "", clause.text])

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


# playbook deliverable key → (filename prefix, builder)
DELIVERABLE_BUILDERS = {
    "compliance_report_docx": build_compliance_report,
    "discovery_brief_docx": build_discovery_brief,
    "handover_docx": build_handover_doc,
    "scorecard_xlsx": build_scorecard_xlsx,
}


def render_deliverable(session: Session, document: Document, key: str) -> bytes:
    builder = DELIVERABLE_BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"未知交付物: {key}")
    return builder(session, document)
