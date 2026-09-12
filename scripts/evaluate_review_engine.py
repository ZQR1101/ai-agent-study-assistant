"""Review-engine evaluation: citation hit rate, refusal correctness, latency.

Runs the engine over ``eval_cases/review_cases.json`` with the REAL LLM
(unless ``--fake`` mechanics check) and reports:

- citation_hit_rate   : share of citations that trace back to the document
- refusal_correctness : gap verdicts (red + gap_reason) that cite no fake
                        evidence and match expected min_red/max_red
- coverage_correctness: verdicts where documented content produced green/amber
- latency             : per-document end-to-end wall time

Output: ``reports/REVIEW_ENGINE_EVAL_REPORT.md`` + a JSON blob next to it.
This script needs a configured model API key; ``--fake`` runs a mechanics-only
pass with a scripted LLM so CI can validate the harness offline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CASES_PATH = PROJECT_ROOT / "eval_cases" / "review_cases.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "REVIEW_ENGINE_EVAL_REPORT.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "REVIEW_ENGINE_EVAL_METRICS.json"


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class MechanicsFakeLLM:
    """Deterministic pass used only to validate the harness offline."""

    def invoke(self, prompt: str) -> _FakeResponse:
        import re

        rule_match = re.search(r"- 规则：(.+)", prompt)
        rule = rule_match.group(1).strip() if rule_match else "规则"
        payload = {
            "rating": "red",
            "rationale": f" MechanicsFake 判定 {rule}：未找到满足标准的条款。",
            "citations": [],
            "gap_reason": "MechanicsFake：文档未约定该事项",
        }
        return _FakeResponse(json.dumps(payload, ensure_ascii=False))


def _metrics_for_run(detail: dict) -> dict:
    verdicts = detail["verdicts"]
    total_citations = sum(len(v["citations"]) for v in verdicts)
    invalid_citations = sum(len(v.get("invalid_citations", [])) for v in verdicts)
    gap_reds = [v for v in verdicts if v["rating"] == "red" and v.get("gap_reason")]
    content_reds = [v for v in verdicts if v["rating"] == "red" and not v.get("gap_reason")]
    greens_ambers = [v for v in verdicts if v["rating"] in ("green", "amber")]
    return {
        "verdicts": len(verdicts),
        "citations": total_citations,
        "invalid_citations": invalid_citations,
        "gap_reds": len(gap_reds),
        "content_reds": len(content_reds),
        "greens_ambers": len(greens_ambers),
    }


def run_case(case: dict, fake: bool) -> dict:
    from backend.cli import cmd_review  # noqa: F401  (ensures cli deps importable)
    from backend.documents.models import Document
    from backend.documents.service import create_document
    from backend.engine.orchestrator import run_review
    from backend.engine.citation_gate import validate_citations
    from backend.platform_db import platform_session

    document_path = PROJECT_ROOT / case["document"]
    content = document_path.read_bytes()
    custom_llm = MechanicsFakeLLM() if fake else None

    session = platform_session()
    try:
        document, created = create_document(
            session,
            playbook_id=case["playbook_id"],
            filename=document_path.name,
            content=content,
            actor="eval",
        )
        document_id = document.id
    finally:
        session.close()

    started = time.perf_counter()
    run_review(document_id, trigger="eval", custom_llm=custom_llm)
    latency_ms = int((time.perf_counter() - started) * 1000)

    session = platform_session()
    try:
        document = session.get(Document, document_id)
        rules = {r.id: r for r in session.query(Rule).filter_by(playbook_id=case["playbook_id"]).all()}
        verdict_rows = []
        for verdict in document.verdicts:
            rule = rules.get(verdict.rule_id)
            verdict_rows.append(
                {
                    "rule_name": rule.name if rule else verdict.rule_id,
                    "rating": verdict.rating,
                    "citations": verdict.citations,
                    "invalid_citations": [],
                    "gap_reason": verdict.gap_reason,
                }
            )
        scorecard = document.scorecard
    finally:
        session.close()

    # Re-validate citations against the raw document for the hit-rate metric.
    doc_text = document_path.read_text(encoding="utf-8", errors="ignore")
    valid_total = 0
    citation_total = 0
    for row in verdict_rows:
        if row["citations"]:
            valid, invalid = validate_citations(row["citations"], doc_text)
            valid_total += len(valid)
            citation_total += len(valid) + len(invalid)
            row["invalid_citations"] = invalid

    metrics = _metrics_for_run({"verdicts": verdict_rows})
    metrics.update(
        {
            "case_id": case["id"],
            "playbook_id": case["playbook_id"],
            "latency_ms": latency_ms,
            "citation_total": citation_total,
            "citation_hit_rate": round(valid_total / citation_total, 4) if citation_total else None,
            "expectation_met": None,
        }
    )

    checks = []
    if "min_red" in case:
        checks.append(metrics["gap_reds"] + metrics["content_reds"] >= case["min_red"])
    if "max_red" in case:
        checks.append(metrics["gap_reds"] + metrics["content_reds"] <= case["max_red"])
    metrics["expectation_met"] = all(checks) if checks else None
    return metrics


def write_report(results: list[dict], *, fake: bool) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Review Engine 评测报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        f"运行模式：{'MechanicsFake（离线机制验证）' if fake else '真实 LLM'}",
        "",
        "| Case | 剧本 | 判定数 | 引用 | 引用命中率 | 缺口红 | 内容红 | 绿/黄 | 时延 | 预期 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['case_id']} | {r['playbook_id']} | {r['verdicts']} | {r['citation_total']} "
            f"| {r['citation_hit_rate'] if r['citation_hit_rate'] is not None else '—'} "
            f"| {r['gap_reds']} | {r['content_reds']} | {r['greens_ambers']} "
            f"| {r['latency_ms']} ms | {'✅' if r['expectation_met'] else '❌'} |"
        )
    overall_citations = [r["citation_hit_rate"] for r in results if r["citation_hit_rate"] is not None]
    if overall_citations:
        lines.append("")
        lines.append(
            f"**整体引用命中率：{sum(overall_citations) / len(overall_citations):.2%}**"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="Offline mechanics pass (no API key)")
    parser.add_argument("--fresh", action="store_true", help="Use a temp platform DB")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if args.fresh:
        import tempfile

        import os

        tmp = tempfile.mkdtemp()
        os.environ["PLATFORM_DB_PATH"] = str(Path(tmp) / "eval.db")
        os.environ["PLATFORM_DOCS_DIR"] = str(Path(tmp) / "docs")
        os.environ["AUTH_SECRET"] = "eval-secret-0123456789abcdef0123456789abcdef"
        from backend.platform_db import init_platform_db, reset_platform_db

        reset_platform_db()
        init_platform_db()

    results = [run_case(case, fake=args.fake) for case in cases]
    write_report(results, fake=args.fake)
    print(f"Wrote {REPORT_PATH}")
    for result in results:
        flag = "PASS" if result["expectation_met"] else "FAIL"
        print(f"[{flag}] {result['case_id']}  red={result['gap_reds']}+{result['content_reds']}")
    return 0


from backend.documents.models import Rule  # noqa: E402  (late import for env setup)

if __name__ == "__main__":
    raise SystemExit(main())
