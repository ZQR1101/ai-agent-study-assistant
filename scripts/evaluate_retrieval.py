"""Clause retrieval evaluation: keyword vs semantic vs hybrid selection.

Runs each annotated case in ``eval_cases/retrieval_cases.json`` against the
annotated sample contract under all three retrieval modes and reports the
selection hit rate (expected clause inside the selected set). Uses the real
local embedding model — run offline-safe only where the model exists; CI never
executes this script.

Output: ``reports/RETRIEVAL_EVAL_REPORT.md``
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CASES_PATH = PROJECT_ROOT / "eval_cases" / "retrieval_cases.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "RETRIEVAL_EVAL_REPORT.md"

MODES = ("keyword", "semantic", "hybrid")


def main() -> int:
    from backend.config import get_config
    from backend.engine.parsing import parse_document
    from backend.engine.retrieval import get_clause_embedder, select_clauses
    from backend.playbooks import get_playbook

    config = get_config()
    case_data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    document_path = PROJECT_ROOT / case_data["document"]
    playbook = get_playbook(case_data["playbook_id"])

    parsed = parse_document(document_path)
    seed_by_name = {seed["name"]: seed for seed in playbook.rule_seeds}

    embedder = get_clause_embedder()
    if embedder is None or not embedder.ensure_ready(parsed.clauses):
        print("[ERROR] 本地 embedding 模型不可用，semantic/hybrid 无法评测")
        return 1
    print(f"[INFO] embedding 模型加载成功：{config.embedding_model}")

    results = {mode: {"hit": 0, "total": 0} for mode in MODES}
    detail_rows = []
    for case in case_data["cases"]:
        seed = seed_by_name[case["rule_name"]]
        rule_text = f"{seed['name']} {seed['guidance']}"
        expected = set(case["expected_ordinals"])
        row = {"rule": case["rule_name"], "expected": sorted(expected)}
        for mode in MODES:
            selected = select_clauses(
                parsed.clauses,
                rule_text,
                embedder=embedder if mode != "keyword" else None,
                mode=mode,
            )
            selected_ordinals = {clause.ordinal for clause in selected}
            hit = bool(expected & selected_ordinals)
            results[mode]["total"] += 1
            results[mode]["hit"] += 1 if hit else 0
            row[mode] = sorted(selected_ordinals)
            row[f"{mode}_hit"] = hit
        detail_rows.append(row)

    lines = [
        "# 条款检索评测报告（keyword vs semantic vs hybrid）",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        f"文档：{case_data['document']}（{len(parsed.clauses)} 条款） · "
        f"用例：{len(detail_rows)} 条 · embedding：{config.embedding_model}",
        "",
        "| 规则 | 期望条款 | keyword 选中 | semantic 选中 | hybrid 选中 | K 命中 | S 命中 | H 命中 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in detail_rows:
        lines.append(
            f"| {row['rule']} | {row['expected']} | {row['keyword']} | {row['semantic']} "
            f"| {row['hybrid']} | {'✅' if row['keyword_hit'] else '❌'} "
            f"| {'✅' if row['semantic_hit'] else '❌'} | {'✅' if row['hybrid_hit'] else '❌'} |"
        )
    lines.append("")
    lines.append("| 模式 | 选段命中率 |")
    lines.append("|---|---|")
    for mode in MODES:
        stats = results[mode]
        rate = stats["hit"] / stats["total"] if stats["total"] else 0
        lines.append(f"| {mode} | {stats['hit']}/{stats['total']} = {rate:.0%} |")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {REPORT_PATH}")
    for mode in MODES:
        stats = results[mode]
        print(f"  {mode:8} hit {stats['hit']}/{stats['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
