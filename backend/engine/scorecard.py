"""Scorecard aggregation: weighted red/amber/green summary."""

from __future__ import annotations

RATING_WEIGHTS = {"red": 3, "amber": 2, "green": 0}


def build_scorecard(
    ratings: list[dict],
    *,
    playbook_name: str,
    dimensions: list[str],
) -> dict:
    """Aggregate per-rule rating rows into a scorecard.

    Each rating row: {"rule_id", "dimension", "name", "rating", "weight",
    "review_state"}. ``score`` = sum(weight * rating_weight) — higher is
    worse; ``risk_index`` normalizes it to 0–100 against the worst possible.
    """

    counts = {"red": 0, "amber": 0, "green": 0}
    dimension_counts: dict[str, dict[str, int]] = {
        dimension: {"red": 0, "amber": 0, "green": 0} for dimension in dimensions
    }
    max_score = 0
    score = 0
    for row in ratings:
        rating = row["rating"]
        if rating not in counts:
            continue
        weight = max(1, int(row.get("weight") or 1))
        counts[rating] += 1
        bucket = dimension_counts.setdefault(row.get("dimension", "未分类"), {"red": 0, "amber": 0, "green": 0})
        bucket[rating] += 1
        max_score += weight * 3
        score += weight * RATING_WEIGHTS[rating]

    total = sum(counts.values())
    risk_index = round(score / max_score * 100) if max_score else 0
    # Coverage = share of rules that did NOT end up red for missing content.
    coverage = round((total - counts["red"]) / total * 100) if total else 0
    return {
        "playbook": playbook_name,
        "total_rules": total,
        "counts": counts,
        "by_dimension": dimension_counts,
        "risk_index": risk_index,
        "coverage_pct": coverage,
    }
