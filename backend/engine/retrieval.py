"""Deterministic clause retrieval for rule-vs-document scoring.

Selects the document clauses most relevant to a rule before the LLM sees
them. Pure offline text matching — no embeddings, no network — so scoring is
reproducible and fully testable without services.
"""

from __future__ import annotations

import re
from functools import lru_cache

from backend.engine.parsing import ParsedClause

MAX_RULE_CONTEXT_CHARS = 6000
STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "与", "及", "或", "对", "为", "不", "按",
    "检查", "应当", "必须", "条款", "内容", "相关", "情况", "要求", "是否",
    "the", "and", "for", "with", "that", "this", "are", "not", "shall", "must",
}


@lru_cache(maxsize=512)
def _keywords(text: str) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,4}|\d+(?:\.\d+)+", text.lower())
    seen: list[str] = []
    for token in tokens:
        if token in STOPWORDS or len(token) < 2:
            continue
        if token not in seen:
            seen.append(token)
    return tuple(seen[:24])


def score_clause(clause_text: str, query_keywords: tuple[str, ...]) -> int:
    haystack = clause_text.lower()
    score = 0
    for keyword in query_keywords:
        if keyword in haystack:
            score += 2 if len(keyword) >= 3 else 1
    return score


def select_clauses(
    clauses: list[ParsedClause],
    rule_text: str,
    *,
    char_budget: int = MAX_RULE_CONTEXT_CHARS,
    min_top: int = 3,
) -> list[ParsedClause]:
    """Return clauses ranked by keyword overlap, bounded by a char budget.

    Always returns at least ``min_top`` clauses (highest scoring first, ties
    keep document order) so the scorer can judge absence of content.
    """

    keywords = _keywords(rule_text)
    ranked = sorted(
        enumerate(clauses),
        key=lambda pair: (-score_clause(pair[1].text, keywords), pair[0]),
    )
    selected: list[ParsedClause] = []
    used = 0
    for _, clause in ranked:
        cost = len(clause.text)
        if selected and used + cost > char_budget:
            continue
        if not selected and cost > char_budget:
            trimmed = ParsedClause(
                ordinal=clause.ordinal,
                heading=clause.heading,
                text=clause.text[:char_budget],
            )
            selected.append(trimmed)
            used += char_budget
            break
        selected.append(clause)
        used += cost
        if len(selected) >= min_top and used >= char_budget:
            break
    if len(selected) < min_top:
        chosen_ordinals = {c.ordinal for c in selected}
        for _, clause in ranked:
            if clause.ordinal in chosen_ordinals:
                continue
            selected.append(clause)
            chosen_ordinals.add(clause.ordinal)
            if len(selected) >= min_top:
                break
    # Document order for stable prompts.
    return sorted(selected, key=lambda c: c.ordinal)
