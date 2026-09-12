"""Document Q&A agent: natural-language questions over one reviewed document.

Answers are grounded ONLY in the selected clauses; when the document does not
contain the answer the agent says so instead of improvising — same refusal
philosophy as scoring.
"""

from __future__ import annotations

from backend.engine.parsing import ParsedClause


def _build_prompt(question: str, clauses: list[ParsedClause]) -> str:
    clause_blocks = "\n\n".join(
        f"[条款 {clause.ordinal}]{(' ' + clause.heading) if clause.heading else ''}\n{clause.text}"
        for clause in clauses
    )
    return f"""你是文档问答代理。仅依据以下条款内容回答用户问题；引用信息时注明条款编号（如 [条款 3]）。
如果条款内容不足以回答，直接回答"文档中没有足够的信息回答该问题"，不要编造。

## 文档条款
{clause_blocks}

## 用户问题
{question}

## 输出
用中文简洁回答（≤300字），引用处标注条款编号。"""


def answer_question(question: str, clauses: list[ParsedClause], *, custom_llm=None) -> dict:
    if not clauses:
        return {"answer": "文档中没有足够的信息回答该问题。", "clauses_used": 0}

    from backend.engine.scoring import _llm_invoke

    prompt = _build_prompt(question, clauses)
    raw = _llm_invoke(custom_llm, prompt)
    answer = (raw or "").strip() or "文档中没有足够的信息回答该问题。"
    return {"answer": answer, "clauses_used": len(clauses)}
