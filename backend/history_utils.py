HISTORY_LIMIT = 6
HISTORY_CONTENT_LIMIT = 1000


def truncate_text(text: str, max_length: int) -> str:
    clean_text = " ".join(str(text or "").split())
    if len(clean_text) <= max_length:
        return clean_text
    return clean_text[:max_length].rstrip() + "..."


def normalize_history(history) -> list[dict]:
    if not isinstance(history, list):
        return []

    normalized = []
    for item in history:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()

        if role not in {"user", "assistant"} or not content:
            continue

        normalized.append({
            "role": role,
            "content": truncate_text(content, HISTORY_CONTENT_LIMIT),
        })

    return normalized[-HISTORY_LIMIT:]


def format_history(history: list[dict]) -> str:
    if not history:
        return ""

    labels = {
        "user": "用户",
        "assistant": "助手",
    }
    return "\n".join(
        f"{labels.get(item['role'], item['role'])}：{item['content']}"
        for item in history
    )


def history_prompt(task: str, text: str, history_context: str) -> str:
    return f"""
请结合最近几轮对话理解指代关系，但必须以当前用户输入为主要任务。

最近对话：
{history_context}

任务：
{task}

当前用户输入：
{text}
"""


def context_prompt(task: str, text: str, context: str, history_context: str | None = None) -> str:
    history_block = ""
    if history_context:
        history_block = f"""
最近对话（仅用于理解指代关系，不要覆盖当前用户输入）：
{history_context}
"""

    return f"""
请优先根据下面的知识库内容完成任务。
如果知识库内容不足以回答，再说明不足之处。

{history_block}

知识库内容：
{context}

任务：
{task}

用户输入：
{text}
"""
