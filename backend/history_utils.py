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


def memory_prompt_block(memory_context: str) -> str:
    """Wrap memory context as a section block for LLM prompt injection."""
    if not memory_context or not memory_context.strip():
        return ""
    return f"""
=== 个人学习画像 ===
{memory_context.strip()}
"""


def history_prompt(
    task: str,
    text: str,
    history_context: str,
    memory_context: str | None = None,
) -> str:
    sections = []

    if memory_context and memory_context.strip():
        sections.append(memory_prompt_block(memory_context))

    sections.append(f"""
请结合最近几轮对话理解指代关系，但必须以当前用户输入为主要任务。

最近对话：
{history_context}
""")

    sections.append(f"""
任务：
{task}

当前用户输入：
{text}
""")

    return "\n".join(sections)


def context_prompt(
    task: str,
    text: str,
    context: str,
    history_context: str | None = None,
    memory_context: str | None = None,
) -> str:
    sections = []

    if memory_context and memory_context.strip():
        sections.append(memory_prompt_block(memory_context))

    if history_context and history_context.strip():
        sections.append(f"""
最近对话（仅用于理解指代关系，不要覆盖当前用户输入）：
{history_context}
""")

    sections.append(f"""
请优先根据下面的知识库内容完成任务。
如果知识库内容包含答案，必须基于知识库内容回答并只使用其中的信息。
如果知识库内容没有直接包含答案（例如缺少具体的数字、配置、版本号或条款），
必须只回答知识库中没有找到该答案并简要说明缺口，禁止用背景知识补全具体细节。

知识库内容：
{context}
""")

    sections.append(f"""
任务：
{task}

用户输入：
{text}
""")

    return "\n".join(sections)
