from backend.config import DEFAULT_BASE_URL, DEFAULT_MODEL, SUPPORTED_MODELS, get_config
from backend.history_utils import context_prompt, history_prompt

_default_llm = None


def normalize_model(model: str | None) -> str:
    if model in SUPPORTED_MODELS:
        return model
    return DEFAULT_MODEL


def build_llm(model: str = DEFAULT_MODEL, temperature: float = 0.7, max_tokens: int = 2000):
    from langchain_openai import ChatOpenAI

    config = get_config()
    selected_model = normalize_model(model)

    return ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url or DEFAULT_BASE_URL,
        model=selected_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_default_llm():
    global _default_llm

    if _default_llm is None:
        _default_llm = build_llm()

    return _default_llm


class LazyLLM:
    def invoke(self, *args, **kwargs):
        return get_default_llm().invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_default_llm(), name)


llm = LazyLLM()


def chat(text: str, context=None, custom_llm=None, history_context: str | None = None) -> str:
    active_llm = custom_llm or llm
    prompt = text

    if context:
        prompt = context_prompt("回答用户问题", text, context, history_context)
    elif history_context:
        prompt = history_prompt("回答用户当前问题", text, history_context)

    response = active_llm.invoke(prompt)
    return response.content


def explain(text: str, context=None, custom_llm=None, history_context: str | None = None) -> str:
    active_llm = custom_llm or llm

    if context:
        prompt = context_prompt("请用简单易懂的中文解释用户输入", text, context, history_context)
    elif history_context:
        prompt = history_prompt("请用简单易懂的中文解释当前用户输入", text, history_context)
    else:
        prompt = f"请用简单易懂的中文解释：\n{text}"

    response = active_llm.invoke(prompt)
    return response.content


def summarize(text: str, context=None, custom_llm=None, history_context: str | None = None) -> str:
    active_llm = custom_llm or llm

    if context:
        prompt = context_prompt("请总结与用户输入相关的知识库内容", text, context, history_context)
    elif history_context:
        prompt = history_prompt(
            "请总结当前用户输入；如果当前输入依赖上文，请结合历史对话理解",
            text,
            history_context,
        )
    else:
        prompt = f"请总结以下内容。如果内容很短或像一个主题，请先解释它的含义，再做简短总结：\n{text}"

    response = active_llm.invoke(prompt)
    return response.content


def generate_questions(text: str, context=None, custom_llm=None, history_context: str | None = None) -> str:
    active_llm = custom_llm or llm

    if context:
        prompt = context_prompt("请基于知识库内容出 3 道练习题，并给出答案", text, context, history_context)
    elif history_context:
        prompt = history_prompt(
            "请根据当前用户要求出 3 道练习题，并给出答案；必要时结合历史对话理解主题",
            text,
            history_context,
        )
    else:
        prompt = f"请根据以下主题或知识点出 3 道练习题，并给出答案：\n{text}"

    response = active_llm.invoke(prompt)
    return response.content
