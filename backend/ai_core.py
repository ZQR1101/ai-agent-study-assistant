import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.rag_store import search_relevant_chunks
from backend.schemas import ChatRequest

load_dotenv()

DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"


def build_llm(model: str = DEFAULT_MODEL, temperature: float = 0.7, max_tokens: int = 2000):
    api_key = (
        os.getenv("MY_MIMO_API_KEY")
        or os.getenv("MIMO_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("MIMO_BASE_URL", DEFAULT_BASE_URL),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


llm = build_llm()


def chat(text: str, custom_llm=None) -> str:
    active_llm = custom_llm or llm
    response = active_llm.invoke(text)
    return response.content


def explain(text: str, custom_llm=None) -> str:
    active_llm = custom_llm or llm
    prompt = f"请用简单易懂的中文解释：\n{text}"
    response = active_llm.invoke(prompt)
    return response.content


def summarize(text: str, custom_llm=None) -> str:
    active_llm = custom_llm or llm
    prompt = f"请总结以下内容：\n{text}"
    response = active_llm.invoke(prompt)
    return response.content


def generate_questions(text: str, custom_llm=None) -> str:
    active_llm = custom_llm or llm
    prompt = f"请根据以下知识点出 3 道练习题，并给出答案：\n{text}"
    response = active_llm.invoke(prompt)
    return response.content


def rag_answer_with_sources(question: str, custom_llm=None, top_k: int = 3) -> dict:
    active_llm = custom_llm or llm
    relevant_chunks = search_relevant_chunks(question, top_k=top_k)

    if not relevant_chunks:
        return {
            "answer": "知识库为空，或没有检索到相关内容。请先上传或添加文档。",
            "sources": [],
        }

    relevant_text = ""

    for chunk in relevant_chunks:
        relevant_text += f"""
来源文件：{chunk["source"]}
相似度：{chunk["score"]:.4f}
内容：
{chunk["text"]}

---
"""

    sources = sorted(set(chunk["source"] for chunk in relevant_chunks))

    prompt = f"""
你必须严格根据下面提供的知识回答问题。
如果知识中没有答案，请回答：知识库中没有相关内容。
不要使用你自己的额外知识。

知识：
{relevant_text}

问题：
{question}
"""

    response = active_llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
    }


def rag_answer(question: str, custom_llm=None, top_k: int = 3) -> str:
    result = rag_answer_with_sources(question, custom_llm=custom_llm, top_k=top_k)

    source_text = "\n".join([f"- {source}" for source in result["sources"]])

    return f"""
{result["answer"]}

---

参考来源：
{source_text}
"""


def agent_router(user_input: str, custom_llm=None) -> str:
    active_llm = custom_llm or llm
    router_prompt = f"""
你是一个任务分类器。

请判断用户请求属于哪一类：

1 = explain
2 = summarize
3 = quiz
4 = rag

你只能返回数字，不要解释。

用户请求：
{user_input}
"""

    response = active_llm.invoke(router_prompt)

    choice = response.content.strip()

    if choice.startswith("1"):
        return explain(user_input, custom_llm=active_llm)

    if choice.startswith("2"):
        return summarize(user_input, custom_llm=active_llm)

    if choice.startswith("3"):
        return generate_questions(user_input, custom_llm=active_llm)

    if choice.startswith("4"):
        return rag_answer(user_input, custom_llm=active_llm)

    return "无法判断用户意图。"


def learning_workflow(topic: str, custom_llm=None, top_k: int = 3) -> dict:
    active_llm = custom_llm or llm

    rag_result = rag_answer(topic, custom_llm=active_llm, top_k=top_k)

    summary = summarize(rag_result, custom_llm=active_llm)

    quiz = generate_questions(summary, custom_llm=active_llm)

    advice_prompt = f"""
请根据下面内容，给出简短的下一步学习建议，不超过 3 条：

{summary}
"""

    advice = active_llm.invoke(advice_prompt).content

    return {
        "knowledge": rag_result,
        "summary": summary,
        "quiz": quiz,
        "advice": advice,
    }


def _format_learning_result(result: dict) -> str:
    return (
        f"知识内容：\n{result.get('knowledge', '')}\n\n"
        f"总结：\n{result.get('summary', '')}\n\n"
        f"练习题：\n{result.get('quiz', '')}\n\n"
        f"学习建议：\n{result.get('advice', '')}"
    )


def run_chat_request(request: ChatRequest) -> dict:
    trace = [
        "收到用户请求",
        f"使用模型：{request.model}",
        f"temperature：{request.temperature}",
        f"top_k：{request.top_k}",
    ]
    custom_llm = build_llm(model=request.model, temperature=request.temperature)
    sources = []
    executed_mode = request.mode

    if request.use_rag or request.mode == "rag":
        executed_mode = "rag"
        trace.append("执行模式：rag")
        trace.append("调用 RAG：是")
        result = rag_answer_with_sources(
            request.message,
            custom_llm=custom_llm,
            top_k=request.top_k,
        )
        answer = result["answer"]
        sources = result.get("sources", [])

    elif request.mode == "auto" or request.use_agent:
        executed_mode = "agent"
        trace.append("执行模式：agent_router")
        trace.append("调用 RAG：由 agent_router 决定")
        answer = agent_router(request.message, custom_llm=custom_llm)

    elif request.mode == "chat":
        trace.append("执行模式：chat")
        trace.append("调用 RAG：否")
        answer = chat(request.message, custom_llm=custom_llm)

    elif request.mode == "explain":
        trace.append("执行模式：explain")
        trace.append("调用 RAG：否")
        answer = explain(request.message, custom_llm=custom_llm)

    elif request.mode == "summarize":
        trace.append("执行模式：summarize")
        trace.append("调用 RAG：否")
        answer = summarize(request.message, custom_llm=custom_llm)

    elif request.mode == "quiz":
        trace.append("执行模式：quiz")
        trace.append("调用 RAG：否")
        answer = generate_questions(request.message, custom_llm=custom_llm)

    elif request.mode == "learn":
        trace.append("执行模式：learn")
        trace.append("调用 RAG：是")
        result = learning_workflow(
            request.message,
            custom_llm=custom_llm,
            top_k=request.top_k,
        )
        answer = _format_learning_result(result)

    else:
        executed_mode = "auto"
        trace.append("执行模式：agent_router")
        trace.append("调用 RAG：由 agent_router 决定")
        answer = agent_router(request.message, custom_llm=custom_llm)

    return {
        "answer": answer,
        "mode": executed_mode,
        "model": request.model,
        "sources": sources,
        "trace": trace,
    }
