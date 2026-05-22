import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from backend.rag_store import search_relevant_chunks

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("MY_MIMO_API_KEY"),
    base_url="https://token-plan-cn.xiaomimimo.com/v1",
    model="mimo-v2.5",
    temperature=0.7,
    max_tokens=2000
)


embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def explain(text: str) -> str:
    prompt = f"请用简单易懂的中文解释：\n{text}"
    response = llm.invoke(prompt)
    return response.content


def summarize(text: str) -> str:
    prompt = f"请总结以下内容：\n{text}"
    response = llm.invoke(prompt)
    return response.content


def generate_questions(text: str) -> str:
    prompt = f"请根据以下知识点出3道练习题，并给出答案：\n{text}"
    response = llm.invoke(prompt)
    return response.content


def load_documents() -> list:
    docs_path = Path(__file__).parent.parent / "docs"
    documents = []

    for file_path in docs_path.iterdir():

        if file_path.suffix in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({
                    "source": file_path.name,
                    "text": f.read()
                })

        elif file_path.suffix == ".pdf":
            reader = PdfReader(file_path)
            pdf_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"

            documents.append({
                "source": file_path.name,
                "text": pdf_text
            })

    return documents


def rag_answer_with_sources(question: str) -> dict:
    relevant_chunks = search_relevant_chunks(question, top_k=3)

    if not relevant_chunks:
        return {
            "answer": "知识库为空，请先上传或添加文档。",
            "sources": []
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

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources
    }


def rag_answer(question: str) -> str:
    result = rag_answer_with_sources(question)

    source_text = "\n".join([f"- {source}" for source in result["sources"]])

    return f"""
{result["answer"]}

---

参考来源：
{source_text}
"""


def agent_router(user_input: str) -> str:
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

    response = llm.invoke(router_prompt)

    choice = response.content.strip()

    if choice.startswith("1"):
        return explain(user_input)

    elif choice.startswith("2"):
        return summarize(user_input)

    elif choice.startswith("3"):
        return generate_questions(user_input)

    elif choice.startswith("4"):
        return rag_answer(user_input)

    else:
        return "无法判断用户意图。"


def learning_workflow(topic: str) -> dict:

    rag_result = rag_answer(topic)

    summary = summarize(rag_result)

    quiz = generate_questions(summary)

    advice_prompt = f"""
请根据下面内容，给出简短的下一步学习建议，不超过5条：

{summary}
"""

    advice = llm.invoke(advice_prompt).content

    return {
        "knowledge": rag_result,
        "summary": summary,
        "quiz": quiz,
        "advice": advice
    }