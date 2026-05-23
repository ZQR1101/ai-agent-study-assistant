from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from backend.ai_core import rag_answer, summarize, generate_questions, llm


class LearningState(TypedDict):
    topic: str
    knowledge: str
    summary: str
    quiz: str
    advice: str


def retrieve_knowledge(state: LearningState) -> dict:
    topic = state["topic"]

    knowledge = rag_answer(topic)

    return {
        "knowledge": knowledge
    }


def summarize_knowledge(state: LearningState) -> dict:
    knowledge = state["knowledge"]

    summary = summarize(knowledge)

    return {
        "summary": summary
    }


def generate_quiz(state: LearningState) -> dict:
    summary = state["summary"]

    quiz = generate_questions(summary)

    return {
        "quiz": quiz
    }


def give_advice(state: LearningState) -> dict:
    topic = state["topic"]
    summary = state["summary"]

    advice_prompt = f"""
我正在学习：{topic}

下面是我已经学习到的总结：
{summary}

请用简单中文告诉我下一步应该怎么学。
要求：
1. 不要太长
2. 给出 3 条具体建议
"""

    advice = llm.invoke(advice_prompt).content

    return {
        "advice": advice
    }


graph_builder = StateGraph(LearningState)

graph_builder.add_node("retrieve", retrieve_knowledge)
graph_builder.add_node("summarize", summarize_knowledge)
graph_builder.add_node("quiz", generate_quiz)
graph_builder.add_node("advice", give_advice)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "summarize")
graph_builder.add_edge("summarize", "quiz")
graph_builder.add_edge("quiz", "advice")
graph_builder.add_edge("advice", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "topic": "RAG",
        "knowledge": "",
        "summary": "",
        "quiz": "",
        "advice": ""
    })

    print("\n========== LangGraph 学习工作流结果 ==========\n")

    print("【主题】")
    print(result["topic"])

    print("\n----------------------------------------\n")

    print("【知识讲解】")
    print(result["knowledge"])

    print("\n----------------------------------------\n")

    print("【总结】")
    print(result["summary"])

    print("\n----------------------------------------\n")

    print("【练习题】")
    print(result["quiz"])

    print("\n----------------------------------------\n")

    print("【下一步建议】")
    print(result["advice"])