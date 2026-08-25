import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 180
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


EVAL_CASES = [
    {
        "id": "LLM-01",
        "message": "什么是 RAG",
        "expected_tools": ["study:explain"],
        "allowed_paths": [["study:explain"]],
        "required_tools": ["study:explain"],
        "forbidden_tools": ["rag_search", "study:flashcard", "study:quiz"],
        "should_use_rag": False,
        "should_generate_flashcards": False,
        "should_generate_quiz": False,
        "notes": "基础概念解释，不应调用 RAG、卡片或出题工具。",
    },
    {
        "id": "LLM-02",
        "message": "请总结 prompt engineering 的核心思想",
        "expected_tools": ["study:summarize"],
        "allowed_paths": [["study:summarize"]],
        "required_tools": ["study:summarize"],
        "forbidden_tools": ["study:flashcard", "study:quiz"],
        "should_use_rag": False,
        "should_generate_flashcards": False,
        "should_generate_quiz": False,
        "notes": "总结意图应优先选择 study summarize，不强制 explain。",
    },
    {
        "id": "LLM-03",
        "message": "根据知识库解释 agentic rag",
        "expected_tools": ["rag_search", "study:explain"],
        "allowed_paths": [["rag_search", "study:explain"]],
        "required_tools": ["rag_search", "study:explain"],
        "forbidden_tools": ["study:flashcard", "study:quiz"],
        "should_use_rag": True,
        "should_generate_flashcards": False,
        "should_generate_quiz": False,
        "notes": "先检索知识库，再基于上下文解释。",
    },
    {
        "id": "LLM-04",
        "message": "根据知识库解释 agentic rag，生成记忆卡片，并出 3 道题",
        "expected_tools": ["rag_search", "study:explain", "study:flashcard", "study:quiz"],
        "allowed_paths": [["rag_search", "study:explain", "study:flashcard", "study:quiz"]],
        "required_tools": ["rag_search", "study:explain", "study:flashcard", "study:quiz"],
        "forbidden_tools": [],
        "should_use_rag": True,
        "should_generate_flashcards": True,
        "should_generate_quiz": True,
        "notes": "典型复合任务，应覆盖 RAG、讲解、卡片和练习题。",
    },
    {
        "id": "LLM-05",
        "message": "帮我生成 RAG 的记忆卡片",
        "expected_tools": ["study:explain", "study:flashcard"],
        "allowed_paths": [["study:flashcard"], ["study:explain", "study:flashcard"]],
        "required_tools": ["study:flashcard"],
        "forbidden_tools": ["study:quiz"],
        "should_use_rag": False,
        "should_generate_flashcards": True,
        "should_generate_quiz": False,
        "notes": "允许只用 study flashcard，或先 explain 再 flashcard。",
    },
    {
        "id": "LLM-06",
        "message": "根据刚才内容出 3 道题",
        "expected_tools": ["study:quiz"],
        "allowed_paths": [["study:quiz"]],
        "required_tools": ["study:quiz"],
        "forbidden_tools": ["study:flashcard"],
        "should_use_rag": False,
        "should_generate_flashcards": False,
        "should_generate_quiz": True,
        "notes": "应结合 history 或 previous context，避免无意义解释。",
    },
    {
        "id": "LLM-07",
        "message": "请解释 RAG，不要出题",
        "expected_tools": ["study:explain"],
        "allowed_paths": [["study:explain"]],
        "required_tools": ["study:explain"],
        "forbidden_tools": ["study:quiz"],
        "should_use_rag": False,
        "should_generate_flashcards": False,
        "should_generate_quiz": False,
        "notes": "必须遵守不要出题，不能调用 study quiz。",
    },
    {
        "id": "LLM-08",
        "message": "不要生成卡片，只解释 agentic rag",
        "expected_tools": ["study:explain"],
        "allowed_paths": [["study:explain"]],
        "required_tools": ["study:explain"],
        "forbidden_tools": ["study:flashcard"],
        "should_use_rag": False,
        "should_generate_flashcards": False,
        "should_generate_quiz": False,
        "notes": "必须遵守不要生成卡片，不能调用 study flashcard。",
    },
    {
        "id": "LLM-09",
        "message": "只根据知识库回答，不要出题",
        "expected_tools": ["rag_search", "study:explain"],
        "allowed_paths": [["rag_search"], ["rag_search", "study:explain"]],
        "required_tools": ["rag_search"],
        "forbidden_tools": ["study:quiz"],
        "should_use_rag": True,
        "should_generate_flashcards": False,
        "should_generate_quiz": False,
        "notes": "必须使用知识库，同时不能调用 study quiz。",
    },
    {
        "id": "LLM-10",
        "message": "请总结这段内容，并生成 3 张复习卡片",
        "expected_tools": ["study:summarize", "study:flashcard"],
        "allowed_paths": [["study:summarize", "study:flashcard"]],
        "required_tools": ["study:summarize", "study:flashcard"],
        "forbidden_tools": ["study:quiz"],
        "should_use_rag": False,
        "should_generate_flashcards": True,
        "should_generate_quiz": False,
        "notes": "先总结，再生成复习卡片。",
    },
    {
        "id": "LLM-11",
        "message": "根据知识库总结 prompt engineering，并生成选择题",
        "expected_tools": ["rag_search", "study:summarize", "study:quiz"],
        "allowed_paths": [["rag_search", "study:summarize", "study:quiz"]],
        "required_tools": ["rag_search", "study:summarize", "study:quiz"],
        "forbidden_tools": ["study:flashcard"],
        "should_use_rag": True,
        "should_generate_flashcards": False,
        "should_generate_quiz": True,
        "notes": "RAG 优先，主要内容是总结，最后生成选择题。",
    },
    {
        "id": "LLM-12",
        "message": "请直接聊天，不要用知识库",
        "expected_tools": ["chat"],
        "allowed_paths": [["chat"], ["study:explain"]],
        "required_tools": [],
        "forbidden_tools": ["rag_search"],
        "should_use_rag": False,
        "should_generate_flashcards": False,
        "should_generate_quiz": False,
        "notes": "必须遵守不要用知识库，不能调用 rag_search。",
    },
    {
        "id": "LLM-13",
        "message": "根据知识库解释一个不存在的概念",
        "expected_tools": ["rag_search", "study:explain"],
        "allowed_paths": [["rag_search"], ["rag_search", "study:explain"]],
        "required_tools": ["rag_search"],
        "forbidden_tools": ["study:flashcard", "study:quiz"],
        "should_use_rag": True,
        "should_generate_flashcards": False,
        "should_generate_quiz": False,
        "notes": "观察 RAG fallback 和回答是否诚实说明知识库未命中。",
    },
    {
        "id": "LLM-14",
        "message": "把刚才那个概念做成卡片",
        "expected_tools": ["study:flashcard"],
        "allowed_paths": [["study:flashcard"]],
        "required_tools": ["study:flashcard"],
        "forbidden_tools": ["study:quiz"],
        "should_use_rag": False,
        "should_generate_flashcards": True,
        "should_generate_quiz": False,
        "notes": "应结合 history，避免重新解释过多。",
    },
    {
        "id": "LLM-15",
        "message": "比较传统 RAG 和 Agentic RAG，并出题",
        "expected_tools": ["study:explain", "study:quiz"],
        "allowed_paths": [["study:explain", "study:quiz"], ["rag_search", "study:explain", "study:quiz"]],
        "required_tools": ["study:explain", "study:quiz"],
        "forbidden_tools": ["study:flashcard"],
        "should_use_rag": False,
        "should_generate_flashcards": False,
        "should_generate_quiz": True,
        "notes": "如果没有要求知识库，RAG 可选；必须包含比较讲解和 quiz。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semi-automatic evaluation cases for LangGraph LLM Planner.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Backend base URL, default: http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--case",
        dest="case_id",
        help="Run a single case by id, for example: LLM-04",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="RAG top_k value for requests. Default: 3",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save evaluation results to outputs/llm_planner_eval_<timestamp>.json",
    )
    return parser.parse_args()


def request_json(url: str, payload: dict | None, timeout: int) -> tuple[int, dict | str]:
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers)

    try:
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:
            return error.code, body
    except urllib.error.URLError as error:
        return 0, f"connection failed: {error.reason}"
    except TimeoutError:
        return 0, "request timed out"
    except json.JSONDecodeError as error:
        return 0, f"invalid json response: {error}"


def check_backend(base_url: str, timeout: int) -> bool:
    status_code, body = request_json(base_url.rstrip("/") + "/health", None, min(timeout, 20))

    if status_code == 200 and isinstance(body, dict) and body.get("status") == "ok":
        return True

    print("Backend is not reachable. Start it with:")
    print("uvicorn backend.server:app --reload")
    print(f"Health status: {status_code}")
    print(f"Health response: {body}")
    return False


def build_payload(case: dict, top_k: int) -> dict:
    return {
        "message": case["message"],
        "mode": "auto",
        "model": "mimo-v2.5",
        "temperature": 0.3,
        "use_agent": True,
        "use_rag": True,
        "use_langgraph": True,
        "planner_mode": "llm",
        "top_k": top_k,
    }


def post_chat(base_url: str, payload: dict, timeout: int) -> dict:
    status_code, body = request_json(base_url.rstrip("/") + "/chat", payload, timeout)

    if status_code != 200:
        raise RuntimeError(f"/chat failed with status={status_code}, response={body}")
    if not isinstance(body, dict):
        raise RuntimeError(f"/chat returned non-object response: {body}")

    return body


def runtime_info(response: dict) -> dict:
    value = response.get("runtime_info", {})
    return value if isinstance(value, dict) else {}


def _tool_label(tool: str, operation: str | None = None) -> str:
    if tool == "rag":
        return "rag_search"
    if tool in {"explain", "summarize", "quiz", "flashcard"}:
        return f"study:{tool}"
    if tool == "study" and operation:
        return f"study:{operation}"
    return tool


def plan_tools(response: dict) -> list[str]:
    plan = response.get("plan", [])
    tools = [
        _tool_label(
            str(step.get("tool")),
            (step.get("arguments") or {}).get("operation") if isinstance(step.get("arguments"), dict) else None,
        )
        for step in plan
        if isinstance(step, dict) and step.get("tool")
    ] if isinstance(plan, list) else []

    if tools:
        return tools

    tool_calls = runtime_info(response).get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []

    return [
        _tool_label(str(call.get("tool")), call.get("operation"))
        for call in tool_calls
        if isinstance(call, dict) and call.get("tool")
    ]


def count_sources(response: dict) -> int:
    sources = response.get("sources", [])
    return len(sources) if isinstance(sources, list) else 0


def count_flashcards(response: dict) -> int:
    flashcards = response.get("flashcards", [])
    return len(flashcards) if isinstance(flashcards, list) else 0


def answer_length(response: dict) -> int:
    return len(str(response.get("answer") or ""))


def tools_label(tools: list[str]) -> str:
    return "->".join(tools) if tools else "(none)"


def truncate(value: str, length: int) -> str:
    if len(value) <= length:
        return value

    return value[: max(0, length - 3)] + "..."


def classify_result(case: dict, actual_tools: list[str], planner_fallback: bool) -> str:
    if planner_fallback:
        return "Fallback"

    forbidden = set(case.get("forbidden_tools", []))
    if forbidden & set(actual_tools):
        return "Incorrect"

    allowed_paths = case.get("allowed_paths", [])
    if any(actual_tools == path for path in allowed_paths):
        return "Correct"

    required = set(case.get("required_tools", []))
    if required and not required.issubset(set(actual_tools)):
        return "Incorrect"

    if required or set(actual_tools) & set(case.get("expected_tools", [])):
        return "Partial"

    return "Incorrect"


def evaluate_case(base_url: str, case: dict, top_k: int, timeout: int) -> dict:
    payload = build_payload(case, top_k)
    response = post_chat(base_url, payload, timeout)
    runtime = runtime_info(response)
    actual_tools = plan_tools(response)
    planner_fallback = bool(runtime.get("planner_fallback"))
    auto_result = classify_result(case, actual_tools, planner_fallback)
    tool_calls = runtime.get("tool_calls", [])

    if not isinstance(tool_calls, list):
        tool_calls = []

    return {
        "id": case["id"],
        "message": case["message"],
        "expected_tools": case["expected_tools"],
        "allowed_paths": case.get("allowed_paths", []),
        "actual_tools": actual_tools,
        "auto_result": auto_result,
        "planner_mode": runtime.get("planner_mode"),
        "planner_fallback": planner_fallback,
        "planner_error": runtime.get("planner_error"),
        "graph_path": runtime.get("graph_path", []),
        "tool_calls": [
            _tool_label(str(call.get("tool", "unknown")), call.get("operation"))
            for call in tool_calls
            if isinstance(call, dict)
        ],
        "sources_count": count_sources(response),
        "flashcards_count": count_flashcards(response),
        "answer_length": answer_length(response),
        "notes": case.get("notes", ""),
        "payload": payload,
    }


def select_cases(case_id: str | None) -> list[dict] | None:
    if not case_id:
        return EVAL_CASES

    normalized = case_id.strip().upper()
    selected = [case for case in EVAL_CASES if case["id"] == normalized]
    if selected:
        return selected

    print(f"Unknown case id: {case_id}")
    print("Available case ids:")
    print(", ".join(case["id"] for case in EVAL_CASES))
    return None


def print_result_table(results: list[dict]) -> None:
    print("ID      Expected                         Actual                           Fallback  Auto")
    for result in results:
        expected = tools_label(result["expected_tools"])
        actual = tools_label(result["actual_tools"])
        fallback = str(result["planner_fallback"]).lower()
        print(
            f"{result['id']:<7} "
            f"{truncate(expected, 32):<32} "
            f"{truncate(actual, 32):<32} "
            f"{fallback:<9} "
            f"{result['auto_result']}"
        )
        planner_error = result.get("planner_error") or "(none)"
        graph_path = tools_label([str(item) for item in result.get("graph_path", [])])
        print(f"        planner_error: {planner_error}")
        print(f"        graph_path: {graph_path}")
        print(
            "        "
            f"sources: {result['sources_count']}, "
            f"flashcards: {result['flashcards_count']}, "
            f"answer_length: {result['answer_length']}"
        )


def summarize_results(results: list[dict]) -> dict:
    counts = {
        "Correct": 0,
        "Partial": 0,
        "Incorrect": 0,
        "Fallback": 0,
    }

    for result in results:
        counts[result["auto_result"]] = counts.get(result["auto_result"], 0) + 1

    total = len(results)
    fallback_rate = (counts["Fallback"] / total * 100) if total else 0.0

    return {
        "total": total,
        **counts,
        "fallback_rate": fallback_rate,
    }


def print_summary(summary: dict) -> None:
    print("")
    print("=== Summary ===")
    print(f"Total: {summary['total']}")
    print(f"Correct: {summary['Correct']}")
    print(f"Partial: {summary['Partial']}")
    print(f"Incorrect: {summary['Incorrect']}")
    print(f"Fallback: {summary['Fallback']}")
    print(f"Fallback rate: {summary['fallback_rate']:.1f}%")


def save_result(payload: dict) -> Path:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"llm_planner_eval_{timestamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    selected_cases = select_cases(args.case_id)
    if selected_cases is None:
        return 1

    base_url = args.base_url.rstrip("/")
    if not check_backend(base_url, args.timeout):
        return 1

    results = []
    try:
        for case in selected_cases:
            results.append(evaluate_case(base_url, case, args.top_k, args.timeout))
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("=== LLM Planner Evaluation ===")
    print(f"Base URL: {base_url}")
    print(f"Cases: {len(results)}")
    print("")
    print_result_table(results)
    summary = summarize_results(results)
    print_summary(summary)

    if args.save:
        output_path = save_result({
            "summary": summary,
            "cases": results,
        })
        print("")
        print(f"Saved result: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
