import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MESSAGE = "根据知识库解释 agentic rag，生成记忆卡片，并出 3 道题"
DEFAULT_TIMEOUT = 180
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError("expected true/false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy Agent and LangGraph Runtime for the same /chat request.",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="Message to send to both runtimes.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Backend base URL, default: http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--use-rag",
        type=parse_bool,
        default=True,
        help="Whether both requests should enable RAG, true/false. Default: true",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="RAG top_k value for both requests. Default: 3",
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
        help="Save raw responses and summary to outputs/runtime_compare_<timestamp>.json",
    )
    parser.add_argument(
        "--include-llm-planner",
        action="store_true",
        help="Also compare LangGraph with planner_mode=llm. This may consume one extra LLM request.",
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


def build_payload(
    message: str,
    use_rag: bool,
    top_k: int,
    *,
    use_langgraph: bool,
    planner_mode: str = "rule",
) -> dict:
    return {
        "message": message,
        "mode": "auto",
        "model": "mimo-v2.5",
        "temperature": 0.3,
        "use_agent": True,
        "use_rag": use_rag,
        "use_langgraph": use_langgraph,
        "planner_mode": planner_mode,
        "top_k": top_k,
    }


def post_chat(base_url: str, payload: dict, timeout: int) -> dict:
    status_code, body = request_json(base_url.rstrip("/") + "/chat", payload, timeout)

    if status_code != 200:
        raise RuntimeError(f"/chat failed with status={status_code}, response={body}")
    if not isinstance(body, dict):
        raise RuntimeError(f"/chat returned non-object response: {body}")

    return body


def plan_path(response: dict) -> str:
    plan = response.get("plan", [])

    if not isinstance(plan, list) or not plan:
        return "(none)"

    tools = [str(step.get("tool", "unknown")) for step in plan if isinstance(step, dict)]
    return " -> ".join(tools) if tools else "(none)"


def source_keys(response: dict) -> set[str]:
    sources = response.get("sources", [])
    keys = set()

    if not isinstance(sources, list):
        return keys

    for source in sources:
        if isinstance(source, str):
            keys.add(source)
        elif isinstance(source, dict):
            keys.add(str(source.get("source") or source.get("text") or source.get("snippet") or "unknown"))

    return keys


def trace_count(response: dict) -> tuple[int, int]:
    trace = response.get("trace", [])

    if not isinstance(trace, list):
        return 0, 0

    item_count = 0
    for block in trace:
        if isinstance(block, dict) and isinstance(block.get("items"), list):
            item_count += len(block["items"])
        else:
            item_count += 1

    return len(trace), item_count


def answer_length(response: dict) -> int:
    return len(str(response.get("answer") or ""))


def flashcards_count(response: dict) -> int:
    flashcards = response.get("flashcards", [])
    return len(flashcards) if isinstance(flashcards, list) else 0


def has_repeated_flashcard_markdown(response: dict) -> bool:
    answer = str(response.get("answer") or "").lower()
    flashcards = response.get("flashcards", [])

    if not isinstance(flashcards, list) or not flashcards:
        return False

    markers = [
        "## flashcard",
        "## 记忆卡片",
        "### 卡片",
        "**正面",
        "**背面",
        "front/back",
    ]
    return any(marker in answer for marker in markers)


def runtime_info(response: dict) -> dict:
    value = response.get("runtime_info", {})
    return value if isinstance(value, dict) else {}


def summarize_response(label: str, response: dict) -> dict:
    trace_blocks, trace_items = trace_count(response)
    runtime = runtime_info(response)
    tool_calls = runtime.get("tool_calls", [])

    if not isinstance(tool_calls, list):
        tool_calls = []

    return {
        "label": label,
        "mode": response.get("mode"),
        "planner_mode": runtime.get("planner_mode"),
        "planner_fallback": runtime.get("planner_fallback"),
        "planner_error": runtime.get("planner_error"),
        "answer_length": answer_length(response),
        "sources_count": len(source_keys(response)),
        "plan": plan_path(response),
        "flashcards_count": flashcards_count(response),
        "trace_blocks": trace_blocks,
        "trace_items": trace_items,
        "graph_path": runtime.get("graph_path", []),
        "node_count": runtime.get("node_count"),
        "tool_calls": [
            str(call.get("tool", "unknown"))
            for call in tool_calls
            if isinstance(call, dict)
        ],
        "finalizer_used": runtime.get("finalizer_used"),
        "repeated_flashcard_markdown": has_repeated_flashcard_markdown(response),
    }


def print_runtime_summary(title: str, summary: dict, is_langgraph: bool = False) -> None:
    print(f"[{title}]")
    print(f"mode: {summary['mode']}")
    if is_langgraph:
        print(f"planner_mode: {summary.get('planner_mode') or '(unknown)'}")
        print(f"planner_fallback: {summary.get('planner_fallback')}")
        print(f"planner_error: {summary.get('planner_error') or '(none)'}")
    print(f"answer length: {summary['answer_length']}")
    print(f"sources count: {summary['sources_count']}")
    print(f"plan: {summary['plan']}")
    print(f"flashcards count: {summary['flashcards_count']}")

    if is_langgraph:
        graph_path = summary.get("graph_path") or []
        print(f"graph_path: {' -> '.join(graph_path) if graph_path else '(none)'}")
        print(f"node_count: {summary.get('node_count')}")
        tool_calls = summary.get("tool_calls") or []
        print(f"tool_calls: {', '.join(tool_calls) if tool_calls else '(none)'}")
    else:
        print(f"trace blocks/count: {summary['trace_blocks']}/{summary['trace_items']}")

    print("")


def source_overlap(left: dict, right: dict) -> int:
    return len(source_keys(left) & source_keys(right))


def yes_no(value: object) -> str:
    return "yes" if value else "no"


def print_comparison(legacy: dict, langgraph_rule: dict, langgraph_llm: dict | None = None) -> None:
    legacy_sources = source_keys(legacy)
    rule_sources = source_keys(langgraph_rule)

    print("[Comparison]")
    print(f"sources overlap legacy/rule: {len(legacy_sources & rule_sources)}")
    if langgraph_llm is not None:
        llm_sources = source_keys(langgraph_llm)
        print(f"sources overlap legacy/llm: {len(legacy_sources & llm_sources)}")
        print(f"sources overlap rule/llm: {len(rule_sources & llm_sources)}")
    print(f"legacy has repeated flashcard markdown: {yes_no(has_repeated_flashcard_markdown(legacy))}")
    print(f"rule finalizer used: {yes_no(runtime_info(langgraph_rule).get('finalizer_used'))}")
    if langgraph_llm is not None:
        llm_runtime = runtime_info(langgraph_llm)
        print(f"llm finalizer used: {yes_no(llm_runtime.get('finalizer_used'))}")
        print(f"llm planner fallback: {yes_no(llm_runtime.get('planner_fallback'))}")
    both_rule_flashcards = flashcards_count(legacy) > 0 and flashcards_count(langgraph_rule) > 0
    print(f"legacy/rule both returned flashcards: {yes_no(both_rule_flashcards)}")
    if langgraph_llm is not None:
        both_llm_flashcards = flashcards_count(legacy) > 0 and flashcards_count(langgraph_llm) > 0
        print(f"legacy/llm both returned flashcards: {yes_no(both_llm_flashcards)}")


def build_comparison(legacy: dict, langgraph_rule: dict, langgraph_llm: dict | None = None) -> dict:
    comparison = {
        "sources_overlap_legacy_rule": source_overlap(legacy, langgraph_rule),
        "legacy_repeated_flashcard_markdown": has_repeated_flashcard_markdown(legacy),
        "rule_finalizer_used": bool(runtime_info(langgraph_rule).get("finalizer_used")),
        "legacy_rule_both_returned_flashcards": flashcards_count(legacy) > 0 and flashcards_count(langgraph_rule) > 0,
    }

    if langgraph_llm is not None:
        comparison.update({
            "sources_overlap_legacy_llm": source_overlap(legacy, langgraph_llm),
            "sources_overlap_rule_llm": source_overlap(langgraph_rule, langgraph_llm),
            "llm_finalizer_used": bool(runtime_info(langgraph_llm).get("finalizer_used")),
            "llm_planner_fallback": bool(runtime_info(langgraph_llm).get("planner_fallback")),
            "legacy_llm_both_returned_flashcards": flashcards_count(legacy) > 0
            and flashcards_count(langgraph_llm) > 0,
        })

    return comparison


def save_result(payload: dict) -> Path:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"runtime_compare_{timestamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    if not check_backend(base_url, args.timeout):
        return 1

    legacy_payload = build_payload(
        args.message,
        args.use_rag,
        args.top_k,
        use_langgraph=False,
        planner_mode="rule",
    )
    langgraph_rule_payload = build_payload(
        args.message,
        args.use_rag,
        args.top_k,
        use_langgraph=True,
        planner_mode="rule",
    )
    langgraph_llm_payload = None
    if args.include_llm_planner:
        langgraph_llm_payload = build_payload(
            args.message,
            args.use_rag,
            args.top_k,
            use_langgraph=True,
            planner_mode="llm",
        )

    print("=== Runtime Compare ===")
    print(f"Message: {args.message}")
    print("")

    try:
        legacy_response = post_chat(base_url, legacy_payload, args.timeout)
        langgraph_rule_response = post_chat(base_url, langgraph_rule_payload, args.timeout)
        langgraph_llm_response = (
            post_chat(base_url, langgraph_llm_payload, args.timeout)
            if langgraph_llm_payload is not None
            else None
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    legacy_summary = summarize_response("Legacy Agent", legacy_response)
    langgraph_rule_summary = summarize_response("LangGraph Rule Planner", langgraph_rule_response)
    langgraph_llm_summary = (
        summarize_response("LangGraph LLM Planner", langgraph_llm_response)
        if langgraph_llm_response is not None
        else None
    )
    comparison = build_comparison(legacy_response, langgraph_rule_response, langgraph_llm_response)

    print_runtime_summary("Legacy Agent", legacy_summary)
    print_runtime_summary("LangGraph Rule Planner", langgraph_rule_summary, is_langgraph=True)
    if langgraph_llm_summary is not None:
        print_runtime_summary("LangGraph LLM Planner", langgraph_llm_summary, is_langgraph=True)
    print_comparison(legacy_response, langgraph_rule_response, langgraph_llm_response)

    if args.save:
        payload = {
            "message": args.message,
            "base_url": base_url,
            "legacy_payload": legacy_payload,
            "langgraph_payload": langgraph_rule_payload,
            "legacy_summary": legacy_summary,
            "langgraph_summary": langgraph_rule_summary,
            "legacy_response": legacy_response,
            "langgraph_response": langgraph_rule_response,
            "comparison": comparison,
        }
        if langgraph_llm_payload is not None and langgraph_llm_summary is not None:
            payload = {
                "message": args.message,
                "base_url": base_url,
                "legacy": {
                    "payload": legacy_payload,
                    "summary": legacy_summary,
                    "response": legacy_response,
                },
                "langgraph_rule": {
                    "payload": langgraph_rule_payload,
                    "summary": langgraph_rule_summary,
                    "response": langgraph_rule_response,
                },
                "langgraph_llm": {
                    "payload": langgraph_llm_payload,
                    "summary": langgraph_llm_summary,
                    "response": langgraph_llm_response,
                },
                "comparison": comparison,
            }
        output_path = save_result(payload)
        print("")
        print(f"Saved result: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
