import argparse
import json
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 120
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


TEST_CASES = {
    "explain": {
        "payload": {
            "message": "什么是 RAG",
            "mode": "explain",
            "model": "mimo-v2.5",
            "temperature": 0.3,
            "use_agent": False,
            "use_rag": False,
            "top_k": 3,
        },
        "checks": [
            ("answer", "truthy"),
            ("mode", "exists"),
            ("trace", "exists"),
        ],
    },
    "rag": {
        "payload": {
            "message": "什么是 prompt engineering",
            "mode": "rag",
            "model": "mimo-v2.5",
            "temperature": 0.3,
            "use_agent": False,
            "use_rag": True,
            "top_k": 3,
        },
        "checks": [
            ("answer", "truthy"),
            ("sources", "exists"),
            ("trace", "exists"),
        ],
    },
    "agent": {
        "payload": {
            "message": "请解释 RAG，并出 3 道练习题",
            "mode": "auto",
            "model": "mimo-v2.5",
            "temperature": 0.3,
            "use_agent": True,
            "use_rag": False,
            "top_k": 3,
        },
        "checks": [
            ("answer", "truthy"),
            ("plan", "exists"),
            ("trace", "exists"),
        ],
    },
    "agentic-rag": {
        "payload": {
            "message": "根据知识库解释 agentic rag，并出 3 道练习题",
            "mode": "auto",
            "model": "mimo-v2.5",
            "temperature": 0.3,
            "use_agent": True,
            "use_rag": True,
            "top_k": 3,
        },
        "checks": [
            ("answer", "truthy"),
            ("sources", "exists"),
            ("plan", "exists"),
            ("trace", "exists"),
        ],
    },
    "flashcard": {
        "payload": {
            "message": "根据知识库生成 RAG 记忆卡片",
            "mode": "auto",
            "model": "mimo-v2.5",
            "temperature": 0.3,
            "use_agent": True,
            "use_rag": True,
            "top_k": 3,
        },
        "checks": [
            ("answer", "truthy"),
            ("flashcards", "exists"),
            ("flashcards", "list"),
            ("trace", "exists"),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test core AI Study Assistant /chat API paths.",
    )
    parser.add_argument(
        "--case",
        choices=["all", *TEST_CASES.keys()],
        default="all",
        help="Which smoke test case to run.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Backend base URL, default: http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SECONDS,
        help="HTTP timeout in seconds.",
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
    health_url = base_url.rstrip("/") + "/health"
    status_code, body = request_json(health_url, None, timeout)

    if status_code == 200 and isinstance(body, dict) and body.get("status") == "ok":
        return True

    probe_results = []
    for path in ["/", "/docs", "/openapi.json"]:
        probe_url = base_url.rstrip("/") + path
        probe_status, probe_body = request_json(probe_url, None, min(timeout, 10))
        probe_results.append((path, probe_status, probe_body))

    print("[ERROR] Backend is not reachable.")
    print(f"        Health check URL: {health_url}")
    print("        Start it first with:")
    print("        uvicorn backend.server:app --reload")
    print(f"        Status code: {status_code}")
    print(f"        Response body: {body}")
    print("        Extra probes:")
    for path, probe_status, probe_body in probe_results:
        print(f"        - {path}: status={probe_status}, body={probe_body}")
    print("        Hint: requests to localhost bypass proxies in this script.")
    print("              If /health still returns 503, restart uvicorn and make sure")
    print("              it is running from this project directory, not an older process.")
    return False


def validate_response(case_name: str, status_code: int, data, checks: list[tuple[str, str]]) -> list[str]:
    failures = []

    if status_code != 200:
        failures.append(f"status_code expected 200, got {status_code}")
        return failures

    if not isinstance(data, dict):
        failures.append("response is not a JSON object")
        return failures

    for field, rule in checks:
        if rule == "exists" and field not in data:
            failures.append(f"missing field: {field}")
        elif rule == "truthy" and not data.get(field):
            failures.append(f"empty or missing field: {field}")
        elif rule == "list" and not isinstance(data.get(field), list):
            failures.append(f"field is not a list: {field}")

    return failures


def run_case(case_name: str, base_url: str, timeout: int) -> bool:
    case = TEST_CASES[case_name]
    status_code, data = request_json(
        base_url.rstrip("/") + "/chat",
        case["payload"],
        timeout,
    )
    failures = validate_response(case_name, status_code, data, case["checks"])

    if failures:
        print(f"[FAIL] {case_name} failed: {'; '.join(failures)}")
        if isinstance(data, dict) and data.get("detail"):
            print(f"       detail: {data['detail']}")
        elif isinstance(data, str):
            print(f"       detail: {data}")
        return False

    print(f"[PASS] {case_name} passed")
    return True


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    if not check_backend(base_url, args.timeout):
        return 1

    case_names = list(TEST_CASES.keys()) if args.case == "all" else [args.case]
    passed = 0
    failed = 0

    for case_name in case_names:
        if run_case(case_name, base_url, args.timeout):
            passed += 1
        else:
            failed += 1

    print("")
    print(f"Summary: passed={passed}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
