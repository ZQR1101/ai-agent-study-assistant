import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.config import get_model_api_settings, normalize_model


DEFAULT_IMAGE_SIZE = "1024*1024"
IMAGE_TASK_TIMEOUT_SECONDS = 90
IMAGE_TASK_POLL_SECONDS = 3


def _request_json(url: str, *, api_key: str, method: str = "GET", payload: dict | None = None, async_task: bool = False) -> dict:
    body = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if async_task:
        headers["X-DashScope-Async"] = "enable"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope image request failed: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"DashScope image request failed: {exc.reason}") from exc


def _extract_task_id(payload: dict) -> str:
    output = payload.get("output") if isinstance(payload, dict) else None
    task_id = output.get("task_id") if isinstance(output, dict) else None
    if not task_id:
        raise RuntimeError(f"DashScope did not return task_id: {payload}")
    return task_id


def _extract_image_urls(payload: dict) -> list[str]:
    output = payload.get("output") if isinstance(payload, dict) else None
    if not isinstance(output, dict):
        return []

    results = output.get("results") or output.get("images") or []
    urls = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("image_url") or item.get("image")
        if url:
            urls.append(url)

    if output.get("url"):
        urls.append(output["url"])

    return urls


def generate_image(prompt: str, model: str, size: str = DEFAULT_IMAGE_SIZE) -> dict:
    selected_model = normalize_model(model)
    base_url, api_key, api_key_source = get_model_api_settings(selected_model)
    if not api_key:
        raise RuntimeError("Missing DashScope API key. Set DASHSCOPE_API_KEY or ALIBABA_API_KEY.")

    create_url = f"{base_url.rstrip('/')}/services/aigc/text2image/image-synthesis"
    task_payload = {
        "model": selected_model,
        "input": {
            "prompt": prompt,
        },
        "parameters": {
            "size": size,
            "n": 1,
        },
    }
    task = _request_json(create_url, api_key=api_key, method="POST", payload=task_payload, async_task=True)
    task_id = _extract_task_id(task)

    deadline = time.monotonic() + IMAGE_TASK_TIMEOUT_SECONDS
    task_url = f"{base_url.rstrip('/')}/tasks/{task_id}"
    latest = task
    while time.monotonic() < deadline:
        time.sleep(IMAGE_TASK_POLL_SECONDS)
        latest = _request_json(task_url, api_key=api_key)
        output = latest.get("output") if isinstance(latest, dict) else {}
        status = output.get("task_status") if isinstance(output, dict) else None

        if status == "SUCCEEDED":
            urls = _extract_image_urls(latest)
            if not urls:
                raise RuntimeError(f"DashScope image task succeeded without image URL: {latest}")
            return {
                "task_id": task_id,
                "model": selected_model,
                "api_key_source": api_key_source,
                "image_urls": urls,
            }
        if status in {"FAILED", "CANCELED", "UNKNOWN"}:
            message = output.get("message") or output.get("task_metrics") or latest
            raise RuntimeError(f"DashScope image task {status}: {message}")

    raise RuntimeError(f"DashScope image task timed out after {IMAGE_TASK_TIMEOUT_SECONDS}s: {task_id}")
