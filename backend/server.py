from pathlib import Path
import codecs
import ipaddress
import logging
import secrets
import socket
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request as UrlRequest, build_opener
from typing import Any, Literal

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.config import get_config
from backend.database import get_database_url, get_db_session, init_db, is_db_history_enabled
from backend.evaluation_store import (
    list_recent_judge_results,
    save_judge_result,
    update_judge_feedback,
)
from backend.judge_service import is_judge_persistence_enabled, is_llm_judge_enabled, judge_answer
from backend.schemas import ChatRequest, ChatResponse, JudgeFeedbackRequest
from backend.run_metadata import build_run_metadata
from backend.run_repository import Run, get_run_repository
from backend.session_store import (
    create_or_get_session,
    get_recent_messages,
    get_session_messages,
    list_sessions,
    save_message,
)
from backend.tool_registry import InvalidConfirmation, ToolConfirmationRequired
from backend.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs"
SUPPORTED_DOC_EXTENSIONS = {".md", ".txt", ".pdf"}
IMAGE_PROXY_MAX_REDIRECTS = 3
UPLOAD_CHUNK_SIZE = 64 * 1024
ALLOWED_UPLOAD_CONTENT_TYPES = {
    ".pdf": {"application/pdf"},
    ".md": {"text/markdown", "text/plain"},
    ".txt": {"text/plain"},
}

app = FastAPI(
    title="AI 学习助手 API",
    openapi_tags=[
        {"name": "Chat", "description": "统一聊天与 Agent 入口。"},
        {"name": "Tools", "description": "Tool Registry 查询、调用和审计。"},
        {"name": "Runs", "description": "统一执行记录，供历史、回放、比较和导出使用。"},
        {"name": "Knowledge Base", "description": "知识文件与 RAG 状态管理。"},
        {"name": "Sessions", "description": "会话历史。"},
        {"name": "Evaluation", "description": "LLM-as-Judge 结果与反馈。"},
        {"name": "System", "description": "服务状态。"},
    ],
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_config().cors_allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Tool-Approval-Key"],
)


@app.on_event("startup")
def _startup_init_db():
    """Initialize database tables on startup when a database is configured."""
    if get_database_url() and (is_db_history_enabled() or is_judge_persistence_enabled()):
        try:
            init_db()
            logger.info("Database tables initialized")
        except Exception as exc:
            logger.warning("Failed to initialize database: %s", exc)

    config = get_config()
    if config.enable_rag_warmup:
        from backend.rag_warmup import start_rag_warmup

        result = start_rag_warmup(load_index=config.rag_warmup_load_index)
        logger.info("RAG warmup startup trigger: started=%s", result["started"])


class TextRequest(BaseModel):
    text: str


class RagRequest(BaseModel):
    text: str
    top_k: int = 3
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "vector"
    reranker_enabled: bool = False


class DebugRagRequest(BaseModel):
    text: str
    top_k: int = 5
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "vector"
    reranker_enabled: bool = False


class ToolInvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirmation_token: str | None = None
    actor: str = "api"


def _is_public_image_url(url: str) -> bool:
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return False

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False

    for item in addresses:
        host = item[4][0]
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        if address.is_private or address.is_loopback or address.is_link_local or address.is_multicast:
            return False

    return True


class _NoImageRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _build_image_opener():
    return build_opener(_NoImageRedirectHandler())


def _open_public_image(url: str):
    opener = _build_image_opener()
    current_url = url

    for redirect_count in range(IMAGE_PROXY_MAX_REDIRECTS + 1):
        if not _is_public_image_url(current_url):
            raise HTTPException(status_code=400, detail="Unsupported image URL")

        request = UrlRequest(
            current_url,
            headers={"User-Agent": "AI-Study-Assistant/1.0"},
        )
        try:
            return opener.open(request, timeout=20)
        except HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            location = exc.headers.get("Location") if exc.headers else None
            exc.close()
            if not location or redirect_count >= IMAGE_PROXY_MAX_REDIRECTS:
                raise HTTPException(
                    status_code=400,
                    detail="Image redirect is invalid or exceeds the redirect limit",
                ) from exc
            current_url = urljoin(current_url, location)

    raise HTTPException(status_code=400, detail="Image redirect limit exceeded")


def _safe_doc_path(filename: str) -> Path:
    clean_name = Path(filename).name

    if clean_name != filename:
        raise HTTPException(status_code=400, detail="Invalid file name")

    file_path = (DOCS_PATH / clean_name).resolve()
    docs_root = DOCS_PATH.resolve()

    if docs_root not in file_path.parents or file_path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return file_path


@app.get("/", tags=["System"])
def home():
    return {"message": "AI 学习助手后端启动成功"}


@app.get("/health", tags=["System"])
def health_check():
    from backend.rag_store import get_rag_index_status
    from backend.rag_warmup import get_rag_warmup_status

    config = get_config()
    rag_warmup_status = get_rag_warmup_status()
    return {
        "status": "ok",
        "config": {
            "model": config.model,
            "base_url": config.base_url,
            "has_api_key": config.has_api_key,
            "api_key_source": config.api_key_source,
            "embedding_model": config.embedding_model,
            "embedding_model_local_only": config.embedding_model_local_only,
        },
        "rag_index": get_rag_index_status(),
        "rag_warmup": {
            "enabled": config.enable_rag_warmup,
            "status": rag_warmup_status["status"],
        },
        "db_history_enabled": is_db_history_enabled(),
        "database_configured": get_database_url() is not None,
        "llm_judge_enabled": is_llm_judge_enabled(),
        "judge_persistence_enabled": is_judge_persistence_enabled(),
    }


@app.get("/tools", tags=["Tools"])
def list_tools_api():
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "category": spec.category.value,
                "requires_confirmation": spec.requires_confirmation,
                "agent_visible": spec.agent_visible,
            }
            for spec in TOOL_REGISTRY.values()
        ]
    }


def _require_tool_approval(tool_name: str, approval_key: str | None) -> None:
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None or not spec.requires_confirmation:
        return

    configured_key = get_config().tool_approval_key
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="Dangerous tool approval is not configured",
        )
    if not approval_key or not secrets.compare_digest(approval_key, configured_key):
        raise HTTPException(status_code=403, detail="Dangerous tool approval denied")


@app.post(
    "/tools/{tool_name}/invoke",
    tags=["Tools"],
    responses={
        409: {"description": "Dangerous tool requires confirmation"},
        400: {"description": "Confirmation token is invalid or expired"},
        403: {"description": "Dangerous tool approval denied"},
        404: {"description": "Tool not found"},
        503: {"description": "Dangerous tool approval is not configured"},
    },
)
def invoke_tool_api(
    tool_name: str,
    request: ToolInvokeRequest,
    approval_key: str | None = Header(default=None, alias="X-Tool-Approval-Key"),
):
    arguments = dict(request.arguments)
    for reserved in ("actor", "confirmed", "confirmation_token"):
        arguments.pop(reserved, None)
    _require_tool_approval(tool_name, approval_key)
    try:
        return TOOL_REGISTRY.execute(
            tool_name,
            confirmation_token=request.confirmation_token,
            actor=request.actor,
            **arguments,
        )
    except ToolConfirmationRequired as exc:
        raise HTTPException(status_code=409, detail=exc.as_dict()) from exc
    except InvalidConfirmation as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/tools/audit/recent", tags=["Tools"])
def recent_tool_audit_api(limit: int = Query(100, ge=1, le=1000)):
    return {"events": TOOL_REGISTRY.audit_log.recent(limit)}


def _run_payload(run: Run) -> dict:
    return run.model_dump(mode="json") if hasattr(run, "model_dump") else run.dict()


@app.get("/runs", tags=["Runs"])
def list_runs_api(
    limit: int = Query(50, ge=1, le=500),
    status: str | None = None,
    session_id: str | None = None,
    include_deleted: bool = False,
):
    runs = get_run_repository().list_runs(
        limit=limit,
        status=status,
        session_id=session_id,
        include_deleted=include_deleted,
    )
    return {"runs": [_run_payload(run) for run in runs], "count": len(runs)}


@app.get("/runs/{run_id}", tags=["Runs"])
def get_run_api(run_id: str):
    try:
        run = get_run_repository().get_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_payload(run)


@app.get("/rag/status", tags=["Knowledge Base"])
def rag_status_api():
    from backend.rag_warmup import get_rag_warmup_status

    return get_rag_warmup_status()


@app.post("/rag/warmup", tags=["Knowledge Base"])
def rag_warmup_api():
    from backend.rag_warmup import start_rag_warmup

    config = get_config()
    return start_rag_warmup(load_index=config.rag_warmup_load_index)


@app.post("/echo", include_in_schema=False)
def echo_api(request: TextRequest):
    return {"echo": request.text}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_api(request: ChatRequest):
    from backend.ai_core import run_chat_request

    request_started_at = perf_counter()
    session_id = request.session_id
    db_history_error = None
    judge_error = None
    judge_persistence_error = None
    run_repository = get_run_repository()

    # --- DB history: load context ---
    if is_db_history_enabled():
        try:
            with get_db_session() as db:
                session_id = create_or_get_session(
                    db, request.session_id, title=request.message
                )
                db_messages = get_recent_messages(db, session_id, limit=10)
                if db_messages and not request.history:
                    request = request.model_copy(update={"history": db_messages, "session_id": session_id})
                elif not request.history:
                    request = request.model_copy(update={"session_id": session_id})
                else:
                    request = request.model_copy(update={"session_id": session_id})
        except Exception as exc:
            db_history_error = f"db_load_error: {exc}"
            logger.warning("DB history load failed: %s", exc)

    request_payload = (
        request.model_dump(mode="json", exclude={"run_id"})
        if hasattr(request, "model_dump")
        else request.dict(exclude={"run_id"})
    )
    run = run_repository.create_run(
        request=request_payload,
        session_id=session_id,
        metadata={"entrypoint": "/chat"},
    )
    run_id = run.id
    request = request.model_copy(update={"run_id": run_id, "session_id": session_id})

    # --- Execute chat ---
    try:
        result = run_chat_request(request)
    except Exception as exc:
        run_repository.finish_run(run_id, status="failed", error=str(exc))
        raise
    result["run_id"] = run_id

    # --- LLM-as-Judge evaluation ---
    if is_llm_judge_enabled() and result.get("answer"):
        try:
            evaluation = judge_answer(
                request.message,
                result["answer"],
                sources=result.get("sources", []),
                trace=result.get("trace", []),
                runtime_info=result.get("runtime_info", {}),
                model=result.get("model") or request.model,
            )
            result["judge_evaluation"] = {
                **evaluation,
                "session_id": session_id,
                "run_id": run_id,
                "question": request.message,
                "answer": result["answer"],
            }
        except Exception as exc:
            judge_error = f"judge_error: {exc}"
            logger.warning("LLM judge evaluation failed: %s", exc)

    # --- Judge evaluation persistence ---
    if result.get("judge_evaluation") and get_database_url() and is_judge_persistence_enabled():
        try:
            with get_db_session() as db:
                persisted_evaluation = save_judge_result(
                    db,
                    session_id=session_id,
                    question=request.message,
                    answer=result["answer"],
                    evaluation=result["judge_evaluation"],
                    run_id=run_id,
                )
                result["judge_evaluation"] = {
                    **persisted_evaluation,
                    "run_id": run_id,
                }
        except Exception as exc:
            judge_persistence_error = f"judge_db_save_error: {exc}"
            logger.warning("Judge evaluation save failed: %s", exc)

    # --- Attach session_id and db error to response ---
    if session_id:
        result["session_id"] = session_id
    if db_history_error:
        runtime_info = result.get("runtime_info", {})
        runtime_info["db_history_error"] = db_history_error
        result["runtime_info"] = runtime_info
    if judge_error or judge_persistence_error:
        runtime_info = result.get("runtime_info", {})
        if judge_error:
            runtime_info["judge_error"] = judge_error
        if judge_persistence_error:
            runtime_info["judge_persistence_error"] = judge_persistence_error
        result["runtime_info"] = runtime_info

    run_summary, run_details = build_run_metadata(
        result,
        duration_ms=max(0, round((perf_counter() - request_started_at) * 1000)),
    )
    result["run_summary"] = run_summary
    result["run_details"] = run_details

    # --- DB history: save messages and the complete assistant response ---
    if is_db_history_enabled() and session_id:
        try:
            response_snapshot = ChatResponse.model_validate(result).model_dump(mode="json")
            with get_db_session() as db:
                save_message(db, session_id, "user", request.message)
                if result.get("answer"):
                    save_message(
                        db,
                        session_id,
                        "assistant",
                        result["answer"],
                        response=response_snapshot,
                    )
        except Exception as exc:
            db_history_error = f"db_save_error: {exc}"
            runtime_info = dict(result.get("runtime_info", {}))
            runtime_info["db_history_error"] = db_history_error
            result["runtime_info"] = runtime_info
            logger.warning("DB history save failed: %s", exc)

    run_repository.update_run(
        run_id,
        session_id=session_id,
        plan=result.get("plan", []),
        tools=result.get("runtime_info", {}).get("tool_calls", []),
        artifacts={
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "flashcards": result.get("flashcards", []),
            "trace": result.get("trace", []),
        },
        metadata={"run_summary": result.get("run_summary", {})},
    )
    finish_status = {
        "failed": "failed",
        "partial": "partial",
    }.get(run_summary.get("status"), "completed")
    response_payload = ChatResponse.model_validate(result).model_dump(mode="json")
    run_repository.finish_run(
        run_id,
        status=finish_status,
        output=response_payload,
        error=result.get("runtime_info", {}).get("error"),
    )

    return result


def _recent_judge_results_payload(limit: int) -> dict:
    """Return recent LLM-as-Judge scores, newest first."""
    if not get_database_url():
        return {
            "results": [],
            "evaluations": [],
            "message": "Database is not configured; judge results are not persisted",
        }
    if not is_judge_persistence_enabled():
        return {
            "results": [],
            "evaluations": [],
            "message": "Judge persistence is disabled (ENABLE_JUDGE_PERSISTENCE=false)",
        }

    try:
        with get_db_session() as db:
            results = list_recent_judge_results(db, limit=limit)
        return {"results": results, "evaluations": results}
    except Exception as exc:
        logger.warning("Failed to list judge results: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@app.get("/judge-results/recent", tags=["Evaluation"])
def recent_judge_results_api(limit: int = Query(20, ge=1, le=100)):
    """Return recent LLM-as-Judge results, newest first."""
    return _recent_judge_results_payload(limit)


@app.get("/judge-evaluations/recent", include_in_schema=False, deprecated=True)
def recent_judge_evaluations_api(limit: int = Query(20, ge=1, le=100)):
    """Backward-compatible alias for recent judge results."""
    return _recent_judge_results_payload(limit)


@app.post("/judge-results/{result_id}/feedback", tags=["Evaluation"])
def judge_result_feedback_api(result_id: int, request: JudgeFeedbackRequest):
    """Persist human feedback about whether a judge result was reasonable."""
    if not get_database_url():
        raise HTTPException(status_code=503, detail="Database is not configured")
    if not is_judge_persistence_enabled():
        raise HTTPException(status_code=503, detail="Judge persistence is disabled")

    try:
        with get_db_session() as db:
            updated = update_judge_feedback(
                db,
                result_id=result_id,
                judge_feedback=request.judge_feedback,
                reason=request.reason,
            )
        if updated is None:
            raise HTTPException(status_code=404, detail="Judge result not found")
        return {"result": updated}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed to save judge feedback: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@app.get("/image-proxy", include_in_schema=False)
def image_proxy(url: str = Query(..., min_length=1)):
    try:
        with _open_public_image(url) as response:
            content_type = response.headers.get("content-type", "image/png").split(";")[0].strip()
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="URL did not return an image")
            content = response.read(20 * 1024 * 1024 + 1)
    except HTTPException:
        raise
    except HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail="Image fetch failed") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {exc.reason}") from exc

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large")

    return Response(content=content, media_type=content_type)


@app.post("/debug-langgraph", include_in_schema=False)
def debug_langgraph(request: ChatRequest):
    try:
        from backend.langgraph_runtime import LangGraphRuntimeUnavailableError, run_langgraph_workflow

        return run_langgraph_workflow(
            request.message,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            reranker_enabled=request.reranker_enabled,
            use_rag=request.use_rag,
            model=request.model,
            temperature=request.temperature,
            history=request.history,
        )
    except LangGraphRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/explain", include_in_schema=False, deprecated=True)
def explain_api(request: TextRequest):
    result = TOOL_REGISTRY.execute(
        "study", step_input=request.text, operation="explain", actor="api"
    )
    return {"result": result["answer"]}


@app.post("/summarize", include_in_schema=False, deprecated=True)
def summarize_api(request: TextRequest):
    result = TOOL_REGISTRY.execute(
        "study", step_input=request.text, operation="summarize", actor="api"
    )
    return {"result": result["answer"]}


@app.post("/quiz", include_in_schema=False, deprecated=True)
def quiz_api(request: TextRequest):
    result = TOOL_REGISTRY.execute(
        "study", step_input=request.text, operation="quiz", actor="api"
    )
    return {"result": result["answer"]}


@app.post("/rag", include_in_schema=False, deprecated=True)
def rag_api(request: RagRequest):
    return TOOL_REGISTRY.execute(
        "rag_search",
        step_input=request.text,
        top_k=request.top_k,
        shared_context={
            "retrieval_mode": request.retrieval_mode,
            "reranker_enabled": request.reranker_enabled,
        },
        actor="api",
    )


@app.post("/agent", include_in_schema=False, deprecated=True)
def agent_api(request: TextRequest):
    from backend.ai_core import agent_router

    return {"result": agent_router(request.text)}


@app.post("/upload", tags=["Knowledge Base"])
async def upload_file(file: UploadFile = File(...)):
    original_filename = file.filename or ""
    filename = Path(original_filename).name
    if not filename or filename != original_filename:
        raise HTTPException(status_code=400, detail="Invalid file name")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOC_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file extension")

    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in ALLOWED_UPLOAD_CONTENT_TYPES[suffix]:
        raise HTTPException(
            status_code=415,
            detail=f"Content-Type {content_type or '<missing>'} is not allowed for {suffix}",
        )

    DOCS_PATH.mkdir(parents=True, exist_ok=True)
    save_path = DOCS_PATH / filename
    max_size = get_config().max_upload_size_bytes
    total_size = 0
    header = b""
    created = False
    text_decoder = (
        codecs.getincrementaldecoder("utf-8")()
        if suffix in {".md", ".txt"}
        else None
    )

    try:
        with save_path.open("xb") as handle:
            created = True
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {max_size} byte upload limit",
                    )
                if len(header) < 5:
                    header = (header + chunk)[:5]
                if text_decoder is not None:
                    try:
                        text_decoder.decode(chunk, final=False)
                    except UnicodeDecodeError as exc:
                        raise HTTPException(
                            status_code=415,
                            detail="Text uploads must be valid UTF-8",
                        ) from exc
                handle.write(chunk)

            if text_decoder is not None:
                try:
                    text_decoder.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    raise HTTPException(
                        status_code=415,
                        detail="Text uploads must be valid UTF-8",
                    ) from exc
            if suffix == ".pdf" and not header.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=415,
                    detail="Uploaded PDF does not have a valid PDF signature",
                )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="A file with this name already exists") from exc
    except HTTPException:
        if created:
            save_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if created:
            save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to store uploaded file") from exc
    finally:
        await file.close()

    return {
        "message": f"{filename} uploaded successfully; rebuild the RAG index to use it",
        "rebuild_required": True,
        "size": total_size,
    }


@app.get("/knowledge-files", tags=["Knowledge Base"])
def knowledge_files_api():
    DOCS_PATH.mkdir(exist_ok=True)
    files = []

    for file_path in sorted(DOCS_PATH.iterdir(), key=lambda item: item.name.lower()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
            continue

        files.append(
            {
                "name": file_path.name,
                "type": file_path.suffix.lower().lstrip("."),
                "size": file_path.stat().st_size,
                "url": f"/knowledge-files/{quote(file_path.name)}",
            }
        )

    return {"count": len(files), "files": files}


@app.get("/knowledge-files/{filename}", tags=["Knowledge Base"])
def open_knowledge_file_api(filename: str):
    file_path = _safe_doc_path(filename)
    encoded_name = quote(file_path.name)
    media_types = {
        ".md": "text/plain; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".pdf": "application/pdf",
    }
    return FileResponse(
        file_path,
        media_type=media_types.get(file_path.suffix.lower(), "application/octet-stream"),
        headers={
            "Content-Disposition": f"inline; filename*=utf-8''{encoded_name}",
        },
    )


@app.get("/knowledge-files/{filename}/content", tags=["Knowledge Base"])
def read_knowledge_file_content_api(filename: str):
    file_path = _safe_doc_path(filename)

    if file_path.suffix.lower() not in {".md", ".txt"}:
        raise HTTPException(status_code=400, detail="Only md/txt files support text preview")

    return {
        "name": file_path.name,
        "type": file_path.suffix.lower().lstrip("."),
        "content": file_path.read_text(encoding="utf-8"),
    }


@app.post("/learn", include_in_schema=False, deprecated=True)
def learn_api(request: TextRequest):
    from backend.ai_core import learning_workflow

    return learning_workflow(request.text)


@app.post("/rebuild-index", include_in_schema=False, deprecated=True)
def rebuild_index_api(
    request: ToolInvokeRequest | None = None,
    approval_key: str | None = Header(default=None, alias="X-Tool-Approval-Key"),
):
    request = request or ToolInvokeRequest()
    _require_tool_approval("rebuild_rag_index", approval_key)
    try:
        TOOL_REGISTRY.execute(
            "rebuild_rag_index",
            confirmation_token=request.confirmation_token,
            actor=request.actor,
        )
    except ToolConfirmationRequired as exc:
        raise HTTPException(status_code=409, detail=exc.as_dict()) from exc
    except InvalidConfirmation as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "RAG 索引已重建"}


@app.get("/debug-index-sources", include_in_schema=False)
def debug_index_sources_api():
    from backend.rag_store import list_index_sources

    sources = list_index_sources()
    return {
        "count": len(sources),
        "sources": sources,
    }


@app.post("/debug-rag", include_in_schema=False)
def debug_rag_api(request: DebugRagRequest):
    from backend.rag_store import search_relevant_chunks

    result = search_relevant_chunks(
        request.text,
        top_k=request.top_k,
        retrieval_mode=request.retrieval_mode,
        reranker_enabled=request.reranker_enabled,
        include_metadata=True,
    )
    chunks = result["chunks"]

    return {
        "question": request.text,
        "retrieval_mode": result.get("retrieval_mode", request.retrieval_mode),
        "reranker_enabled": result.get("reranker_enabled", request.reranker_enabled),
        "reranker_used": result.get("reranker_used", False),
        "reranker_model": result.get("reranker_model"),
        "reranker_top_n": result.get("reranker_top_n"),
        "reranker_error": result.get("reranker_error"),
        "count": len(chunks),
        "chunks": chunks,
    }


# --- Session history APIs ---


@app.get("/sessions", tags=["Sessions"])
def list_sessions_api(limit: int = Query(50, ge=1, le=200)):
    """List recent chat sessions."""
    if not is_db_history_enabled():
        return {"sessions": [], "message": "DB history is disabled (ENABLE_DB_HISTORY=false)"}

    try:
        with get_db_session() as db:
            sessions = list_sessions(db, limit=limit)
        return {"sessions": sessions}
    except Exception as exc:
        logger.warning("Failed to list sessions: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@app.get("/sessions/{session_id}/messages", tags=["Sessions"])
def get_session_messages_api(session_id: str, limit: int = Query(50, ge=1, le=500)):
    """Get messages for a specific session."""
    if not is_db_history_enabled():
        return {
            "session_id": session_id,
            "messages": [],
            "message": "DB history is disabled (ENABLE_DB_HISTORY=false)",
        }

    try:
        with get_db_session() as db:
            messages = get_session_messages(db, session_id, limit=limit)
        return {"session_id": session_id, "messages": messages}
    except Exception as exc:
        logger.warning("Failed to get session messages: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
