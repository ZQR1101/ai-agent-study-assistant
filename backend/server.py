from pathlib import Path
import ipaddress
import logging
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from backend.config import get_config
from backend.database import get_database_url, get_db_session, init_db, is_db_history_enabled
from backend.evaluation_store import (
    list_recent_judge_results,
    save_judge_result,
    update_judge_feedback,
)
from backend.judge_service import is_judge_persistence_enabled, is_llm_judge_enabled, judge_answer
from backend.schemas import ChatRequest, ChatResponse, JudgeFeedbackRequest
from backend.session_store import (
    create_or_get_session,
    get_recent_messages,
    get_session_messages,
    list_sessions,
    save_message,
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs"
SUPPORTED_DOC_EXTENSIONS = {".md", ".txt", ".pdf"}

app = FastAPI(title="AI 学习助手 API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _is_public_image_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
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


@app.get("/")
def home():
    return {"message": "AI 学习助手后端启动成功"}


@app.get("/health")
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


@app.get("/rag/status")
def rag_status_api():
    from backend.rag_warmup import get_rag_warmup_status

    return get_rag_warmup_status()


@app.post("/rag/warmup")
def rag_warmup_api():
    from backend.rag_warmup import start_rag_warmup

    config = get_config()
    return start_rag_warmup(load_index=config.rag_warmup_load_index)


@app.post("/echo")
def echo_api(request: TextRequest):
    return {"echo": request.text}


@app.post("/chat", response_model=ChatResponse)
def chat_api(request: ChatRequest):
    from backend.ai_core import run_chat_request

    session_id = request.session_id
    db_history_error = None
    judge_error = None
    judge_persistence_error = None

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

    # --- Execute chat ---
    result = run_chat_request(request)

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
                "question": request.message,
                "answer": result["answer"],
            }
        except Exception as exc:
            judge_error = f"judge_error: {exc}"
            logger.warning("LLM judge evaluation failed: %s", exc)

    # --- DB history: save messages ---
    if is_db_history_enabled() and session_id:
        try:
            with get_db_session() as db:
                save_message(db, session_id, "user", request.message)
                if result.get("answer"):
                    save_message(db, session_id, "assistant", result["answer"])
        except Exception as exc:
            db_history_error = f"db_save_error: {exc}"
            logger.warning("DB history save failed: %s", exc)

    # --- Judge evaluation persistence ---
    if result.get("judge_evaluation") and get_database_url() and is_judge_persistence_enabled():
        try:
            with get_db_session() as db:
                result["judge_evaluation"] = save_judge_result(
                    db,
                    session_id=session_id,
                    question=request.message,
                    answer=result["answer"],
                    evaluation=result["judge_evaluation"],
                )
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


@app.get("/judge-results/recent")
def recent_judge_results_api(limit: int = Query(20, ge=1, le=100)):
    """Return recent LLM-as-Judge results, newest first."""
    return _recent_judge_results_payload(limit)


@app.get("/judge-evaluations/recent")
def recent_judge_evaluations_api(limit: int = Query(20, ge=1, le=100)):
    """Backward-compatible alias for recent judge results."""
    return _recent_judge_results_payload(limit)


@app.post("/judge-results/{result_id}/feedback")
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


@app.get("/image-proxy")
def image_proxy(url: str = Query(..., min_length=1)):
    if not _is_public_image_url(url):
        raise HTTPException(status_code=400, detail="Unsupported image URL")

    request = UrlRequest(url, headers={"User-Agent": "AI-Study-Assistant/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
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


@app.post("/debug-langgraph")
def debug_langgraph(request: ChatRequest):
    try:
        from backend.langgraph_runtime import LangGraphRuntimeUnavailableError, run_langgraph_workflow

        return run_langgraph_workflow(
            request.message,
            top_k=request.top_k,
            use_rag=request.use_rag,
            model=request.model,
            temperature=request.temperature,
            history=request.history,
        )
    except LangGraphRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/explain")
def explain_api(request: TextRequest):
    from backend.ai_core import explain

    result = explain(request.text)
    return {"result": result}


@app.post("/summarize")
def summarize_api(request: TextRequest):
    from backend.ai_core import summarize

    result = summarize(request.text)
    return {"result": result}


@app.post("/quiz")
def quiz_api(request: TextRequest):
    from backend.ai_core import generate_questions

    result = generate_questions(request.text)
    return {"result": result}


@app.post("/rag")
def rag_api(request: TextRequest):
    from backend.ai_core import rag_answer_with_sources

    return rag_answer_with_sources(request.text)


@app.post("/agent")
def agent_api(request: TextRequest):
    from backend.ai_core import agent_router

    return {"result": agent_router(request.text)}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    from backend.rag_store import rebuild_rag_index

    DOCS_PATH.mkdir(exist_ok=True)
    filename = Path(file.filename or "").name

    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    save_path = DOCS_PATH / filename

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    rebuild_rag_index()

    return {"message": f"{filename} 上传成功，知识库索引已更新"}


@app.get("/knowledge-files")
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


@app.get("/knowledge-files/{filename}")
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


@app.get("/knowledge-files/{filename}/content")
def read_knowledge_file_content_api(filename: str):
    file_path = _safe_doc_path(filename)

    if file_path.suffix.lower() not in {".md", ".txt"}:
        raise HTTPException(status_code=400, detail="Only md/txt files support text preview")

    return {
        "name": file_path.name,
        "type": file_path.suffix.lower().lstrip("."),
        "content": file_path.read_text(encoding="utf-8"),
    }


@app.post("/learn")
def learn_api(request: TextRequest):
    from backend.ai_core import learning_workflow

    return learning_workflow(request.text)


@app.post("/rebuild-index")
def rebuild_index_api():
    from backend.rag_store import rebuild_rag_index

    rebuild_rag_index()
    return {"message": "RAG 索引已重建"}


@app.get("/debug-index-sources")
def debug_index_sources_api():
    from backend.rag_store import list_index_sources

    sources = list_index_sources()
    return {
        "count": len(sources),
        "sources": sources,
    }


@app.post("/debug-rag")
def debug_rag_api(request: TextRequest):
    from backend.rag_store import search_relevant_chunks

    chunks = search_relevant_chunks(request.text, top_k=5)

    return {
        "question": request.text,
        "count": len(chunks),
        "chunks": chunks,
    }


# --- Session history APIs ---


@app.get("/sessions")
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


@app.get("/sessions/{session_id}/messages")
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
