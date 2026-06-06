from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.ai_core import (
    agent_router,
    explain,
    generate_questions,
    learning_workflow,
    rag_answer_with_sources,
    run_chat_request,
    summarize,
)
from backend.rag_store import list_index_sources, rebuild_rag_index, search_relevant_chunks
from backend.schemas import ChatRequest, ChatResponse


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


class TextRequest(BaseModel):
    text: str


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
    return {"status": "ok"}


@app.post("/echo")
def echo_api(request: TextRequest):
    return {"echo": request.text}


@app.post("/chat", response_model=ChatResponse)
def chat_api(request: ChatRequest):
    return run_chat_request(request)


@app.post("/explain")
def explain_api(request: TextRequest):
    result = explain(request.text)
    return {"result": result}


@app.post("/summarize")
def summarize_api(request: TextRequest):
    result = summarize(request.text)
    return {"result": result}


@app.post("/quiz")
def quiz_api(request: TextRequest):
    result = generate_questions(request.text)
    return {"result": result}


@app.post("/rag")
def rag_api(request: TextRequest):
    return rag_answer_with_sources(request.text)


@app.post("/agent")
def agent_api(request: TextRequest):
    return {"result": agent_router(request.text)}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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
    return learning_workflow(request.text)


@app.post("/rebuild-index")
def rebuild_index_api():
    rebuild_rag_index()
    return {"message": "RAG 索引已重建"}


@app.get("/debug-index-sources")
def debug_index_sources_api():
    sources = list_index_sources()
    return {
        "count": len(sources),
        "sources": sources,
    }


@app.post("/debug-rag")
def debug_rag_api(request: TextRequest):
    chunks = search_relevant_chunks(request.text, top_k=5)

    return {
        "question": request.text,
        "count": len(chunks),
        "chunks": chunks,
    }
