from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi import UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from backend.ai_core import (
    explain,
    summarize,
    generate_questions,
    rag_answer,
    agent_router,
    learning_workflow,
    rag_answer_with_sources,
)
from backend.rag_store import rebuild_rag_index

app = FastAPI(title="AI学习助手 API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "AI学习助手后端启动成功"}


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
    result = rag_answer_with_sources(request.text)
    return result


@app.post("/agent")
def agent_api(request: TextRequest):

    result = agent_router(request.text)

    return {
        "result": result
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    docs_path = Path(__file__).parent.parent / "docs"
    docs_path.mkdir(exist_ok=True)

    save_path = docs_path / file.filename

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    rebuild_rag_index()

    return {
        "message": f"{file.filename} 上传成功，知识库索引已更新"
    }

    return {
        "message": f"{file.filename} 上传成功"
    }


@app.post("/learn")
def learn_api(request: TextRequest):
    result = learning_workflow(request.text)
    return result


@app.post("/rebuild-index")
def rebuild_index_api():
    rebuild_rag_index()
    return {
        "message": "RAG 索引已重建"
    }