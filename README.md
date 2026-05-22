# AI Study Assistant

![Cover](images/cover.png)

This project is still under active development.

一个基于 FastAPI + RAG + Agent 的 AI 学习助手。

支持：

- RAG 知识库问答
- Agent 自动路由
- PDF 上传
- ChatPDF
- Web 聊天界面

---

# Tech Stack

## Backend

- FastAPI
- LangChain
- SentenceTransformers

## Frontend

- HTML
- JavaScript

## AI

- RAG
- Embedding
- Agent Routing

---

# Features

## RAG Question Answering

基于本地知识库进行检索增强问答。

## Agent Routing

自动判断用户请求并选择工具。

## ChatPDF

支持上传 PDF 后进行问答。

## Web Chat UI

支持网页聊天交互。

---

# Project Structure

```text
backend/
    server.py
    ai_core.py

frontend/
    index.html

docs/
    knowledge base files

images/
    README images
```

---

# Run Backend

```bash
uvicorn backend.server:app --reload
```

---

# Open Frontend

打开：

```text
frontend/index.html
```

---

# Future Plans

- Vector Database
- Multi-Agent
- React Frontend
- Long-term Memory
- Streaming Response

————————————————

# Environment Variables

Create a `.env` file based on `.env.example`.

```env
MY_MIMO_API_KEY=your_api_key_here
```
