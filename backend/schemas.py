from typing import Literal

from pydantic import BaseModel, Field


ChatMode = Literal["chat", "rag", "explain", "summarize", "quiz", "learn", "auto"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: ChatMode = "auto"
    model: str = "mimo-v2.5"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    use_agent: bool = False
    use_rag: bool = False
    top_k: int = Field(3, ge=1, le=10)


class SourceChunk(BaseModel):
    source: str
    score: float | None = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str
    sources: list[SourceChunk] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
