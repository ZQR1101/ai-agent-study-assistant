from typing import Literal

from pydantic import BaseModel, Field


ChatMode = Literal["chat", "rag", "explain", "summarize", "quiz", "learn", "auto"]
AgentToolName = Literal["chat", "rag", "explain", "summarize", "quiz", "flashcard"]
PlannerMode = Literal["rule", "llm"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: ChatMode = "auto"
    model: str = "mimo-v2.5"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    use_agent: bool = False
    use_rag: bool = False
    use_langgraph: bool = False
    planner_mode: PlannerMode = "rule"
    top_k: int = Field(3, ge=1, le=10)
    session_id: str | None = None
    history: list[dict] = Field(default_factory=list)


class SourceChunk(BaseModel):
    source: str
    score: float | None = None
    snippet: str | None = None
    text: str | None = None


class AgentPlanStep(BaseModel):
    tool: AgentToolName
    input: str = Field(..., min_length=1)
    reason: str | None = None


class AgentPlan(BaseModel):
    goal: str = Field(..., min_length=1)
    steps: list[AgentPlanStep] = Field(..., min_length=1)
    fallback: bool = False


class TraceBlock(BaseModel):
    title: str
    items: list[str] = Field(default_factory=list)


class FlashcardItem(BaseModel):
    front: str = Field(..., min_length=1)
    back: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    card_type: Literal["text", "image"] = "text"
    image_url: str | None = None
    image_alt: str | None = None


class FlashcardPayload(BaseModel):
    cards: list[FlashcardItem] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str
    session_id: str | None = None
    sources: list[SourceChunk] = Field(default_factory=list)
    trace: list[TraceBlock] = Field(default_factory=list)
    plan: list[AgentPlanStep] = Field(default_factory=list)
    flashcards: list[FlashcardItem] = Field(default_factory=list)
    runtime_info: dict = Field(default_factory=dict)


class SessionSummary(BaseModel):
    id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    message_count: int = 0


class MessageSummary(BaseModel):
    id: int | None = None
    session_id: str
    role: str
    content: str
    created_at: str | None = None
