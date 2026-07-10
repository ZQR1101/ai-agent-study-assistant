from typing import Literal

from pydantic import BaseModel, Field


ChatMode = Literal["chat", "rag", "explain", "summarize", "quiz", "learn", "auto"]
AgentToolName = Literal[
    "chat",
    "rag_search",
    "study",
    "save_note",
    "save_flashcards",
    "save_quiz",
    # Accepted while stored plans migrate to the merged v2 names.
    "rag",
    "explain",
    "summarize",
    "quiz",
    "flashcard",
    "delete_saved_item",
    "delete_run",
    "delete_knowledge_file",
    "reset_saved_items",
    "reset_rag_index",
    "rebuild_rag_index",
]
PlannerMode = Literal["rule", "llm"]
RetrievalMode = Literal["vector", "bm25", "hybrid"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: ChatMode = "auto"
    model: str = "deepseek-v4-pro"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    use_agent: bool = False
    use_rag: bool = False
    use_langgraph: bool = False
    planner_mode: PlannerMode = "rule"
    top_k: int = Field(3, ge=1, le=10)
    retrieval_mode: RetrievalMode = "vector"
    reranker_enabled: bool = False
    session_id: str | None = None
    run_id: str | None = None
    history: list[dict] = Field(default_factory=list)


class SourceChunk(BaseModel):
    source: str
    score: float | None = None
    snippet: str | None = None
    text: str | None = None
    chunk_id: str | None = None
    retrieval: str | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    reranker_used: bool | None = None


class AgentPlanStep(BaseModel):
    tool: AgentToolName
    input: str = Field(..., min_length=1)
    reason: str | None = None
    arguments: dict = Field(default_factory=dict)


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


class JudgeDeduction(BaseModel):
    metric: str
    points: float | None = Field(default=None, ge=0.0)
    reason: str


class JudgeEvaluationResult(BaseModel):
    id: int | None = None
    session_id: str | None = None
    run_id: str | None = None
    question: str | None = None
    answer: str | None = None
    judge_model: str | None = None
    accuracy: float = Field(..., ge=0.0, le=10.0)
    completeness: float = Field(..., ge=0.0, le=10.0)
    citation_quality: float | None = Field(default=None, ge=0.0, le=10.0)
    overall_score: float = Field(..., ge=0.0, le=10.0)
    verdict: Literal["PASS", "WEAK_PASS", "FAIL"] | None = None
    deductions: list[JudgeDeduction] = Field(default_factory=list)
    feedback: str | None = None
    raw_output: str | None = None
    judge_feedback: Literal["good", "bad"] | None = None
    judge_feedback_reason: str | None = None
    created_at: str | None = None


class JudgeFeedbackRequest(BaseModel):
    judge_feedback: Literal["good", "bad"]
    reason: str | None = None


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str
    session_id: str | None = None
    run_id: str | None = None
    sources: list[SourceChunk] = Field(default_factory=list)
    trace: list[TraceBlock] = Field(default_factory=list)
    plan: list[AgentPlanStep] = Field(default_factory=list)
    flashcards: list[FlashcardItem] = Field(default_factory=list)
    runtime_info: dict = Field(default_factory=dict)
    judge_evaluation: JudgeEvaluationResult | None = None
    run_summary: dict = Field(default_factory=dict)
    run_details: dict = Field(default_factory=dict)
    pending_actions: list[dict] = Field(default_factory=list)


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
    response: dict | None = None
    created_at: str | None = None
