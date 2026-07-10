import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = DEEPSEEK_BASE_URL
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / DEFAULT_EMBEDDING_MODEL
QUERY_REWRITE_MODES = {"off", "conditional", "always"}
CORS_DEFAULT_ORIGINS = (
    "http://127.0.0.1:5500",
    "http://localhost:5500",
)
TOOL_SECRET_PLACEHOLDERS = {
    "replace-with-a-long-random-requester-secret",
    "replace-with-a-different-long-random-approver-secret",
}
DASHSCOPE_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
MODEL_PROVIDERS = {
    DEFAULT_MODEL: {
        "kind": "chat",
        "base_url": DEEPSEEK_BASE_URL,
        "api_key_env_names": ("DEEPSEEK_API_KEY",),
    },
    "deepseek-v4-flash": {
        "kind": "chat",
        "base_url": DEEPSEEK_BASE_URL,
        "api_key_env_names": ("DEEPSEEK_API_KEY",),
    },
    "qwen3.7-max": {
        "kind": "chat",
        "base_url": DASHSCOPE_OPENAI_BASE_URL,
        "api_key_env_names": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY", "QWEN_API_KEY"),
    },
    "wanx2.1-t2i-plus": {
        "kind": "image",
        "base_url": DASHSCOPE_API_BASE_URL,
        "api_key_env_names": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY"),
    },
}
SUPPORTED_MODELS = set(MODEL_PROVIDERS)
API_KEY_ENV_NAMES = MODEL_PROVIDERS[DEFAULT_MODEL]["api_key_env_names"]

load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    docs_path: Path
    rag_index_dir: Path
    model: str
    base_url: str
    api_key: str | None
    api_key_source: str | None
    embedding_model: str
    embedding_model_local_only: bool
    enable_rag_auto_build: bool
    enable_rag_warmup: bool
    rag_warmup_load_index: bool
    enable_reranker: bool
    reranker_model: str
    reranker_top_n: int
    query_rewrite_mode: str
    cors_allowed_origins: tuple[str, ...]
    enable_insecure_dev_tool_keys: bool
    tool_approval_key: str | None
    tool_approver_key: str | None
    max_upload_size_bytes: int
    max_upload_total_bytes: int
    upload_max_concurrency: int
    upload_rate_limit: int
    upload_rate_window_seconds: int
    image_proxy_max_response_bytes: int
    image_proxy_max_concurrency: int
    image_proxy_rate_limit: int
    image_proxy_rate_window_seconds: int
    image_proxy_timeout_seconds: int
    max_pdf_pages: int
    pdf_validation_timeout_seconds: int
    pdf_validation_max_memory_bytes: int
    enable_ocr: bool
    ocr_engine: str
    ocr_min_text_chars: int
    ocr_render_dpi: int
    ocr_max_pages: int

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def normalize_model(model: str | None) -> str:
    if model in SUPPORTED_MODELS:
        return model
    return DEFAULT_MODEL


def get_model_provider(model: str | None) -> dict:
    return MODEL_PROVIDERS[normalize_model(model)]


def is_image_model(model: str | None) -> bool:
    return get_model_provider(model)["kind"] == "image"


def read_api_key(env_names: tuple[str, ...] = API_KEY_ENV_NAMES) -> tuple[str | None, str | None]:
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value, name
    return None, None


def get_model_api_settings(model: str | None) -> tuple[str, str | None, str | None]:
    selected_model = normalize_model(model)
    provider = get_model_provider(selected_model)
    api_key, api_key_source = read_api_key(provider["api_key_env_names"])
    base_url = provider["base_url"]
    if selected_model.startswith("deepseek-"):
        base_url = os.getenv("DEEPSEEK_BASE_URL", base_url)
    return base_url, api_key, api_key_source


def read_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def read_positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def read_query_rewrite_mode() -> str:
    mode = (os.getenv("QUERY_REWRITE_MODE", "off").strip().lower() or "off")
    return mode if mode in QUERY_REWRITE_MODES else "off"


def read_cors_allowed_origins() -> tuple[str, ...]:
    raw_value = os.getenv("CORS_ALLOWED_ORIGINS")
    if raw_value is None:
        return CORS_DEFAULT_ORIGINS

    origins = tuple(
        origin.strip().rstrip("/")
        for origin in raw_value.split(",")
        if origin.strip()
    )
    if not origins:
        return CORS_DEFAULT_ORIGINS
    if "*" in origins:
        raise ValueError("CORS_ALLOWED_ORIGINS must not contain '*'")
    return origins


def read_tool_secret(name: str, *, allow_insecure_dev_key: bool | None = None) -> str | None:
    value = (os.getenv(name) or "").strip()
    if allow_insecure_dev_key is None:
        allow_insecure_dev_key = read_bool_env("ENABLE_INSECURE_DEV_TOOL_KEYS", False)

    minimum_length = 8 if allow_insecure_dev_key else 32
    if not value or len(value) < minimum_length or value.lower() in TOOL_SECRET_PLACEHOLDERS:
        return None
    return value


def get_embedding_model_settings() -> tuple[str, bool]:
    configured_model = os.getenv("EMBEDDING_MODEL_PATH") or os.getenv("EMBEDDING_MODEL")
    if configured_model:
        return configured_model, read_bool_env("EMBEDDING_MODEL_LOCAL_ONLY", True)

    if DEFAULT_EMBEDDING_MODEL_PATH.exists():
        return str(DEFAULT_EMBEDDING_MODEL_PATH), True

    return DEFAULT_EMBEDDING_MODEL, read_bool_env("EMBEDDING_MODEL_LOCAL_ONLY", True)


def get_config() -> AppConfig:
    api_key, api_key_source = read_api_key()
    embedding_model, embedding_model_local_only = get_embedding_model_settings()
    enable_insecure_dev_tool_keys = read_bool_env(
        "ENABLE_INSECURE_DEV_TOOL_KEYS",
        False,
    )
    return AppConfig(
        project_root=PROJECT_ROOT,
        docs_path=PROJECT_ROOT / "docs",
        rag_index_dir=PROJECT_ROOT / "rag_index",
        model=normalize_model(os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)),
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        api_key_source=api_key_source,
        embedding_model=embedding_model,
        embedding_model_local_only=embedding_model_local_only,
        enable_rag_auto_build=read_bool_env("ENABLE_RAG_AUTO_BUILD", True),
        enable_rag_warmup=read_bool_env("ENABLE_RAG_WARMUP", False),
        rag_warmup_load_index=read_bool_env("RAG_WARMUP_LOAD_INDEX", True),
        enable_reranker=read_bool_env("ENABLE_RERANKER", False),
        reranker_model=os.getenv("RERANKER_MODEL", "").strip(),
        reranker_top_n=read_positive_int_env("RERANKER_TOP_N", 20),
        query_rewrite_mode=read_query_rewrite_mode(),
        cors_allowed_origins=read_cors_allowed_origins(),
        enable_insecure_dev_tool_keys=enable_insecure_dev_tool_keys,
        tool_approval_key=read_tool_secret(
            "TOOL_APPROVAL_KEY",
            allow_insecure_dev_key=enable_insecure_dev_tool_keys,
        ),
        tool_approver_key=read_tool_secret(
            "TOOL_APPROVER_KEY",
            allow_insecure_dev_key=enable_insecure_dev_tool_keys,
        ),
        max_upload_size_bytes=read_positive_int_env(
            "MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024
        ),
        max_upload_total_bytes=read_positive_int_env(
            "MAX_UPLOAD_TOTAL_BYTES", 100 * 1024 * 1024
        ),
        upload_max_concurrency=read_positive_int_env("UPLOAD_MAX_CONCURRENCY", 2),
        upload_rate_limit=read_positive_int_env("UPLOAD_RATE_LIMIT", 10),
        upload_rate_window_seconds=read_positive_int_env(
            "UPLOAD_RATE_WINDOW_SECONDS", 60
        ),
        image_proxy_max_response_bytes=read_positive_int_env(
            "IMAGE_PROXY_MAX_RESPONSE_BYTES", 20 * 1024 * 1024
        ),
        image_proxy_max_concurrency=read_positive_int_env(
            "IMAGE_PROXY_MAX_CONCURRENCY", 4
        ),
        image_proxy_rate_limit=read_positive_int_env("IMAGE_PROXY_RATE_LIMIT", 30),
        image_proxy_rate_window_seconds=read_positive_int_env(
            "IMAGE_PROXY_RATE_WINDOW_SECONDS", 60
        ),
        image_proxy_timeout_seconds=read_positive_int_env(
            "IMAGE_PROXY_TIMEOUT_SECONDS", 20
        ),
        max_pdf_pages=read_positive_int_env("MAX_PDF_PAGES", 500),
        pdf_validation_timeout_seconds=read_positive_int_env(
            "PDF_VALIDATION_TIMEOUT_SECONDS", 5
        ),
        pdf_validation_max_memory_bytes=read_positive_int_env(
            "PDF_VALIDATION_MAX_MEMORY_BYTES", 256 * 1024 * 1024
        ),
        enable_ocr=read_bool_env("ENABLE_OCR", False),
        ocr_engine=(os.getenv("OCR_ENGINE", "none").strip().lower() or "none"),
        ocr_min_text_chars=read_positive_int_env("OCR_MIN_TEXT_CHARS", 80),
        ocr_render_dpi=read_positive_int_env("OCR_RENDER_DPI", 200),
        ocr_max_pages=read_positive_int_env("OCR_MAX_PAGES", 20),
    )
