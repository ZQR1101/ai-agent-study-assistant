import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / DEFAULT_EMBEDDING_MODEL
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DASHSCOPE_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
MODEL_PROVIDERS = {
    DEFAULT_MODEL: {
        "kind": "chat",
        "base_url": DEFAULT_BASE_URL,
        "api_key_env_names": ("MY_MIMO_API_KEY", "MIMO_API_KEY", "OPENAI_API_KEY"),
    },
    "deepseek-v4-pro": {
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
    enable_rag_warmup: bool
    rag_warmup_load_index: bool
    enable_reranker: bool
    reranker_model: str
    reranker_top_n: int

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
    return provider["base_url"], api_key, api_key_source


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
    return AppConfig(
        project_root=PROJECT_ROOT,
        docs_path=PROJECT_ROOT / "docs",
        rag_index_dir=PROJECT_ROOT / "rag_index",
        model=normalize_model(os.getenv("MIMO_MODEL", DEFAULT_MODEL)),
        base_url=os.getenv("MIMO_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        api_key_source=api_key_source,
        embedding_model=embedding_model,
        embedding_model_local_only=embedding_model_local_only,
        enable_rag_warmup=read_bool_env("ENABLE_RAG_WARMUP", False),
        rag_warmup_load_index=read_bool_env("RAG_WARMUP_LOAD_INDEX", True),
        enable_reranker=read_bool_env("ENABLE_RERANKER", False),
        reranker_model=os.getenv("RERANKER_MODEL", "").strip(),
        reranker_top_n=read_positive_int_env("RERANKER_TOP_N", 20),
    )
