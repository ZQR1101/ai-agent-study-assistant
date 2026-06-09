import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
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


def get_config() -> AppConfig:
    api_key, api_key_source = read_api_key()
    return AppConfig(
        project_root=PROJECT_ROOT,
        docs_path=PROJECT_ROOT / "docs",
        rag_index_dir=PROJECT_ROOT / "rag_index",
        model=normalize_model(os.getenv("MIMO_MODEL", DEFAULT_MODEL)),
        base_url=os.getenv("MIMO_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        api_key_source=api_key_source,
    )
