import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
SUPPORTED_MODELS = {DEFAULT_MODEL}
API_KEY_ENV_NAMES = ("MY_MIMO_API_KEY", "MIMO_API_KEY", "OPENAI_API_KEY")

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


def read_api_key() -> tuple[str | None, str | None]:
    for name in API_KEY_ENV_NAMES:
        value = os.getenv(name)
        if value:
            return value, name
    return None, None


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
