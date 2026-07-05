import importlib.util
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
CORE_DIRS = ("backend", "frontend", "docs", "scripts")
DOCS_DIR = PROJECT_ROOT / "docs"
RAG_INDEX_FILE = PROJECT_ROOT / "rag_index" / "index.faiss"
RAG_CHUNKS_FILE = PROJECT_ROOT / "rag_index" / "chunks.json"
API_KEY_ENV_NAMES = ("MY_MIMO_API_KEY", "MIMO_API_KEY", "OPENAI_API_KEY")

REQUIREMENT_IMPORTS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python-dotenv": "dotenv",
    "langchain": "langchain",
    "langchain-openai": "langchain_openai",
    "sentence-transformers": "sentence_transformers",
    "scikit-learn": "sklearn",
    "pypdf": "pypdf",
    "python-multipart": "multipart",
    "faiss-cpu": "faiss",
    "langgraph": "langgraph",
    "SQLAlchemy": "sqlalchemy",
    "psycopg[binary]": "psycopg",
}
OPTIONAL_DEPENDENCIES = {
    "rapidocr-onnxruntime": "rapidocr_onnxruntime",
    "PyMuPDF": "fitz",
}

CORE_MODULES = (
    "backend.config",
    "backend.history_utils",
    "backend.schemas",
)


def print_result(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def read_dotenv(path: Path) -> dict[str, str]:
    values = {}

    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 10)
    status = "OK" if ok else "ERROR"
    print_result(status, f"Python {version.major}.{version.minor}.{version.micro}")
    return ok


def check_project_structure() -> bool:
    ok = True

    for directory in CORE_DIRS:
        path = PROJECT_ROOT / directory
        if path.is_dir():
            print_result("OK", f"{directory}/ directory exists")
        else:
            print_result("ERROR", f"{directory}/ directory not found")
            ok = False

    if ENV_EXAMPLE_FILE.exists():
        print_result("OK", ".env.example exists")
    else:
        print_result("ERROR", ".env.example not found")
        ok = False

    return ok


def check_dependencies() -> bool:
    if not REQUIREMENTS_FILE.exists():
        print_result("ERROR", "requirements.txt not found")
        return False

    missing = []
    for raw_line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        requirement = raw_line.strip()
        if not requirement or requirement.startswith("#"):
            continue

        package_name = requirement.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].strip()
        import_name = REQUIREMENT_IMPORTS.get(package_name, package_name.replace("-", "_"))
        if importlib.util.find_spec(import_name) is None:
            missing.append(requirement)

    if missing:
        print_result("ERROR", "Missing dependencies: " + ", ".join(missing))
        print_result("HINT", "Run: pip install -r requirements.txt")
        return False

    print_result("OK", "All requirements imports are available")
    return True


def check_core_modules() -> bool:
    ok = True

    for module_name in CORE_MODULES:
        try:
            __import__(module_name)
            print_result("OK", f"Imported {module_name}")
        except Exception as error:
            print_result("ERROR", f"Cannot import {module_name}: {error}")
            ok = False

    try:
        from backend.config import get_config

        config = get_config()
        print_result("OK", f"Config loaded; model={config.model}")
    except Exception as error:
        print_result("ERROR", f"backend.config cannot read config: {error}")
        ok = False

    return ok


def check_optional_dependencies() -> bool:
    available = []
    missing = []
    for package_name, import_name in OPTIONAL_DEPENDENCIES.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
        else:
            available.append(package_name)

    if available:
        print_result("OK", "Optional OCR dependencies available: " + ", ".join(available))
    if missing:
        print_result(
            "INFO",
            "Optional OCR dependencies not installed: "
            + ", ".join(missing)
            + "; ordinary PDF/TXT/MD support is unaffected",
        )
    return True


def check_env() -> bool:
    env_values = read_dotenv(ENV_FILE)

    if ENV_FILE.exists():
        print_result("OK", ".env exists")
    else:
        print_result("WARN", ".env not found; copy .env.example to .env for model calls")

    key_source = None
    for name in API_KEY_ENV_NAMES:
        if os.getenv(name) or env_values.get(name):
            key_source = name
            break

    if key_source:
        print_result("OK", f"API key configured via {key_source}")
    else:
        print_result("WARN", "No model API key found; offline tests work, LLM calls will fail")

    base_url = os.getenv("MIMO_BASE_URL") or env_values.get("MIMO_BASE_URL")
    if base_url:
        print_result("OK", f"MIMO_BASE_URL={base_url}")
    else:
        print_result("OK", "MIMO_BASE_URL not set; default will be used")

    return True


def check_reranker_config() -> bool:
    from backend.config import get_config

    config = get_config()
    if not config.enable_reranker:
        print_result("OK", "Reranker disabled (ENABLE_RERANKER=false)")
        return True
    if not config.reranker_model:
        print_result("WARN", "Reranker enabled but RERANKER_MODEL is empty; retrieval will fall back")
        return True

    model_name = config.reranker_model
    model_path = Path(model_name)
    is_local_path = model_path.is_absolute() or model_name.startswith((".", "/", "\\"))
    if is_local_path and not model_path.exists():
        print_result("WARN", f"Reranker model path not found: {model_name}; retrieval will fall back")
        return True

    print_result(
        "OK",
        f"Reranker configured; model={model_name}, top_n={config.reranker_top_n}",
    )
    return True


def check_docs_dir() -> bool:
    if not DOCS_DIR.exists():
        print_result("ERROR", "docs directory not found")
        return False

    if not DOCS_DIR.is_dir():
        print_result("ERROR", "docs exists but is not a directory")
        return False

    supported_files = [
        path
        for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf"}
    ]
    print_result("OK", f"docs directory exists; supported files={len(supported_files)}")
    return True


def check_rag_index() -> bool:
    index_exists = RAG_INDEX_FILE.exists()
    chunks_exists = RAG_CHUNKS_FILE.exists()

    if not index_exists or not chunks_exists:
        print_result("WARN", "RAG index is missing or incomplete")
        print_result("HINT", "Start the backend and run: POST /rebuild-index")
        return True

    try:
        chunks = json.loads(RAG_CHUNKS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print_result("WARN", f"RAG chunks file exists but cannot be read: {error}")
        return True

    print_result("OK", f"RAG index files exist; chunks={len(chunks)}")
    return True


def main() -> int:
    checks = [
        check_python(),
        check_project_structure(),
        check_dependencies(),
        check_optional_dependencies(),
        check_core_modules(),
        check_env(),
        check_reranker_config(),
        check_docs_dir(),
        check_rag_index(),
    ]

    if all(checks):
        print_result("OK", "Setup check completed")
        return 0

    print_result("ERROR", "Setup check found blocking issues")
    return 1


if __name__ == "__main__":
    sys.exit(main())
