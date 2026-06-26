from pathlib import Path
import json
import os
import re

from backend.config import get_embedding_model_settings


embedding_model = None
rag_index_error = None


def _get_pdf_reader():
    from pypdf import PdfReader

    return PdfReader


def _get_faiss():
    import faiss

    return faiss


def _get_numpy():
    import numpy as np

    return np


def _get_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        model_name_or_path, local_only = get_embedding_model_settings()
        model_path = Path(model_name_or_path)
        is_path_like = (
            model_path.is_absolute()
            or model_name_or_path.startswith((".", "/", "\\"))
            or "/" in model_name_or_path
            or "\\" in model_name_or_path
        )
        if local_only and is_path_like and not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model path not found: {model_name_or_path}. "
                "Set EMBEDDING_MODEL_PATH to an existing local model directory, "
                "or set EMBEDDING_MODEL_LOCAL_ONLY=false to allow model download."
            )
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        SentenceTransformer = _get_sentence_transformer()
        embedding_model = SentenceTransformer(
            model_name_or_path,
            local_files_only=local_only,
        )

    return embedding_model

chunks = []
index = None


PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs"
INDEX_DIR = PROJECT_ROOT / "rag_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

SIMILARITY_THRESHOLD = 0.55
MIN_CHUNK_LENGTH = 30


def is_valid_chunk(text: str, min_length: int = MIN_CHUNK_LENGTH) -> bool:
    clean_text = " ".join(str(text or "").split())

    if len(clean_text) < min_length:
        return False

    if re.fullmatch(r"[\s\|\-\:\#\*_`~\.=]+", clean_text):
        return False

    table_separator = re.fullmatch(r"[\|\s\-\:]+", clean_text)
    if table_separator:
        return False

    meaningful_chars = re.findall(r"[0-9A-Za-z\u4e00-\u9fff]", clean_text)
    if len(meaningful_chars) < 20:
        return False

    meaningful_ratio = len(meaningful_chars) / max(len(clean_text), 1)
    if meaningful_ratio < 0.25:
        return False

    return True


def expand_query(query: str) -> str:
    lowered = query.lower()
    expansions = []

    if "agentic rag" in lowered or "agent rag" in lowered:
        expansions.extend([
            "agentic rag",
            "代理式RAG",
            "代理式检索增强生成",
            "智能体RAG",
            "agent rag",
        ])

    if "langgraph" in lowered:
        expansions.extend(["LangGraph", "图工作流", "状态图", "agent workflow"])

    if "prompt engineering" in lowered or "提示工程" in lowered:
        expansions.extend(["prompt engineering", "提示工程", "提示词工程", "prompting best practices"])

    if re.search(r"\brag\b", lowered) or "检索增强生成" in query:
        expansions.extend(["RAG", "检索增强生成", "知识库问答", "retrieval augmented generation"])

    unique_expansions = []
    for item in expansions:
        if item not in query and item not in unique_expansions:
            unique_expansions.append(item)

    if not unique_expansions:
        return query

    return f"{query} {' '.join(unique_expansions)}"


def load_documents():
    documents = []

    if not DOCS_PATH.exists():
        DOCS_PATH.mkdir(exist_ok=True)
        return documents

    for file_path in DOCS_PATH.iterdir():
        suffix = file_path.suffix.lower()

        if suffix in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({
                    "source": file_path.name,
                    "text": f.read()
                })

        elif suffix == ".pdf":
            PdfReader = _get_pdf_reader()
            reader = PdfReader(file_path)
            pdf_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"

            documents.append({
                "source": file_path.name,
                "text": pdf_text
            })

    return documents


def build_chunks():
    documents = load_documents()

    chunk_size = 500
    overlap = 100

    new_chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]

            if is_valid_chunk(chunk):
                new_chunks.append({
                    "source": source,
                    "text": chunk.strip()
                })

    return new_chunks


def save_rag_index():
    INDEX_DIR.mkdir(exist_ok=True)

    if index is not None:
        faiss = _get_faiss()
        faiss.write_index(index, str(INDEX_FILE))

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_rag_index():
    global index
    global chunks
    global rag_index_error

    rag_index_error = None

    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        return False

    faiss = _get_faiss()
    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"已加载 FAISS RAG 索引，共 {len(chunks)} 个 chunks。")
    return True


def rebuild_rag_index():
    global chunks
    global index
    global rag_index_error

    rag_index_error = None

    print("正在构建 FAISS RAG 索引...")

    chunks = build_chunks()

    if not chunks:
        index = None
        print("知识库为空，未构建索引。")
        return

    chunk_texts = [chunk["text"] for chunk in chunks]

    faiss = _get_faiss()
    np = _get_numpy()
    model = get_embedding_model()
    embeddings = model.encode(chunk_texts)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    save_rag_index()

    print(f"FAISS RAG 索引构建完成，共 {len(chunks)} 个 chunks，并已保存到本地。")


def ensure_rag_index():
    global index
    global chunks
    global rag_index_error

    if index is not None:
        return

    loaded = load_rag_index()

    if not loaded:
        try:
            rebuild_rag_index()
        except Exception as exc:
            index = None
            chunks = []
            rag_index_error = str(exc)
            print(f"RAG index unavailable: {rag_index_error}")


def get_rag_index_status() -> dict:
    chunks_count = None
    chunks_error = None
    model_name_or_path, local_only = get_embedding_model_settings()
    model_path = Path(model_name_or_path)
    is_path_like = (
        model_path.is_absolute()
        or model_name_or_path.startswith((".", "/", "\\"))
        or "/" in model_name_or_path
        or "\\" in model_name_or_path
    )
    model_path_missing = bool(local_only and is_path_like and not model_path.exists())

    if CHUNKS_FILE.exists():
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks_count = len(json.load(f))
        except (OSError, json.JSONDecodeError) as error:
            chunks_error = str(error)

    ready = (
        INDEX_FILE.exists()
        and CHUNKS_FILE.exists()
        and chunks_error is None
        and not model_path_missing
        and rag_index_error is None
    )
    message = (
        "RAG index is ready"
        if ready
        else (
            f"Embedding model path is missing: {model_name_or_path}"
            if model_path_missing
            else "RAG index is missing or incomplete; run python scripts/check_setup.py or POST /rebuild-index"
        )
    )

    return {
        "ready": ready,
        "index_file": str(INDEX_FILE),
        "chunks_file": str(CHUNKS_FILE),
        "index_exists": INDEX_FILE.exists(),
        "chunks_exists": CHUNKS_FILE.exists(),
        "chunks_count": chunks_count,
        "chunks_error": chunks_error,
        "embedding_model": model_name_or_path,
        "embedding_model_local_only": local_only,
        "embedding_model_path_missing": model_path_missing,
        "error": rag_index_error,
        "message": message,
    }


def list_index_sources() -> list[str]:
    ensure_rag_index()
    return sorted(set(chunk["source"] for chunk in chunks))


def _extract_query_terms(question: str) -> list[str]:
    raw_terms = re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", question.lower())
    stop_words = {"什么是", "什么", "怎么", "如何", "the", "and", "for", "with"}
    return [
        term
        for term in raw_terms
        if len(term) >= 2 and term not in stop_words
    ]


def _keyword_relevant_chunks(
    question: str,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
):
    terms = _extract_query_terms(question)

    if not terms:
        return []

    min_hits = 2 if len(terms) >= 2 else 1
    results = []

    for chunk in chunks:
        if not is_valid_chunk(chunk["text"]):
            continue

        source_text = chunk["source"].lower()
        content_text = chunk["text"].lower()
        source_hits = sum(1 for term in terms if term in source_text)
        content_hits = sum(1 for term in terms if term in content_text)
        total_hits = source_hits + content_hits

        if total_hits < min_hits:
            continue

        score = similarity_threshold + min(0.15, total_hits * 0.03 + source_hits * 0.02)
        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(score),
            "_keyword_hits": total_hits,
        })

    results.sort(
        key=lambda item: (
            item["_keyword_hits"],
            item["score"],
            len(item["text"]),
        ),
        reverse=True,
    )

    return [
        {
            "source": item["source"],
            "text": item["text"],
            "score": item["score"],
        }
        for item in results[:top_k]
    ]


def search_relevant_chunks(
    question: str,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    include_metadata: bool = False,
):
    global index
    global chunks
    global rag_index_error

    ensure_rag_index()
    expanded_question = expand_query(question)

    if rag_index_error or index is None or not chunks:
        if include_metadata:
            return {
                "chunks": [],
                "highest_score": None,
                "threshold": similarity_threshold,
                "passed_threshold": False,
                "expanded_query": expanded_question,
                "raw_count": 0,
                "valid_count": 0,
                "discarded_invalid_count": 0,
                "error": rag_index_error,
            }
        return []

    try:
        model = get_embedding_model()
        faiss = _get_faiss()
        np = _get_numpy()
        question_embedding = model.encode([expanded_question])
        question_embedding = np.array(question_embedding).astype("float32")
    except Exception as exc:
        rag_index_error = str(exc)
        if include_metadata:
            return {
                "chunks": [],
                "highest_score": None,
                "threshold": similarity_threshold,
                "passed_threshold": False,
                "expanded_query": expanded_question,
                "raw_count": 0,
                "valid_count": 0,
                "discarded_invalid_count": 0,
                "error": rag_index_error,
            }
        return []

    faiss.normalize_L2(question_embedding)

    search_k = min(len(chunks), max(top_k * 5, top_k))
    scores, indices = index.search(question_embedding, search_k)

    highest_score = None
    raw_results = []
    results = []

    if len(indices[0]) > 0 and indices[0][0] != -1:
        highest_score = float(scores[0][0])

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        if score < similarity_threshold:
            continue

        chunk = chunks[idx]
        raw_results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(score)
        })

    raw_count = len(raw_results)
    valid_vector_results = [
        item
        for item in raw_results
        if is_valid_chunk(item["text"])
    ]

    keyword_results = _keyword_relevant_chunks(
        expanded_question,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    combined_results = valid_vector_results + [
        item for item in keyword_results if is_valid_chunk(item["text"])
    ]

    seen = set()
    deduped_results = []
    for item in combined_results:
        key = (item["source"], item["text"])
        if key in seen:
            continue

        seen.add(key)
        deduped_results.append(item)

    deduped_results.sort(key=lambda item: item["score"], reverse=True)
    results = deduped_results[:top_k]
    valid_count = len(results)
    discarded_invalid_count = raw_count - len(valid_vector_results)

    if results:
        highest_score = max(float(item["score"]) for item in results)
    elif include_metadata:
        return {
            "chunks": [],
            "highest_score": highest_score,
            "threshold": similarity_threshold,
            "passed_threshold": False,
            "expanded_query": expanded_question,
            "raw_count": raw_count,
            "valid_count": 0,
            "discarded_invalid_count": discarded_invalid_count,
        }

    if include_metadata:
        return {
            "chunks": results,
            "highest_score": highest_score,
            "threshold": similarity_threshold,
            "passed_threshold": bool(results),
            "expanded_query": expanded_question,
            "raw_count": raw_count,
            "valid_count": valid_count,
            "discarded_invalid_count": discarded_invalid_count,
        }

    return results
