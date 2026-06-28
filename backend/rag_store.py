from pathlib import Path
from collections import Counter
import json
import math
import os
import re
import threading

from backend.config import get_embedding_model_settings


embedding_model = None
_embedding_model_lock = threading.Lock()
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
        with _embedding_model_lock:
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
_rag_index_lock = threading.Lock()
_bm25_index = None
_bm25_lock = threading.Lock()


PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs"
INDEX_DIR = PROJECT_ROOT / "rag_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

SIMILARITY_THRESHOLD = 0.55
MIN_CHUNK_LENGTH = 30
HYBRID_VECTOR_WEIGHT = 1.0
HYBRID_BM25_WEIGHT = 1.15


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


_RETRIEVAL_QUERY_NOISE_PATTERNS = (
    re.compile(r"(?:请问|请|麻烦)?(?:告诉我|给我讲讲|介绍一下|解释一下|说明一下)"),
    re.compile(r"(?:并且|并|同时|然后)?(?:请)?(?:给我|帮我)(?:一个|一份)?"),
    re.compile(r"(?:什么是|是什么)"),
)


def normalize_retrieval_query(query: str) -> str:
    original = " ".join(str(query or "").split()).strip()
    normalized = original

    for pattern in _RETRIEVAL_QUERY_NOISE_PATTERNS:
        normalized = pattern.sub(" ", normalized)

    normalized = re.sub(r"[，,。；;！？!?]+", " ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z0-9_.])(?=[\u4e00-\u9fff])", " ", normalized)
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9_./-])", " ", normalized)
    parts = normalized.split()
    normalized = " ".join(
        part
        for index, part in enumerate(parts)
        if index == 0 or part.casefold() != parts[index - 1].casefold()
    ).strip()
    return normalized or original


def expand_query(query: str) -> str:
    normalized_query = normalize_retrieval_query(query)
    lowered = normalized_query.lower()
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

    if re.search(r"(?<![a-z0-9_])rag(?![a-z0-9_])", lowered) or "检索增强生成" in normalized_query:
        expansions.extend(["RAG", "检索增强生成", "知识库问答", "retrieval augmented generation"])

    if (
        re.search(r"(?<![a-z0-9_])skills?(?![a-z0-9_])", lowered)
        or "技能包" in normalized_query
        or "工作流知识包" in normalized_query
    ):
        expansions.extend([
            "Agent Skill",
            "Agent Skills",
            "SKILL.md",
            "可复用工作流",
            "工作流知识包",
        ])

    if re.search(r"(?<![a-z0-9_])ocr(?![a-z0-9_])", lowered) or "文字识别" in normalized_query:
        expansions.extend(["OCR", "文字识别", "光学字符识别"])

    unique_expansions = []
    for item in expansions:
        if item not in normalized_query and item not in unique_expansions:
            unique_expansions.append(item)

    if not unique_expansions:
        return normalized_query

    return f"{normalized_query} {' '.join(unique_expansions)}"


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

    with _rag_index_lock:
        rag_index_error = None

        if index is not None:
            return True

        if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
            return False

        faiss = _get_faiss()
        loaded_index = faiss.read_index(str(INDEX_FILE))

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)

        index = loaded_index
        chunks = loaded_chunks
        _reset_bm25_index()

        print(f"已加载 FAISS RAG 索引，共 {len(chunks)} 个 chunks。")
        return True


def rebuild_rag_index():
    global chunks
    global index
    global rag_index_error

    rag_index_error = None

    print("正在构建 FAISS RAG 索引...")

    chunks = build_chunks()
    _reset_bm25_index()

    if not chunks:
        with _rag_index_lock:
            index = None
        _reset_bm25_index()
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

    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(embeddings)

    with _rag_index_lock:
        index = new_index
        save_rag_index()
    _reset_bm25_index()

    print(f"FAISS RAG 索引构建完成，共 {len(chunks)} 个 chunks，并已保存到本地。")


def ensure_rag_index():
    global index
    global chunks
    global rag_index_error

    with _rag_index_lock:
        if index is not None:
            return

    loaded = load_rag_index()

    if not loaded:
        try:
            rebuild_rag_index()
        except Exception as exc:
            with _rag_index_lock:
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


_BM25_TOKEN_PATTERN = re.compile(
    r"/[A-Za-z0-9_./-]+|[A-Za-z0-9_][A-Za-z0-9_./-]*|[\u4e00-\u9fff]+"
)
_BM25_CHINESE_STOP_WORDS = {
    "一个",
    "么是",
    "什么",
    "什么是",
    "介绍",
    "可以",
    "告诉",
    "告诉我",
    "如何",
    "并且",
    "并给",
    "怎么",
    "我一",
    "是否",
    "给我",
    "能否",
    "请帮",
    "请问",
}
_BM25_ASCII_ALIASES = {
    "skills": ("skill",),
}


def _append_ascii_bm25_token(tokens: list[str], token: str) -> None:
    if not token:
        return

    tokens.append(token)
    lowered = token.lower()
    if lowered != token:
        tokens.append(lowered)
    tokens.extend(_BM25_ASCII_ALIASES.get(lowered, ()))


def tokenize_for_bm25(text: str) -> list[str]:
    tokens = []

    for raw_token in _BM25_TOKEN_PATTERN.findall(str(text or "")):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw_token):
            chinese_tokens = []
            if 1 < len(raw_token) <= 8:
                chinese_tokens.append(raw_token)
            if len(raw_token) > 1:
                chinese_tokens.extend(
                    raw_token[index:index + 2]
                    for index in range(len(raw_token) - 1)
                )
            tokens.extend(
                token
                for token in chinese_tokens
                if token not in _BM25_CHINESE_STOP_WORDS
            )
            continue

        _append_ascii_bm25_token(tokens, raw_token)

        for part in re.split(r"[/._\-]+", raw_token):
            if not part or part == raw_token:
                continue
            _append_ascii_bm25_token(tokens, part)

    return [token for token in tokens if token]


def _chunk_id(chunk: dict, position: int) -> str:
    if chunk.get("chunk_id"):
        return str(chunk["chunk_id"])

    source = str(chunk.get("source") or "unknown")
    chunk_index = chunk.get("chunk_index", position)
    return f"{source}:{chunk_index}"


def _chunk_result(chunk: dict, position: int, score: float, retrieval: str) -> dict:
    text = str(chunk.get("text") or "")
    return {
        "source": chunk.get("source", ""),
        "score": float(score),
        "snippet": text,
        "text": text,
        "chunk_id": _chunk_id(chunk, position),
        "chunk_index": chunk.get("chunk_index", position),
        "retrieval": retrieval,
    }


def _chunks_fingerprint() -> tuple:
    return tuple((chunk.get("source"), hash(chunk.get("text", ""))) for chunk in chunks)


def _reset_bm25_index() -> None:
    global _bm25_index

    with _bm25_lock:
        _bm25_index = None


def _load_chunks_file_only() -> bool:
    global chunks
    global rag_index_error

    if not CHUNKS_FILE.exists():
        return False

    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        rag_index_error = str(exc)
        return False

    with _rag_index_lock:
        chunks = loaded_chunks
    _reset_bm25_index()
    return True


def _ensure_keyword_chunks() -> None:
    global chunks
    global rag_index_error

    if chunks:
        return

    if _load_chunks_file_only():
        return

    try:
        chunks = build_chunks()
        rag_index_error = None
    except Exception as exc:
        chunks = []
        rag_index_error = str(exc)
    finally:
        _reset_bm25_index()


def _build_bm25_index() -> dict:
    documents = [
        tokenize_for_bm25(
            f"{chunk.get('source', '')} "
            f"{str(chunk.get('source', '')).replace('_', ' ')} "
            f"{str(chunk.get('source', '')).replace('_', ' ')} "
            f"{chunk.get('text', '')}"
        )
        for chunk in chunks
    ]
    doc_freq = Counter()
    doc_term_counts = []
    doc_lengths = []

    for tokens in documents:
        counts = Counter(tokens)
        doc_term_counts.append(counts)
        doc_lengths.append(len(tokens))
        doc_freq.update(counts.keys())

    doc_count = len(documents)
    avg_doc_length = sum(doc_lengths) / doc_count if doc_count else 0.0
    idf = {
        term: math.log(1 + (doc_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in doc_freq.items()
    }

    return {
        "fingerprint": _chunks_fingerprint(),
        "idf": idf,
        "term_counts": doc_term_counts,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
    }


def _get_bm25_index() -> dict:
    global _bm25_index

    fingerprint = _chunks_fingerprint()
    with _bm25_lock:
        if _bm25_index is None or _bm25_index.get("fingerprint") != fingerprint:
            _bm25_index = _build_bm25_index()
        return _bm25_index


def _bm25_score(query_terms: list[str], bm25_index: dict, doc_position: int) -> float:
    term_counts = bm25_index["term_counts"][doc_position]
    doc_length = bm25_index["doc_lengths"][doc_position]
    avg_doc_length = bm25_index["avg_doc_length"] or 1.0
    idf = bm25_index["idf"]
    k1 = 1.5
    b = 0.75
    score = 0.0

    for term in set(query_terms):
        frequency = term_counts.get(term, 0)
        if frequency <= 0:
            continue

        denominator = frequency + k1 * (1 - b + b * doc_length / avg_doc_length)
        score += idf.get(term, 0.0) * (frequency * (k1 + 1) / denominator)

    return float(score)


def search_keyword_chunks(query: str, top_k: int = 10) -> list[dict]:
    _ensure_keyword_chunks()

    if not chunks:
        return []

    query_terms = tokenize_for_bm25(query)
    if not query_terms:
        return []

    bm25_index = _get_bm25_index()
    results = []
    for position, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "")
        if not is_valid_chunk(text):
            continue

        score = _bm25_score(query_terms, bm25_index, position)
        if score <= 0:
            continue

        item = _chunk_result(chunk, position, score, "bm25")
        item["bm25_score"] = float(score)
        results.append(item)

    results.sort(key=lambda item: item["bm25_score"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["bm25_rank"] = rank

    return results[:top_k]


def _empty_search_metadata(
    *,
    retrieval_mode: str,
    expanded_query: str,
    threshold,
    error: str | None = None,
    candidate_k: int | None = None,
) -> dict:
    return {
        "chunks": [],
        "highest_score": None,
        "threshold": threshold,
        "passed_threshold": False,
        "expanded_query": expanded_query,
        "raw_count": 0,
        "valid_count": 0,
        "discarded_invalid_count": 0,
        "error": error,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "vector_candidates": 0,
        "bm25_candidates": 0,
        "hybrid_used": False,
    }


def _search_vector_chunks_with_metadata(
    question: str,
    top_k: int = 10,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    global rag_index_error

    ensure_rag_index()
    expanded_question = expand_query(question)
    candidate_k = min(len(chunks), max(top_k * 5, top_k)) if chunks else top_k

    if rag_index_error or index is None or not chunks:
        return _empty_search_metadata(
            retrieval_mode="vector",
            expanded_query=expanded_question,
            threshold=similarity_threshold,
            error=rag_index_error,
            candidate_k=candidate_k,
        )

    try:
        model = get_embedding_model()
        faiss = _get_faiss()
        np = _get_numpy()
        question_embedding = model.encode([expanded_question])
        question_embedding = np.array(question_embedding).astype("float32")
    except Exception as exc:
        rag_index_error = str(exc)
        return _empty_search_metadata(
            retrieval_mode="vector",
            expanded_query=expanded_question,
            threshold=similarity_threshold,
            error=rag_index_error,
            candidate_k=candidate_k,
        )

    faiss.normalize_L2(question_embedding)
    scores, indices = index.search(question_embedding, candidate_k)

    highest_score = None
    raw_results = []
    vector_candidates = 0
    if len(indices[0]) > 0 and indices[0][0] != -1:
        highest_score = float(scores[0][0])

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue

        vector_candidates += 1
        if score < similarity_threshold:
            continue

        chunk = chunks[idx]
        item = _chunk_result(chunk, idx, score, "vector")
        item["vector_score"] = float(score)
        item["vector_rank"] = rank
        raw_results.append(item)

    valid_results = [item for item in raw_results if is_valid_chunk(item["text"])]
    results = valid_results[:top_k]
    if results:
        highest_score = max(float(item["score"]) for item in results)

    return {
        "chunks": results,
        "highest_score": highest_score,
        "threshold": similarity_threshold,
        "passed_threshold": bool(results) and highest_score is not None and highest_score >= similarity_threshold,
        "expanded_query": expanded_question,
        "raw_count": len(raw_results),
        "valid_count": len(results),
        "discarded_invalid_count": len(raw_results) - len(valid_results),
        "error": rag_index_error,
        "retrieval_mode": "vector",
        "candidate_k": candidate_k,
        "vector_candidates": vector_candidates,
        "bm25_candidates": 0,
        "hybrid_used": False,
    }


def search_vector_chunks(
    query: str,
    top_k: int = 10,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    return _search_vector_chunks_with_metadata(
        query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )["chunks"]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    *,
    k: int = 60,
    top_k: int = 3,
    weights: list[float] | None = None,
) -> list[dict]:
    fused = {}
    list_weights = weights or [1.0] * len(ranked_lists)

    for list_index, ranked_list in enumerate(ranked_lists):
        retrieval_name = "vector" if list_index == 0 else "bm25"
        weight = list_weights[list_index] if list_index < len(list_weights) else 1.0
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("chunk_id") or (item.get("source"), item.get("text"))
            if key not in fused:
                fused[key] = {
                    **item,
                    "score": 0.0,
                    "hybrid_score": 0.0,
                    "retrieval": "hybrid",
                }

            fused_item = fused[key]
            fused_item["score"] += weight / (k + rank)
            fused_item["hybrid_score"] = fused_item["score"]

            if retrieval_name == "vector":
                fused_item["vector_score"] = item.get("vector_score", item.get("score"))
                fused_item["vector_rank"] = item.get("vector_rank", rank)
            else:
                fused_item["bm25_score"] = item.get("bm25_score", item.get("score"))
                fused_item["bm25_rank"] = item.get("bm25_rank", rank)

    results = list(fused.values())
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _search_hybrid_chunks_with_metadata(
    question: str,
    top_k: int = 3,
    candidate_k: int | None = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    expanded_question = expand_query(question)
    search_k = candidate_k or max(top_k * 5, 10)
    vector_results = search_vector_chunks(
        expanded_question,
        top_k=search_k,
        similarity_threshold=similarity_threshold,
    )
    bm25_results = search_keyword_chunks(expanded_question, top_k=search_k)
    fused_results = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        top_k=top_k,
        weights=[HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT],
    )
    highest_score = max((float(item["score"]) for item in fused_results), default=None)

    return {
        "chunks": fused_results,
        "highest_score": highest_score,
        "threshold": None,
        "passed_threshold": bool(fused_results),
        "expanded_query": expanded_question,
        "raw_count": len(fused_results),
        "valid_count": len(fused_results),
        "discarded_invalid_count": 0,
        "error": rag_index_error,
        "retrieval_mode": "hybrid",
        "candidate_k": search_k,
        "vector_candidates": len(vector_results),
        "bm25_candidates": len(bm25_results),
        "hybrid_used": True,
    }


def search_hybrid_chunks(
    query: str,
    top_k: int = 3,
    candidate_k: int = 10,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    return _search_hybrid_chunks_with_metadata(
        query,
        top_k=top_k,
        candidate_k=candidate_k,
        similarity_threshold=similarity_threshold,
    )["chunks"]


def _search_keyword_chunks_with_metadata(question: str, top_k: int = 3) -> dict:
    expanded_question = expand_query(question)
    results = search_keyword_chunks(expanded_question, top_k=top_k)
    highest_score = max((float(item["score"]) for item in results), default=None)

    return {
        "chunks": results,
        "highest_score": highest_score,
        "threshold": None,
        "passed_threshold": bool(results),
        "expanded_query": expanded_question,
        "raw_count": len(results),
        "valid_count": len(results),
        "discarded_invalid_count": 0,
        "error": rag_index_error,
        "retrieval_mode": "bm25",
        "candidate_k": top_k,
        "vector_candidates": 0,
        "bm25_candidates": len(results),
        "hybrid_used": False,
    }


def search_relevant_chunks(
    question: str,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    include_metadata: bool = False,
    retrieval_mode: str = "vector",
    candidate_k: int | None = None,
):
    mode = retrieval_mode if retrieval_mode in {"vector", "bm25", "hybrid"} else "vector"

    if mode == "bm25":
        metadata = _search_keyword_chunks_with_metadata(question, top_k=top_k)
    elif mode == "hybrid":
        metadata = _search_hybrid_chunks_with_metadata(
            question,
            top_k=top_k,
            candidate_k=candidate_k,
            similarity_threshold=similarity_threshold,
        )
    else:
        metadata = _search_vector_chunks_with_metadata(
            question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    if include_metadata:
        return metadata

    return metadata["chunks"]
