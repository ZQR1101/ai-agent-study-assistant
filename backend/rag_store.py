from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re


embedding_model = None


def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    return embedding_model

chunks = []
index = None


PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs"
INDEX_DIR = PROJECT_ROOT / "rag_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

SIMILARITY_THRESHOLD = 0.55


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

            if chunk.strip():
                new_chunks.append({
                    "source": source,
                    "text": chunk.strip()
                })

    return new_chunks


def save_rag_index():
    INDEX_DIR.mkdir(exist_ok=True)

    if index is not None:
        faiss.write_index(index, str(INDEX_FILE))

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_rag_index():
    global index
    global chunks

    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        return False

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"已加载 FAISS RAG 索引，共 {len(chunks)} 个 chunks。")
    return True


def rebuild_rag_index():
    global chunks
    global index

    print("正在构建 FAISS RAG 索引...")

    chunks = build_chunks()

    if not chunks:
        index = None
        print("知识库为空，未构建索引。")
        return

    chunk_texts = [chunk["text"] for chunk in chunks]

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

    if index is not None:
        return

    loaded = load_rag_index()

    if not loaded:
        rebuild_rag_index()


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

    ensure_rag_index()

    if index is None or not chunks:
        if include_metadata:
            return {
                "chunks": [],
                "highest_score": None,
                "threshold": similarity_threshold,
                "passed_threshold": False,
            }
        return []

    model = get_embedding_model()
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    faiss.normalize_L2(question_embedding)

    search_k = min(len(chunks), max(top_k * 5, top_k))
    scores, indices = index.search(question_embedding, search_k)

    highest_score = None
    results = []

    if len(indices[0]) > 0 and indices[0][0] != -1:
        highest_score = float(scores[0][0])

    if highest_score is None or highest_score < similarity_threshold:
        keyword_results = _keyword_relevant_chunks(
            question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if keyword_results:
            keyword_highest_score = max(item["score"] for item in keyword_results)
            if include_metadata:
                return {
                    "chunks": keyword_results,
                    "highest_score": keyword_highest_score,
                    "threshold": similarity_threshold,
                    "passed_threshold": True,
                }
            return keyword_results

        if include_metadata:
            return {
                "chunks": [],
                "highest_score": highest_score,
                "threshold": similarity_threshold,
                "passed_threshold": False,
            }
        return []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        if score < similarity_threshold:
            continue

        chunk = chunks[idx]

        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(score)
        })

        if len(results) >= top_k:
            break

    if include_metadata:
        return {
            "chunks": results,
            "highest_score": highest_score,
            "threshold": similarity_threshold,
            "passed_threshold": bool(results),
        }

    return results
