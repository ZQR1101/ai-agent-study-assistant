from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


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

SIMILARITY_THRESHOLD = 0.30


def load_documents():
    documents = []

    if not DOCS_PATH.exists():
        DOCS_PATH.mkdir(exist_ok=True)
        return documents

    for file_path in DOCS_PATH.iterdir():

        if file_path.suffix in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({
                    "source": file_path.name,
                    "text": f.read()
                })

        elif file_path.suffix == ".pdf":
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


def search_relevant_chunks(question: str, top_k: int = 3):
    global index
    global chunks

    ensure_rag_index()

    if index is None or not chunks:
        return []

    model = get_embedding_model()
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(question_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[idx]

        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results
