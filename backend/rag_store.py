from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

chunks = []
index = None


def load_documents():
    docs_path = Path(__file__).parent.parent / "docs"
    documents = []

    for file_path in docs_path.iterdir():

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

    embeddings = embedding_model.encode(chunk_texts)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS RAG 索引构建完成，共 {len(chunks)} 个 chunks。")


def search_relevant_chunks(question: str, top_k: int = 3):
    global index
    global chunks

    if index is None:
        rebuild_rag_index()

    if index is None or not chunks:
        return []

    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(question_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        if score < 0.25:
            continue

        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results