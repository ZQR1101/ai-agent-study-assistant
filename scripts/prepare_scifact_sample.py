import json
import random
from pathlib import Path

RAW_DIR = Path("docs/raw/beir_scifact")
CORPUS_FILE = RAW_DIR / "corpus.jsonl"
QUERIES_FILE = RAW_DIR / "queries.jsonl"
QRELS_FILE = RAW_DIR / "qrels" / "test.tsv"

OUT_DOCS_DIR = Path("docs/curated/beir_scifact_docs")
OUT_EVAL_DIR = Path("eval_data/beir_scifact")

NUM_QUERIES = 100
NUM_RANDOM_DISTRACTORS = 200

random.seed(42)


def load_jsonl(path: Path) -> dict:
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[str(item["_id"])] = item
    return data


def load_qrels(path: Path) -> dict:
    """
    返回格式：
    {
        query_id: [corpus_id1, corpus_id2, ...]
    }
    """
    qrels = {}

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip().split("\t")

        # 兼容有表头和无表头两种情况
        has_header = first_line[0].lower() in {"query-id", "query_id", "qid"}

        if not has_header:
            parts = first_line
            if len(parts) >= 3:
                qid, cid, score = parts[0], parts[1], parts[2]
                if int(score) > 0:
                    qrels.setdefault(str(qid), []).append(str(cid))

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            qid, cid, score = parts[0], parts[1], parts[2]

            try:
                score = int(score)
            except ValueError:
                continue

            if score > 0:
                qrels.setdefault(str(qid), []).append(str(cid))

    return qrels


def safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def main():
    OUT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(CORPUS_FILE)
    queries = load_jsonl(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    # 只选择有标准答案的 query
    available_query_ids = [qid for qid in qrels.keys() if qid in queries]
    selected_query_ids = available_query_ids[:NUM_QUERIES]

    positive_doc_ids = set()
    for qid in selected_query_ids:
        for cid in qrels[qid]:
            if cid in corpus:
                positive_doc_ids.add(cid)

    # 加一些干扰文档，这样检索任务才真实
    all_doc_ids = list(corpus.keys())
    distractor_ids = set(random.sample(all_doc_ids, min(NUM_RANDOM_DISTRACTORS, len(all_doc_ids))))

    selected_doc_ids = positive_doc_ids | distractor_ids

    # 1. 导出 markdown 文档
    for cid in selected_doc_ids:
        item = corpus[cid]
        title = item.get("title", "").strip()
        text = item.get("text", "").strip()

        content = f"""# {title}

{text}

---

source: BEIR SciFact
corpus_id: {cid}
"""

        filename = f"scifact_{safe_filename(cid)}.md"
        output_path = OUT_DOCS_DIR / filename
        output_path.write_text(content, encoding="utf-8")

    # 2. 导出测试 query
    with (OUT_EVAL_DIR / "queries_sample.jsonl").open("w", encoding="utf-8") as f:
        for qid in selected_query_ids:
            item = queries[qid]
            f.write(json.dumps({
                "query_id": qid,
                "text": item.get("text", "")
            }, ensure_ascii=False) + "\n")

    # 3. 导出标准答案 qrels
    with (OUT_EVAL_DIR / "qrels_sample.tsv").open("w", encoding="utf-8") as f:
        f.write("query_id\tcorpus_id\tscore\n")
        for qid in selected_query_ids:
            for cid in qrels[qid]:
                if cid in selected_doc_ids:
                    f.write(f"{qid}\t{cid}\t1\n")

    print("Done.")
    print(f"Exported docs: {len(selected_doc_ids)}")
    print(f"Exported queries: {len(selected_query_ids)}")
    print(f"Docs path: {OUT_DOCS_DIR}")
    print(f"Eval path: {OUT_EVAL_DIR}")


if __name__ == "__main__":
    main()