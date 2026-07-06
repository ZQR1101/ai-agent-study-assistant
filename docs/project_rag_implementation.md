# 当前项目的 RAG 实现

- source: `backend/rag_store.py`, `backend/rag_service.py`, `backend/reranker.py`
- source_commit: `b28d30c`
- curated_at: `2026-07-06 Asia/Shanghai`

## 索引构建

`rag_store.load_documents()` 扫描 `docs/` 顶层的 `.txt`、`.md` 和 `.pdf`。PDF 先尝试普通文本解析；低文本页、扫描件或损坏文件可进入 OCR fallback。文本按 500 字符切块，重叠 100 字符，并经过最小长度和有效字符比例过滤。

向量索引使用 SentenceTransformer 生成 float32 embedding，L2 归一化后写入 FAISS `IndexFlatIP`。因此当前向量分数是归一化向量的内积，可按余弦相似度理解。索引和 chunk 元数据分别保存到 `rag_index/index.faiss` 与 `rag_index/chunks.json`。

## 三种召回模式

- Vector：对 expanded query 编码，在 FAISS 中检索，并使用 `SIMILARITY_THRESHOLD=0.55` 过滤低分候选。
- BM25：对中英文混合文本分词，在内存 BM25 索引上计算关键词相关性。
- Hybrid：分别取 Vector 与 BM25 候选，再用 Reciprocal Rank Fusion 合并；当前权重为 Vector 1.0、BM25 1.15。

Hybrid 的融合分数不是余弦相似度，不能直接套用 Vector 的 0.55 阈值。当前 BM25 和 Hybrid 只要有结果就视为通过，这也是负样本 source pollution 需要单独评测的原因。

## Reranker 与 fallback

Reranker 只对候选 chunks 精排，不扫描全库。它默认由请求字段控制，同时还要求服务端 `ENABLE_RERANKER=true` 且配置模型。模型使用 CrossEncoder 懒加载；加载失败时返回原召回顺序并记录 `reranker_error`。

`rag_service.get_rag_context()` 把命中结果转换为 context 和 sources。Vector 未过阈值时返回空来源；回答层使用固定 fallback 文案，不把弱相关 chunk 注入模型。

