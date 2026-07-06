# FAISS Getting Started

- source: `https://github.com/facebookresearch/faiss/wiki/Getting-started`
- curated_at: `2026-07-06 Asia/Shanghai`
- publisher: `facebookresearch/faiss official GitHub wiki`

FAISS 面向固定维度的稠密向量集合。输入矩阵按 row-major 存放，并使用 32-bit float。数据库向量通常记为 `xb`，查询向量记为 `xq`，两者的第二个维度必须等于索引维度 `d`。

FAISS 的核心抽象是 `Index`。最简单的 `IndexFlatL2` 做精确 L2 距离搜索，不需要训练；向量通过 `add()` 加入，通过 `search()` 做 k-nearest-neighbor 查询。搜索返回距离矩阵和邻居 ID 矩阵，每个查询对应按距离排序的 k 个结果。

部分索引需要训练，部分索引支持显式整数 ID。选择索引时需要区分精确搜索、近似搜索、内存占用和训练成本。

本项目使用 `IndexFlatIP`。文档向量和查询向量都先做 L2 归一化，因此 inner product 可作为 cosine similarity；索引位置与 `chunks.json` 中 chunk 顺序对应。

