# FAISS 简介

FAISS 是 Facebook AI Research 开发的向量相似度搜索库，常用于 RAG 系统中的向量检索。

在 RAG 中，FAISS 的作用是保存文档 chunk 的 embedding，并根据用户问题快速找到最相似的文本片段。

FAISS 可以让知识库问答系统避免每次都重新计算所有文档向量，从而提升检索速度。
