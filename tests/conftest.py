import os


# Pytest must remain hermetic even when a developer's local .env enables RAG.
os.environ.setdefault("ENABLE_RAG_AUTO_BUILD", "false")
os.environ.setdefault("ENABLE_RAG_WARMUP", "false")
os.environ.setdefault("EMBEDDING_MODEL_LOCAL_ONLY", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
