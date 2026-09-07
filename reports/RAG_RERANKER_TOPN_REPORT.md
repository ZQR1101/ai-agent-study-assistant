# Reranker Top-N Optimization Report

## Scope

- Corpus: existing FAISS index, 359 documents and 2,579 chunks
- Cases: 55 total, 40 positive and 15 negative
- Retrieval path: Hybrid + CrossEncoder Reranker
- Final `top_k`: 5
- Reranker model: `bge-reranker-base`
- No documents were re-parsed and no index was rebuilt during this comparison.

## Results

| Reranker Top-N | Top-1 | Top-3 | MRR | Fallback Success | Source Pollution | Avg Latency | P95 Latency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 82.5% | 90.0% | 0.863 | 80.0% | 20.0% | 1063.1 ms | 1516.9 ms |
| **15** | **90.0%** | **97.5%** | **0.938** | **80.0%** | **20.0%** | **1274.8 ms** | **2025.6 ms** |
| 20 | 90.0% | 97.5% | 0.938 | 80.0% | 20.0% | 1711.8 ms | 2995.2 ms |

## Delta: Top-N 15 vs 20

- Top-1: `0.0 pts`
- Top-3: `0.0 pts`
- MRR: `0.000`
- Fallback Success: `0.0 pts`
- Source Pollution: `0.0 pts`
- Average latency: `-436.9 ms` (`-25.5%`)
- P95 latency: `-969.6 ms` (`-32.4%`)

## Decision

`top_n=10` is too aggressive for this corpus: it saves latency but loses 7.5 points on Top-1 and 0.075 MRR. `top_n=15` preserves every measured retrieval and negative-sample metric from `top_n=20` while materially reducing CrossEncoder work. The default is therefore set to 15, with `RERANKER_TOP_N` remaining configurable for future corpus-specific tuning.

This is an empirical setting, not a universal threshold. A substantially different corpus, embedding model, or reranker model should repeat the same Top-N sweep before changing the default.
