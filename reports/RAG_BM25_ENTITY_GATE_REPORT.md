# BM25 Term Coverage / Entity Matching Report

- generated_at: 2026-07-10
- benchmark: V3 full retrieval cases, 55 cases total = 40 positive / 15 negative
- top_k: 5
- modes: `BM25`, `Hybrid`, `Hybrid+Reranker`
- implementation: BM25 candidate gate using term coverage and entity matching
- output_dir: `outputs/rag_bm25_entity_gate/`

## Scope Note

This run was executed on the current local working tree. The indexed corpus is larger than the previous 63-document V3 report because the current `docs/` workspace contains additional documents.

| Metric | Value |
|---|---:|
| Documents | 359 |
| Chunks | 2579 |
| Parse failures | 0 |
| OCR triggered | 3 |
| OCR used | 3 |
| Low-quality chunks flagged | 13 |
| Chunks dropped by quality filter | 77 |

To make the BM25 change comparable, the report uses two runs on the same rebuilt index:

1. `without_gate`: BM25 term/entity gate monkeypatched off.
2. `with_gate`: BM25 term/entity gate enabled.

## Implementation Summary

The BM25 path now computes match diagnostics for each candidate:

- `bm25_term_coverage`: matched unique BM25 terms / unique query terms
- `bm25_entity_match_count`: matched entity terms
- `bm25_entity_term_coverage`: matched entity terms / query entity terms

Candidate filtering rule:

- If the query has entity terms such as acronyms, English terms, API paths, file names, or numbers, at least one entity must match and entity coverage must pass the floor.
- If the query has no entity terms, the candidate must pass a minimum term coverage floor.

This targets false positives such as a query mentioning `Kubernetes HPA` matching a generic project deployment document only because it shares vague Chinese terms like `项目`.

## Retrieval Metrics

### Without BM25 Gate

| Mode | Top-1 | Top-3 | MRR | Fallback Success | Source Pollution | Avg ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM25` | 72.5% | 80.0% | 0.762 | 0.0% | 100.0% | 106.294 | 127.442 |
| `Hybrid` | 75.0% | 85.0% | 0.800 | 66.7% | 33.3% | 115.047 | 144.422 |
| `Hybrid+Reranker` | 90.0% | 97.5% | 0.933 | 66.7% | 33.3% | 2304.196 | 3579.368 |

### With BM25 Gate

| Mode | Top-1 | Top-3 | MRR | Fallback Success | Source Pollution | Avg ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM25` | 75.0% | 87.5% | 0.812 | 40.0% | 60.0% | 122.863 | 198.168 |
| `Hybrid` | 72.5% | 85.0% | 0.799 | 80.0% | 20.0% | 137.797 | 216.971 |
| `Hybrid+Reranker` | 90.0% | 97.5% | 0.938 | 80.0% | 20.0% | 1778.766 | 2969.751 |

## Delta

| Mode | Top-1 Delta | Top-3 Delta | MRR Delta | Fallback Delta | Pollution Delta |
|---|---:|---:|---:|---:|---:|
| `BM25` | +2.5 pts | +7.5 pts | +0.050 | +40.0 pts | -40.0 pts |
| `Hybrid` | -2.5 pts | 0.0 pts | -0.001 | +13.3 pts | -13.3 pts |
| `Hybrid+Reranker` | 0.0 pts | 0.0 pts | +0.005 | +13.3 pts | -13.3 pts |

## Negative Cases

### Fixed Source Pollution

| Mode | Cases fixed |
|---|---|
| `BM25` | `v1_negative_blockchain`, `v1_negative_ios`, `v1_negative_kubernetes`, `v2_negative_celery`, `v3_negative_cv`, `v3_negative_graph_neural_network` |
| `Hybrid` | `v1_negative_kubernetes`, `v2_negative_prometheus` |
| `Hybrid+Reranker` | `v1_negative_kubernetes`, `v2_negative_prometheus` |

### Remaining Source Pollution

| Mode | Remaining cases |
|---|---|
| `BM25` | `v1_negative_kafka`, `v2_negative_django`, `v2_negative_grpc`, `v2_negative_prometheus`, `v2_negative_redis`, `v3_negative_database`, `v3_negative_diffusion`, `v3_negative_quantum`, `v3_negative_speech` |
| `Hybrid` | `v2_negative_django`, `v3_negative_graph_neural_network`, `v3_negative_quantum` |
| `Hybrid+Reranker` | `v2_negative_django`, `v3_negative_graph_neural_network`, `v3_negative_quantum` |

## Conclusion

The BM25 term/entity gate is positive overall.

- BM25-only source pollution improved from 100.0% to 60.0%, while Top-1, Top-3 and MRR improved on the 40 positive cases.
- Hybrid source pollution improved from 33.3% to 20.0%; Top-3 stayed flat, MRR was effectively unchanged, and Top-1 dropped by 2.5 pts.
- Hybrid+Reranker kept Top-1 and Top-3 unchanged, improved MRR slightly, and reduced source pollution from 33.3% to 20.0%.

The remaining pollution cases suggest the next refinement should handle semantically adjacent but out-of-scope entities. A useful next step is to add stricter entity coverage for multi-entity queries and a small allowlist of high-value domain aliases to avoid over-filtering positive cases.

