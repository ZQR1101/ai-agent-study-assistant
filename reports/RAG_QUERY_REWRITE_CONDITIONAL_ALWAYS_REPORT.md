# RAG Query Rewrite: Conditional vs Always Benchmark

## Scope

- Date: 2026-07-10
- Cases: 55 total (40 positive, 15 negative)
- Corpus: 359 documents, 2,579 indexed chunks
- Retrieval modes: Vector, BM25, Hybrid, Hybrid + Reranker
- Conditional report: `outputs/rag_query_rewrite_modes/v3_rewrite_conditional_retrieval.json`
- Always report: `outputs/rag_query_rewrite_modes/v3_rewrite_always_retrieval.json`
- Rewrite model: `deepseek-v4-pro`

The existing 55-case benchmark contains no `history_context`. Therefore, `conditional` correctly attempted zero rewrites. This run verifies that self-contained queries are not rewritten, but it does not measure the benefit of conditional rewrite for conversational follow-up questions.

The `always` run rewrote each unique query once and reused the same rewritten query across all four retrieval modes. It made 55 API calls rather than 220 repeated calls.

## Positive And Negative Metrics

| Mode | Policy | Top-1 | Top-3 | MRR | Fallback Success | Source Pollution |
|---|---|---:|---:|---:|---:|---:|
| Vector | Conditional | 57.5% | 65.0% | 0.617 | 86.7% | 13.3% |
| Vector | Always | 57.5% | 65.0% | 0.613 | 86.7% | 13.3% |
| BM25 | Conditional | 75.0% | 87.5% | 0.812 | 40.0% | 60.0% |
| BM25 | Always | 72.5% | 87.5% | 0.796 | 33.3% | 66.7% |
| Hybrid | Conditional | 72.5% | 85.0% | 0.799 | 80.0% | 20.0% |
| Hybrid | Always | 72.5% | 87.5% | 0.808 | 80.0% | 20.0% |
| Hybrid + Reranker | Conditional | **90.0%** | **97.5%** | **0.938** | **80.0%** | **20.0%** |
| Hybrid + Reranker | Always | **90.0%** | **97.5%** | 0.933 | **80.0%** | **20.0%** |

## Delta: Always Minus Conditional

| Mode | Top-1 | Top-3 | MRR | Fallback Success | Source Pollution |
|---|---:|---:|---:|---:|---:|
| Vector | 0.0 pts | 0.0 pts | -0.004 | 0.0 pts | 0.0 pts |
| BM25 | -2.5 pts | 0.0 pts | -0.016 | -6.7 pts | +6.7 pts |
| Hybrid | 0.0 pts | +2.5 pts | +0.009 | 0.0 pts | 0.0 pts |
| Hybrid + Reranker | 0.0 pts | 0.0 pts | -0.005 | 0.0 pts | 0.0 pts |

## Rewrite Diagnostics

| Metric | Conditional | Always |
|---|---:|---:|
| Unique queries | 55 | 55 |
| Rewrite attempts | 0 | 55 |
| Rewrite successes | 0 | 55 |
| Rewrite success rate | N/A | 100.0% |
| Rewrite fallback count | 0 | 0 |
| Actual API calls | 0 | 55 |
| Average rewrite latency | 0.0 ms | 5,607.6 ms |
| P95 rewrite latency | 0.0 ms | 13,622.2 ms |
| Query fusion count | 0 | 55 per retrieval mode |

## Latency

The generated always JSON stores rewrite latency separately and used a cache across retrieval modes. For a fair user-facing comparison, the table below reports:

- Conditional: measured retrieval latency; no rewrite was attempted.
- Always retrieval-only: local dual-query retrieval and fusion after rewrite.
- Always end-to-end: rewrite latency plus retrieval-only latency.

| Mode | Conditional Avg / P95 | Always Retrieval-only Avg / P95 | Always End-to-end Avg / P95 |
|---|---:|---:|---:|
| Vector | 19.2 / 24.2 ms | 35.7 / 52.4 ms | 5,643.3 / 13,655.0 ms |
| BM25 | 87.2 / 110.1 ms | 197.0 / 281.4 ms | 5,804.6 / 13,827.6 ms |
| Hybrid | 102.1 / 125.7 ms | 236.0 / 329.3 ms | 5,843.7 / 13,861.4 ms |
| Hybrid + Reranker | 1,628.6 / 3,121.3 ms | 3,259.9 / 6,094.1 ms | 8,867.6 / 17,616.4 ms |

## Case-Level Changes

- Vector regression: `v1_fact_chunking` moved from rank 5 to a miss.
- BM25 regression: `v1_fact_chunking` moved from rank 1 to rank 3.
- BM25 pollution regression: `v1_negative_blockchain` changed from fallback to polluted retrieval.
- Hybrid improvements: `v1_fact_vector_threshold` moved from rank 5 to rank 3; `v3_survey_motivation` moved from a miss to rank 4.
- Hybrid + Reranker regression: `v1_mixed_stategraph` moved from rank 2 to rank 3, which accounts for the small MRR decrease.

## PDF/OCR And Chunk Quality

- Parse failures: 0
- OCR-triggered documents: 3
- OCR-used documents: 3
- Chunks retained: 2,566
- Low-quality chunks retained and marked: 13
- Chunks dropped by quality filter: 77

OCR retention and the chunk quality filter remained operational in both runs.

## Conclusion

1. `conditional` caused no regression on the current single-turn benchmark because it correctly skipped all 55 self-contained queries. A separate multi-turn benchmark with explicit history is still required to measure its intended benefit.
2. `always` did not improve the final Hybrid + Reranker Top-1, Top-3, fallback success, or source pollution. MRR decreased from 0.938 to 0.933.
3. Original-query plus rewritten-query RRF fusion successfully prevented the large recall regression seen in the earlier rewrite-only ablation: final Top-1 stayed at 90.0% instead of falling to 82.5%.
4. Full rewrite remains unsuitable as the production default because average end-to-end Hybrid + Reranker latency increased from 1.63 seconds to 8.87 seconds, with a 17.62-second P95.
5. Keep the production default at `off` until a conversational benchmark exists. Use `conditional` for a controlled grey rollout and `always` only for ablation testing.
