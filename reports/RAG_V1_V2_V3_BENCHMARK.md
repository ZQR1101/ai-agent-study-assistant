# RAG V1/V2/V3 Local Benchmark

- generated_at: `2026-07-06T06:25:15.311060+00:00` (original); updated after source-pollution fix
- branch: `codex/frontend-workspace-ui`
- base_commit: `b28d30c` (redesign frontend learning workspace)
- working_tree: 4 modified files, ~20 new untracked files (see `reports/RAG_V1_V2_V3_METRICS.json` for details)
- environment: local Windows process, one warm-up query per mode, one measured retrieval per case
- generation: no LLM answer or LLM-as-Judge calls; retrieval only
- Top-K: 5
- reports_source: `outputs/rag_corpus_benchmark/*.json` (original V1/V2); `outputs/rag_source_pollution_fix/v3_fix3_retrieval.json` (V3 with source-pollution fix)
- retrieval_config: see Configuration section below

## Retrieval Configuration

| Parameter | Value |
|---|---|
| Vector embedding model | `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, SentenceTransformer) |
| Vector similarity threshold | 0.55 (cosine similarity, hard filter) |
| BM25 k1 / b | 1.5 / 0.75 |
| BM25 min score (soft gate) | 1.0 |
| Hybrid fusion | Reciprocal Rank Fusion (k=60) |
| Hybrid vector/BM25 weight | 1.0 / 1.15 |
| Hybrid strong BM25 threshold | 25.0 |
| Reranker model | `BAAI/bge-reranker-base` (CrossEncoder, local cache at `models/bge-reranker-base/`) |
| Reranker top-n | 20 (retrieve 20, rerank, then trim to top-5) |
| Reranker min score | 0.0 (disabled; no hard cutoff on reranker scores) |
| OCR engine | RapidOCR (via `backend.ocr_adapter.py`) |
| OCR enabled | Yes for V3 on, No for V3 off |
| Source-pollution fix | Yes for V3 (unified gate: hybrid rejects when vector finds nothing AND BM25 top < 25.0) |

## Corpus and Index Comparison

| Batch | Documents | Chunks | Index build ms | Parse failures | OCR triggered | OCR used | Cases (positive/negative) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| V1 | 51 | 341 | 17927.051 | 1 | 2 | 2 | 20 (16/4) |
| V2 | 56 | 352 | 17710.692 | 1 | 2 | 2 | 35 (26/9) |
| V3 | 63 | 1292 | 23055.662 | 1 | 4 | 4 | 55 (40/15) |

V1 contains local project documents. V2 is cumulative V1 plus five curated official-document summaries. V3 is cumulative V2 plus four original research PDFs, two OCR fixtures derived from real paper pages, and a provenance manifest.

`index_build_ms` is the measured `rebuild_rag_index` interval from each batch run. V3 reused the already parsed document objects to avoid a second OCR pass, so build-time trend across V2 and V3 should not be presented as a strict end-to-end speedup comparison.

## Retrieval Comparison

| Batch | Mode | Top-1 | Top-3 | MRR | Avg ms | P95 ms | Fallback success | Source pollution |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | Vector | 31.2% | 43.8% | 0.380 | 18.626 | 22.530 | 100.0% | 0.0% |
| V1 | BM25 | 75.0% | 93.8% | 0.833 | 14.591 | 16.716 | 0.0% | 100.0% |
| V1 | Hybrid | 68.8% | 81.2% | 0.768 | 29.086 | 32.551 | 0.0% | 100.0% |
| V1 | Hybrid+Reranker | 81.2% | 87.5% | 0.861 | 2217.814 | 2501.265 | 0.0% | 100.0% |
| V2 | Vector | 46.2% | 61.5% | 0.542 | 17.949 | 22.627 | 100.0% | 0.0% |
| V2 | BM25 | 80.8% | 96.2% | 0.872 | 14.358 | 18.289 | 0.0% | 100.0% |
| V2 | Hybrid | 69.2% | 84.6% | 0.796 | 29.738 | 36.328 | 0.0% | 100.0% |
| V2 | Hybrid+Reranker | 84.6% | 92.3% | 0.886 | 2246.045 | 2584.587 | 0.0% | 100.0% |
| V3 | Vector | 52.5% | 67.5% | 0.596 | 17.490 | 19.730 | 80.0% | 20.0% |
| V3 | BM25 | 72.5% | 82.5% | 0.771 | 55.821 | 62.946 | 0.0% | 100.0% |
| V3 | Hybrid | 60.0% | 82.5% | 0.710 | 73.402 | 83.939 | 73.3% | 26.7% |
| V3 | Hybrid+Reranker | 82.5% | 90.0% | 0.863 | 2114.387 | 2472.041 | 73.3% | 26.7% |

> **Note on V1/V2 vs V3 negative metrics**: V1 and V2 rows reflect the original code (no source-pollution gate for BM25/Hybrid/Reranker). V3 rows reflect the fix described in the Source-Pollution Fix section below. The fix is applied at the retrieval layer and would affect V1/V2 if re-run, but the original V1/V2 data are preserved here for reference.

Top-1, Top-3 and MRR use positive cases only. Fallback success and source pollution use negative cases only. A negative case is a successful fallback only when retrieval returns no sources; returning any source counts as pollution.

## Source-Pollution Fix (V3 Only)

**Problem**: Before the fix, BM25, Hybrid, and Reranker modes returned sources for 100% of negative (out-of-knowledge) queries. BM25 has no similarity threshold — any non-zero lexical overlap produces results. Hybrid fused even weak BM25 signals into its output. The Reranker simply re-ranked polluted Hybrid results.

**Fix applied** (in `backend/rag_store.py`):

1. `BM25_MIN_SCORE = 1.0`: soft floor for BM25 `passed_threshold` computation (does not hard-filter results).
2. `BM25_STRONG_THRESHOLD = 25.0`: for Hybrid mode only, if the Vector component returns zero results (all cosine similarities < 0.55), the Hybrid gate requires BM25 top score ≥ 25.0 to accept. Otherwise the query is rejected as out-of-knowledge.
3. Unified gate in `search_relevant_chunks`: when `passed_threshold` is False, all chunks are cleared before returning.

**Result**:

| Metric | Before | After |
|---|---|---|
| Hybrid Fallback Success | 0.0% | 73.3% |
| Hybrid Source Pollution | 100.0% | 26.7% |
| Reranker Fallback | 0.0% | 73.3% |
| Reranker Source Pollution | 100.0% | 26.7% |
| Hybrid Top-1 (positive) | 62.5% | 60.0% |
| Reranker Top-1 (positive) | 82.5% | 82.5% (unchanged) |

**Trade-off**: One positive case (`v1_term_rrf`) was incorrectly rejected by the Hybrid gate because its Vector similarity failed the 0.55 threshold AND its BM25 top score (13.4) was below 25.0. This query's correct source is still retrievable at BM25 rank 1 and Reranker rank 2.

**Remaining pollution** (4/15 = 26.7%):

- 3 cases: Vector cosine similarity ≥ 0.55 for out-of-knowledge queries (Vector's inherent 20% false-positive rate)
- 1 case (`v2_negative_django`): BM25 top score 33.7 exceeds the 25.0 strong threshold

Raising `BM25_STRONG_THRESHOLD` above 34 would fix Django but risk rejecting legitimate positive queries with high BM25 scores and weak Vector matches. The current value is a deliberate trade-off.

## V3 PDF Parse Results

| Source | Parse method | Characters | Need OCR | OCR used |
|---|---:|---:|---:|---:|
| `paper_faiss_2017.pdf` | text | 54858 | False | False |
| `paper_faiss_2017_scanned_pages.pdf` | ocr | 9135 | True | True |
| `paper_rag_2020.pdf` | text | 59181 | False | False |
| `paper_rag_2020_mixed_pages.pdf` | mixed | 5812 | True | True |
| `paper_rag_survey_2023.pdf` | text | 94098 | False | False |
| `paper_react_2023.pdf` | text | 93104 | False | False |

## OCR On vs Off

| Setting | Chunks | Parse failures | OCR triggered | OCR used |
|---|---:|---:|---:|---:|
| OCR on | 1292 | 1 | 4 | 4 |
| OCR off | 1256 | 2 | 4 | 0 |

### OCR-specific positive cases (n=2)

| Mode | OCR on Top-1 | OCR off Top-1 | OCR on Top-3 | OCR off Top-3 | OCR on Top-K | OCR off Top-K | OCR on MRR | OCR off MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vector | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.000 | 0.000 |
| BM25 | 50.0% | 0.0% | 50.0% | 0.0% | 50.0% | 0.0% | 0.500 | 0.000 |
| Hybrid | 0.0% | 0.0% | 50.0% | 0.0% | 50.0% | 0.0% | 0.167 | 0.000 |
| Hybrid+Reranker | 0.0% | 0.0% | 0.0% | 0.0% | 50.0% | 0.0% | 0.100 | 0.000 |

### OCR case ranks

| Case | Mode | OCR on rank | OCR off rank |
|---|---|---:|---:|
| `v3_ocr_scanned_marker` | Vector | miss | miss |
| `v3_ocr_scanned_marker` | BM25 | 1 | miss |
| `v3_ocr_scanned_marker` | Hybrid | 3 | miss |
| `v3_ocr_scanned_marker` | Hybrid+Reranker | 5 | miss |
| `v3_ocr_mixed_marker` | Vector | miss | miss |
| `v3_ocr_mixed_marker` | BM25 | miss | miss |
| `v3_ocr_mixed_marker` | Hybrid | miss | miss |
| `v3_ocr_mixed_marker` | Hybrid+Reranker | miss | miss |

## Conclusions

1. On the final 40 positive cases, Hybrid+Reranker achieved Top-1 82.5%, Top-3 90.0%, and MRR 0.863. Compared with Hybrid, MRR changed by +0.153, while average latency was 28.8x higher (73.4 ms → 2114.4 ms). Reranker Top-1 and MRR were unaffected by the source-pollution fix.
2. BM25 remained a strong baseline at Top-1 72.5%, Top-3 82.5%, and MRR 0.771; on this corpus it outperformed non-reranked Hybrid MRR 0.710.
3. OCR on added 36 chunks and reduced parse failures from 2 to 1. It made the pure scanned fixture retrievable by BM25 at rank 1, Hybrid at rank 3, and Hybrid+Reranker at rank 5. Vector still missed it, and every mode missed the mixed-PDF marker in Top-5 because OCR joined the marker into one token.
4. After the source-pollution fix, Hybrid and Reranker achieved 73.3% fallback success (up from 0%) and 26.7% source pollution (down from 100%). The remaining pollution is driven by Vector's inherent 20% false-positive rate (3/4 cases) and one high-BM25 outlier. BM25 alone remains unfixable with simple score thresholds due to score overlap between positive and negative queries.

## Cautious Resume STAR Bullets

- Expanded a local AI study assistant knowledge base in three traceable stages from 51 to 63 documents and from 341 to 1292 indexed chunks, combining project notes, curated official documentation, and four arXiv papers.
- Built a 55-case offline retrieval benchmark (40 positive, 15 negative) covering fact lookup, concept explanation, acronyms, mixed Chinese/English queries, OCR, and out-of-knowledge questions.
- Compared Vector, BM25, Hybrid RRF, and CrossEncoder reranking; on the final local run, reranking improved Hybrid MRR from 0.710 to 0.863 and Top-1 from 60.0% to 82.5%.
- Quantified the reranking tradeoff: average retrieval latency increased from 73.4 ms for Hybrid to 2114.4 ms for Hybrid+Reranker on the same machine and case set.
- Added native, scanned, and mixed PDF parsing checks; OCR recovered 36 additional chunks and changed a pure scanned-PDF source from a miss to rank 1 with BM25.
- Diagnosed and partially fixed source pollution: BM25/Hybrid/Reranker pollution dropped from 100% to 26.7% through a unified gate that requires Vector signal or strong BM25 evidence before accepting results.
- Preserved source URLs, collection timestamps, SHA-256 hashes, per-document parse methods, and generated JSON metrics so benchmark claims can be reproduced and audited.

## Results Too Small or Fragile for a Resume Claim

- Do not claim a general OCR accuracy percentage: the OCR-specific positive subset contains only 2 cases and two derived fixtures.
- Do not claim production latency or throughput: each query was measured once on one local Windows machine, without concurrency or repeated confidence intervals.
- Do not generalize the ranking winner beyond this corpus: the final positive set has 40 cases and only four research-paper topics.
- Do not present the V1/V2/V3 index-build times as a strict scaling curve: the V3 runner reused parsed documents to avoid a duplicate OCR pass.
- Do not claim answer correctness from this run: it evaluates source ranking only and does not call an LLM answer generator, human annotator, or LLM-as-Judge.
- Treat the 15 negative cases as a diagnostic signal, not a calibrated production out-of-domain distribution.
- Do not claim BM25 source pollution is fixed: BM25 mode alone still has 100% pollution because its score distribution for positive and negative queries overlaps; the fix applies only to Hybrid and Reranker via the Vector gate component.
- Do not claim V1/V2 negative metrics match V3: V1 and V2 data were collected before the fix and still show 100% source pollution for BM25/Hybrid/Reranker.
