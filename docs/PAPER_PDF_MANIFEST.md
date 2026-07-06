# V3 Paper PDF and OCR Fixture Manifest

- source: arXiv paper pages and PDFs listed below
- collected_at: `2026-07-06 Asia/Shanghai`
- fixture_script: `scripts/create_pdf_ocr_fixtures.py`

## Original research PDFs

| Local file | Paper | Original URL | SHA-256 | Expected parse path |
|---|---|---|---|---|
| `paper_rag_2020.pdf` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | https://arxiv.org/abs/2005.11401 | `23E3249E9A1E75418D82EFECAB0EA8C4D033B89C93742F63208D47CE01F21233` | native text |
| `paper_faiss_2017.pdf` | Billion-scale similarity search with GPUs | https://arxiv.org/abs/1702.08734 | `89B2371B261846059D416BDF1423B15A6E3CC7D1B2D5DC2729EE5F6C3399049A` | native text |
| `paper_react_2023.pdf` | ReAct: Synergizing Reasoning and Acting in Language Models | https://arxiv.org/abs/2210.03629 | `F285B0971AE4A790E402FB93966BED3ADDE2CF0A04977D08B2B40D6AB0CACE69` | native text |
| `paper_rag_survey_2023.pdf` | Retrieval-Augmented Generation for Large Language Models: A Survey | https://arxiv.org/abs/2312.10997 | `396A0FADEB4CD40F5C8CCC36B73A0815F6CB4D7F6BFA53B6C48C1F9ABA7C7E02` | native text |

The files were downloaded from the official arXiv PDF endpoints. The hashes identify the exact files used by this local benchmark.

## OCR fixtures derived from real paper pages

| Local file | Derivation | SHA-256 | Native text before OCR | Verified parse path |
|---|---|---|---:|---|
| `paper_faiss_2017_scanned_pages.pdf` | First two FAISS paper pages rendered as images; the first image includes a unique OCR benchmark marker | `51CDB8C983D2195CFE1FFD6D8BB51A314EBA201FB54983BDB9C97C8A169A0953` | 0 characters | OCR, marker recognized |
| `paper_rag_2020_mixed_pages.pdf` | First RAG paper page kept as native PDF; the second image page includes a different OCR benchmark marker | `FE3D4B162D3BA944DD0E57124F3B798A33CCAB7691713013AF4FB8D9ED730850` | 2,898 characters | mixed native text + OCR, marker recognized |

The exact marker strings are intentionally omitted from this indexable manifest. They exist only in the fixture images and the generator source, so retrieval cannot answer an OCR test from this metadata file instead of the parsed PDF.
