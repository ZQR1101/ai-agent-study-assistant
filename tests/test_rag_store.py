import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pypdf import PdfWriter

from backend import rag_store


class OCRRagStoreTests(unittest.TestCase):
    def setUp(self):
        self.original_chunks = rag_store.chunks
        self.original_index = rag_store.index

    def tearDown(self):
        rag_store.chunks = self.original_chunks
        rag_store.index = self.original_index
        rag_store._reset_bm25_index()

    def test_txt_document_metadata_remains_plain_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_path = Path(tmpdir)
            (docs_path / "notes.txt").write_text(
                "Retrieval augmented generation uses relevant context before answering. " * 3,
                encoding="utf-8",
            )
            with patch.object(rag_store, "DOCS_PATH", docs_path):
                chunks, _ = rag_store.build_chunks()

        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["parse_method"], "text")
        self.assertFalse(chunks[0]["ocr_used"])
        self.assertFalse(chunks[0]["need_ocr"])

    def test_load_documents_recurses_and_skips_raw_corpus_files(self):
        parse_result = {
            "text": "Loaded document content for recursive knowledge base scanning.",
            "method": "text",
            "ocr_used": False,
            "need_ocr": False,
            "text_char_count": 61,
            "warnings": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_path = Path(tmpdir)
            (docs_path / "root.md").write_text("root", encoding="utf-8")
            nested = docs_path / "nested"
            nested.mkdir()
            (nested / "guide.txt").write_text("nested", encoding="utf-8")
            (nested / "ignored.jsonl").write_text("{}", encoding="utf-8")
            raw = docs_path / "raw"
            raw.mkdir()
            (raw / "corpus.jsonl").write_text("answer leakage", encoding="utf-8")
            (raw / "queries.tsv").write_text("query leakage", encoding="utf-8")

            with (
                patch.object(rag_store, "DOCS_PATH", docs_path),
                patch(
                    "backend.rag_store.extract_text_from_document",
                    return_value=parse_result,
                ) as extract,
            ):
                documents = rag_store.load_documents()

        self.assertEqual(
            [document["source"] for document in documents],
            ["nested/guide.txt", "root.md"],
        )
        self.assertEqual(extract.call_count, 2)

    def test_ocr_metadata_is_copied_to_chunks(self):
        document = {
            "source": "scan.pdf",
            "text": "recognized scanned document content for retrieval " * 4,
            "parse_method": "ocr",
            "ocr_used": True,
            "need_ocr": True,
        }
        with patch("backend.rag_store.load_documents", return_value=[document]):
            chunks, _ = rag_store.build_chunks()

        self.assertEqual(chunks[0]["parse_method"], "ocr")
        self.assertTrue(chunks[0]["ocr_used"])
        self.assertTrue(chunks[0]["need_ocr"])

    def test_build_chunks_can_reuse_preparsed_documents(self):
        documents = [{
            "source": "preparsed.md",
            "text": "Preparsed retrieval content is reused without parsing the document again. " * 3,
            "parse_method": "text",
            "ocr_used": False,
            "need_ocr": False,
        }]
        with patch("backend.rag_store.load_documents") as load_documents:
            chunks, _ = rag_store.build_chunks(documents=documents)

        load_documents.assert_not_called()
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["source"], "preparsed.md")

    def test_build_chunks_uses_heading_paragraph_metadata(self):
        documents = [{
            "source": "rag_handbook.md",
            "text": (
                "# RAG Handbook\n\n"
                + "## Retrieval Pipeline\n\n"
                + "Dense retrieval uses vector indexes for semantic matching and candidate recall. " * 2
                + "\n\n## Evaluation\n\n"
                + "Offline benchmarks measure source hit rate, MRR, and fallback behavior. " * 2
            ),
            "parse_method": "text",
            "ocr_used": False,
            "need_ocr": False,
        }]

        chunks, _ = rag_store.build_chunks(documents=documents)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["document"], "rag_handbook.md")
        self.assertEqual(chunks[0]["document_title"], "RAG Handbook")
        self.assertEqual(chunks[0]["title"], "Retrieval Pipeline")
        self.assertEqual(chunks[0]["section"], "RAG Handbook > Retrieval Pipeline")
        self.assertEqual(chunks[1]["title"], "Evaluation")

    def test_rapidocr_document_text_enters_chunks_and_keyword_search(self):
        parse_result = {
            "text": (
                "Scanned invoice reference RAPID-OCR-2026 contains a payable total "
                "and enough recognized text for retrieval. " * 3
            ),
            "method": "ocr",
            "need_ocr": True,
            "ocr_used": True,
            "page_count": 1,
            "text_char_count": 300,
            "ocr_error": None,
            "warnings": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_path = Path(tmpdir)
            (docs_path / "scan.pdf").write_bytes(b"%PDF-mocked")
            with (
                patch.object(rag_store, "DOCS_PATH", docs_path),
                patch(
                    "backend.rag_store.extract_text_from_document",
                    return_value=parse_result,
                ),
            ):
                chunks, _ = rag_store.build_chunks()

        self.assertTrue(chunks)
        self.assertIn("RAPID-OCR-2026", chunks[0]["text"])
        self.assertEqual(chunks[0]["parse_method"], "ocr")
        self.assertTrue(chunks[0]["ocr_used"])

        rag_store.chunks = chunks
        rag_store._reset_bm25_index()
        results = rag_store.search_keyword_chunks("RAPID-OCR-2026", top_k=1)
        self.assertEqual(results[0]["source"], "scan.pdf")
        self.assertEqual(results[0]["parse_method"], "ocr")

        with patch("backend.rag_store.search_vector_chunks", return_value=[]):
            hybrid_results = rag_store.search_hybrid_chunks("RAPID-OCR-2026", top_k=1)
        self.assertEqual(hybrid_results[0]["source"], "scan.pdf")
        self.assertEqual(hybrid_results[0]["retrieval"], "hybrid")
        self.assertTrue(hybrid_results[0]["ocr_used"])

    def test_scanned_pdf_without_ocr_does_not_break_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_path = Path(tmpdir)
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with (docs_path / "scan.pdf").open("wb") as handle:
                writer.write(handle)

            with (
                patch.object(rag_store, "DOCS_PATH", docs_path),
                patch(
                    "backend.ocr_service.get_config",
                    return_value=SimpleNamespace(
                        enable_ocr=False,
                        ocr_engine="none",
                        ocr_min_text_chars=80,
                        ocr_render_dpi=200,
                        ocr_max_pages=20,
                    ),
                ),
                patch("backend.rag_store.get_embedding_model") as embedding_model,
            ):
                rag_store.rebuild_rag_index()

        self.assertIsNone(rag_store.index)
        embedding_model.assert_not_called()

    def test_unexpected_document_parser_failure_does_not_break_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_path = Path(tmpdir)
            (docs_path / "broken.pdf").write_bytes(b"%PDF-broken")
            with (
                patch.object(rag_store, "DOCS_PATH", docs_path),
                patch(
                    "backend.rag_store.extract_text_from_document",
                    side_effect=MemoryError("extreme parser failure"),
                ),
                patch("backend.rag_store.get_embedding_model") as embedding_model,
            ):
                rag_store.rebuild_rag_index()

        self.assertIsNone(rag_store.index)
        embedding_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
