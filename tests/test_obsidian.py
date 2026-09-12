import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from backend import rag_store, tool_actions
from backend.config import read_obsidian_vault_path


class ObsidianVaultPathTests(unittest.TestCase):
    def test_empty_env_does_not_resolve_to_cwd(self):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": ""}, clear=False):
            self.assertIsNone(read_obsidian_vault_path())

    def test_missing_env_is_none(self):
        env = {key: value for key, value in os.environ.items() if key != "OBSIDIAN_VAULT_PATH"}
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(read_obsidian_vault_path())

    def test_configured_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)
            with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(vault)}):
                self.assertEqual(read_obsidian_vault_path(), vault.resolve())


class ObsidianVaultLoadTests(unittest.TestCase):
    def test_parse_vault_note_strips_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            note = Path(tmpdir) / "welcome.md"
            note.write_text(
                "---\ntitle: 欢迎\nsource: obsidian\n---\n正文从这里开始。\n",
                encoding="utf-8",
            )
            self.assertEqual(rag_store._parse_vault_note(note), "正文从这里开始。")

    def test_load_vault_skips_system_dirs_and_prefixes_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)
            (vault / "欢迎.md").write_text("这是库里的欢迎笔记内容。", encoding="utf-8")
            nested = vault / "CS"
            nested.mkdir()
            (nested / "os.md").write_text("操作系统笔记正文。", encoding="utf-8")
            hidden = vault / ".obsidian"
            hidden.mkdir()
            (hidden / "workspace.md").write_text("should skip", encoding="utf-8")
            trash = vault / ".trash"
            trash.mkdir()
            (trash / "gone.md").write_text("should skip", encoding="utf-8")

            with patch.object(rag_store, "VAULT_PATH", vault):
                documents = rag_store._load_vault_documents()

        sources = [document["source"] for document in documents]
        self.assertEqual(sources, ["obsidian:CS/os.md", "obsidian:欢迎.md"])
        self.assertTrue(all(document["parse_method"] == "vault" for document in documents))

    def test_load_documents_merges_docs_and_vault(self):
        parse_result = {
            "text": "docs note",
            "method": "text",
            "ocr_used": False,
            "need_ocr": False,
            "text_char_count": 9,
            "warnings": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_path = root / "docs"
            vault = root / "vault"
            docs_path.mkdir()
            vault.mkdir()
            (docs_path / "guide.md").write_text("guide", encoding="utf-8")
            (vault / "欢迎.md").write_text("vault welcome body", encoding="utf-8")

            with (
                patch.object(rag_store, "DOCS_PATH", docs_path),
                patch.object(rag_store, "VAULT_PATH", vault),
                patch(
                    "backend.rag_store.extract_text_from_document",
                    return_value=parse_result,
                ),
            ):
                documents = rag_store.load_documents()

        self.assertEqual(
            [document["source"] for document in documents],
            ["guide.md", "obsidian:欢迎.md"],
        )

    def test_load_documents_includes_vault_when_docs_dir_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_path = root / "docs"
            vault = root / "vault"
            vault.mkdir()
            (vault / "欢迎.md").write_text("only vault", encoding="utf-8")

            with (
                patch.object(rag_store, "DOCS_PATH", docs_path),
                patch.object(rag_store, "VAULT_PATH", vault),
            ):
                documents = rag_store.load_documents()

            self.assertEqual(
                [document["source"] for document in documents],
                ["obsidian:欢迎.md"],
            )
            self.assertTrue(docs_path.is_dir())


class ObsidianSaveNoteTests(unittest.TestCase):
    def test_sanitize_filename_strips_unsafe_characters(self):
        self.assertEqual(tool_actions._sanitize_filename('a/b:c*d?.md'), "a_b_c_d_.md")
        self.assertEqual(tool_actions._sanitize_filename("   "), "untitled")

    def test_save_note_writes_json_and_vault_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            saved_dir = root / "saved"
            vault = root / "vault"
            vault.mkdir()

            with (
                patch.object(tool_actions, "SAVED_ITEMS_DIR", saved_dir),
                patch.object(tool_actions, "OBSIDIAN_VAULT_PATH", vault),
                patch("backend.tool_actions.datetime") as mock_datetime,
            ):
                mock_datetime.now.return_value = datetime(2026, 8, 27, 14, 30, 52)
                result = tool_actions.save_note(
                    title="操作系统",
                    content="进程和线程。",
                    tags=["os"],
                )

            note_path = vault / "AI Study Assistant" / "操作系统_20260827_143052.md"
            self.assertTrue(note_path.exists())
            written = note_path.read_text(encoding="utf-8")
            self.assertIn("source: rulebook", written)
            self.assertIn("进程和线程。", written)
            self.assertIn("操作系统", written)
            self.assertTrue((saved_dir / "notes.json").exists())
            self.assertIn("Obsidian:", result["answer"])
            self.assertEqual(result["saved_item"]["title"], "操作系统")

    def test_save_note_does_not_overwrite_existing_vault_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)
            notes_dir = vault / "AI Study Assistant"
            notes_dir.mkdir()
            existing = notes_dir / "note_20260827_143052.md"
            existing.write_text("keep me", encoding="utf-8")

            with (
                patch.object(tool_actions, "OBSIDIAN_VAULT_PATH", vault),
                patch("backend.tool_actions.datetime") as mock_datetime,
            ):
                mock_datetime.now.return_value = datetime(2026, 8, 27, 14, 30, 52)
                path = tool_actions._vault_note_path("note")

            self.assertEqual(path.name, "note_20260827_143052_2.md")
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep me")

    def test_save_note_without_vault_still_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_dir = Path(tmpdir)
            with (
                patch.object(tool_actions, "SAVED_ITEMS_DIR", saved_dir),
                patch.object(tool_actions, "OBSIDIAN_VAULT_PATH", None),
            ):
                result = tool_actions.save_note(title="only json", content="body")

            self.assertNotIn("Obsidian:", result["answer"])
            self.assertTrue((saved_dir / "notes.json").exists())
