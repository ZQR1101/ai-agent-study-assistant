import unittest

from pydantic import ValidationError

from backend.agent_core import _extract_json_object, _fallback_agent_plan
from backend.config import DEFAULT_MODEL, get_config, normalize_model
from backend.history_utils import HISTORY_LIMIT, format_history, normalize_history
from backend import rag_store
from backend.rag_store import expand_query, get_rag_index_status, is_valid_chunk
from backend.schemas import AgentPlan, AgentPlanStep, ChatRequest, FlashcardItem
from backend.tools import TOOL_REGISTRY, ToolSpec


class HistoryUtilsTests(unittest.TestCase):
    def test_empty_history_returns_empty_values(self):
        self.assertEqual(normalize_history([]), [])
        self.assertEqual(normalize_history(None), [])
        self.assertEqual(format_history([]), "")

    def test_history_is_limited_to_recent_items(self):
        history = [
            {"role": "user", "content": f"message {index}"}
            for index in range(HISTORY_LIMIT + 3)
        ]

        normalized = normalize_history(history)

        self.assertEqual(len(normalized), HISTORY_LIMIT)
        self.assertEqual(normalized[0]["content"], "message 3")
        self.assertEqual(normalized[-1]["content"], f"message {HISTORY_LIMIT + 2}")

    def test_invalid_roles_and_empty_content_are_filtered(self):
        normalized = normalize_history(
            [
                {"role": "system", "content": "do not include"},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "   "},
                {"role": "user", "content": "include me"},
            ]
        )

        self.assertEqual(normalized, [{"role": "user", "content": "include me"}])

    def test_formatted_history_contains_user_and_assistant_content(self):
        formatted = format_history(
            normalize_history(
                [
                    {"role": "user", "content": "What is RAG?"},
                    {"role": "assistant", "content": "RAG retrieves context first."},
                ]
            )
        )

        self.assertIn("What is RAG?", formatted)
        self.assertIn("RAG retrieves context first.", formatted)


class RagStoreTests(unittest.TestCase):
    def test_invalid_chunks_are_rejected(self):
        for text in ["|", "---", "###", "too short"]:
            with self.subTest(text=text):
                self.assertFalse(is_valid_chunk(text))

    def test_valid_chinese_chunk_is_accepted(self):
        text = (
            "检索增强生成会先从知识库中查找相关内容，"
            "再让语言模型基于这些上下文生成更可靠的回答。"
        )

        self.assertTrue(is_valid_chunk(text))

    def test_valid_english_chunk_is_accepted(self):
        text = (
            "Retrieval augmented generation first retrieves relevant context "
            "from a knowledge base, then asks the language model to answer."
        )

        self.assertTrue(is_valid_chunk(text))

    def test_expand_query_adds_known_expansions(self):
        self.assertIn("代理式RAG", expand_query("agentic rag"))
        self.assertIn("提示工程", expand_query("prompt engineering"))
        self.assertEqual(expand_query("ordinary question"), "ordinary question")

    def test_pure_rag_helpers_do_not_load_embedding_model(self):
        is_valid_chunk("Retrieval augmented generation uses retrieved context before answering.")
        expand_query("agentic rag")

        self.assertIsNone(rag_store.embedding_model)


class SchemaTests(unittest.TestCase):
    def test_chat_request_can_be_created(self):
        request = ChatRequest(message="What is RAG?", mode="explain", top_k=3)

        self.assertEqual(request.message, "What is RAG?")
        self.assertEqual(request.mode, "explain")

    def test_chat_request_rejects_invalid_values(self):
        invalid_payloads = [
            {"message": ""},
            {"message": "hello", "temperature": -0.1},
            {"message": "hello", "temperature": 2.1},
            {"message": "hello", "top_k": 0},
            {"message": "hello", "top_k": 11},
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    ChatRequest(**payload)

    def test_flashcard_difficulty_is_restricted(self):
        FlashcardItem(front="Q", back="A", difficulty="easy")
        FlashcardItem(front="Q", back="A", difficulty="medium")
        FlashcardItem(front="Q", back="A", difficulty="hard")

        with self.assertRaises(ValidationError):
            FlashcardItem(front="Q", back="A", difficulty="expert")

    def test_agent_plan_step_tool_is_restricted(self):
        AgentPlanStep(tool="chat", input="hello")

        with self.assertRaises(ValidationError):
            AgentPlanStep(tool="unknown", input="hello")


class AgentCoreTests(unittest.TestCase):
    def test_extract_json_object_from_plain_json(self):
        self.assertEqual(_extract_json_object('{"goal": "test"}'), {"goal": "test"})

    def test_extract_json_object_from_surrounding_text(self):
        text = 'Planner result follows: {"goal": "test", "steps": []} done.'

        self.assertEqual(
            _extract_json_object(text),
            {"goal": "test", "steps": []},
        )

    def test_extract_json_object_returns_none_for_invalid_json(self):
        self.assertIsNone(_extract_json_object("not json at all"))
        self.assertIsNone(_extract_json_object('before {"goal": "missing end" after'))

    def test_fallback_agent_plan_returns_valid_agent_plan(self):
        plan = _fallback_agent_plan("hello")

        if hasattr(AgentPlan, "model_validate"):
            validated = AgentPlan.model_validate(plan)
        else:
            validated = AgentPlan.parse_obj(plan)

        self.assertTrue(validated.fallback)
        self.assertEqual(validated.steps[0].tool, "chat")

    def test_fallback_agent_plan_routes_learning_requests(self):
        plan = _fallback_agent_plan("请解释 RAG，并出 3 道题")
        tools = {step["tool"] for step in plan["steps"]}

        self.assertTrue(tools & {"explain", "quiz", "rag"})

    def test_fallback_agent_plan_routes_unknown_input_to_chat(self):
        plan = _fallback_agent_plan("hello there")

        self.assertEqual(plan["steps"][0]["tool"], "chat")


class ToolsTests(unittest.TestCase):
    def test_tool_registry_contains_expected_tools(self):
        expected_tools = {"chat", "rag", "explain", "summarize", "quiz", "flashcard"}

        self.assertEqual(set(TOOL_REGISTRY), expected_tools)

    def test_registered_tools_have_required_structure(self):
        for name, tool in TOOL_REGISTRY.items():
            with self.subTest(tool=name):
                self.assertIsInstance(tool, ToolSpec)
                self.assertEqual(tool.name, name)
                self.assertTrue(tool.description.strip())
                self.assertTrue(callable(tool.run))


class ConfigTests(unittest.TestCase):
    def test_unknown_model_falls_back_to_default(self):
        self.assertEqual(normalize_model("unknown-model"), DEFAULT_MODEL)

    def test_config_paths_are_project_local(self):
        config = get_config()

        self.assertEqual(config.docs_path.name, "docs")
        self.assertEqual(config.rag_index_dir.name, "rag_index")


class RagIndexStatusTests(unittest.TestCase):
    def test_rag_index_status_does_not_load_index(self):
        status = get_rag_index_status()

        self.assertIn("ready", status)
        self.assertIn("message", status)
        self.assertIsNone(rag_store.embedding_model)


if __name__ == "__main__":
    unittest.main()
