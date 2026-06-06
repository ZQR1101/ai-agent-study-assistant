import builtins
import unittest
from unittest.mock import patch


class LangGraphDemoTests(unittest.TestCase):
    def test_module_can_be_imported_without_running_graph(self):
        import backend.langgraph_demo as demo

        self.assertTrue(hasattr(demo, "LangGraphDemoState"))
        self.assertTrue(callable(demo.run_langgraph_demo))

    def test_missing_langgraph_error_is_clear(self):
        import backend.langgraph_demo as demo

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("langgraph"):
                raise ImportError("blocked langgraph import")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "LangGraph is not installed"):
                demo.build_demo_graph()

    def test_build_graph_when_langgraph_is_installed(self):
        import backend.langgraph_demo as demo

        try:
            graph = demo.build_demo_graph()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        self.assertTrue(hasattr(graph, "invoke"))

    def _run_with_mocks(self, message):
        import backend.langgraph_demo as demo

        try:
            demo.build_demo_graph()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        with (
            patch.object(
                demo,
                "get_rag_context",
                return_value={
                    "found": True,
                    "context": "mock rag context",
                    "sources": [{"source": "mock.md", "score": 0.9}],
                },
            ) as rag_mock,
            patch.object(demo, "explain", return_value="mock explanation") as explain_mock,
            patch.object(demo, "generate_questions", return_value="mock quiz") as quiz_mock,
        ):
            result = demo.run_langgraph_demo(message)

        return result, rag_mock, explain_mock, quiz_mock

    def _trace_path(self, result):
        path = []

        for item in result["trace"]:
            node = item.split(":", 1)[0]
            if node in {"planner", "rag", "explain", "flashcard", "quiz"}:
                if not path or path[-1] != node:
                    path.append(node)

        return path

    def test_plain_explain_routes_to_explain_then_end(self):
        result, rag_mock, explain_mock, quiz_mock = self._run_with_mocks("什么是 RAG")

        self.assertEqual(self._trace_path(result), ["planner", "explain"])
        self.assertIn("mock explanation", result["answer"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["explain"])
        rag_mock.assert_not_called()
        explain_mock.assert_called_once()
        quiz_mock.assert_not_called()

    def test_knowledge_base_explain_routes_through_rag(self):
        result, rag_mock, explain_mock, quiz_mock = self._run_with_mocks("根据知识库解释 agentic rag")

        self.assertEqual(self._trace_path(result), ["planner", "rag", "explain"])
        self.assertEqual(result["sources"], [{"source": "mock.md", "score": 0.9}])
        self.assertEqual([step["tool"] for step in result["plan"]], ["rag", "explain"])
        rag_mock.assert_called_once()
        explain_mock.assert_called_once()
        quiz_mock.assert_not_called()

    def test_explain_and_quiz_routes_to_quiz(self):
        result, rag_mock, explain_mock, quiz_mock = self._run_with_mocks("请解释 RAG，并出 3 道练习题")

        self.assertEqual(self._trace_path(result), ["planner", "explain", "quiz"])
        self.assertIn("mock quiz", result["answer"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["explain", "quiz"])
        rag_mock.assert_not_called()
        explain_mock.assert_called_once()
        quiz_mock.assert_called_once()

    def test_knowledge_base_flashcard_routes_to_flashcard(self):
        result, rag_mock, explain_mock, quiz_mock = self._run_with_mocks("根据知识库生成 agentic rag 记忆卡片")

        self.assertEqual(self._trace_path(result), ["planner", "rag", "explain", "flashcard"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["rag", "explain", "flashcard"])
        self.assertEqual(len(result["flashcards"]), 1)
        rag_mock.assert_called_once()
        explain_mock.assert_called_once()
        quiz_mock.assert_not_called()

    def test_knowledge_base_flashcard_and_quiz_routes_through_all_nodes(self):
        message = "根据知识库解释 agentic rag，生成记忆卡片，并出 3 道题"
        result, rag_mock, explain_mock, quiz_mock = self._run_with_mocks(message)

        self.assertEqual(self._trace_path(result), ["planner", "rag", "explain", "flashcard", "quiz"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["rag", "explain", "flashcard", "quiz"])
        self.assertIn("mock explanation", result["answer"])
        self.assertIn("mock quiz", result["answer"])
        self.assertEqual(len(result["flashcards"]), 1)
        rag_mock.assert_called_once()
        explain_mock.assert_called_once()
        quiz_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
