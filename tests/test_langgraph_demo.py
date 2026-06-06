import builtins
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch


@dataclass
class FakeTool:
    name: str
    description: str
    calls: list = field(default_factory=list)

    def run(
        self,
        step_input: str,
        original_input: str = "",
        custom_llm=None,
        top_k: int = 3,
        shared_context: dict | None = None,
    ) -> dict:
        shared_context = shared_context or {}
        self.calls.append({
            "step_input": step_input,
            "original_input": original_input,
            "top_k": top_k,
            "shared_context": shared_context,
        })

        if self.name == "rag":
            return {
                "answer": "rag answer",
                "sources": [{"source": "rag.md", "score": 0.9}],
                "trace": ["registry rag trace"],
                "context": "mock rag context",
                "flashcards": [],
                "used_context": False,
                "context_sources": [],
            }

        if self.name == "explain":
            return {
                "answer": "explain answer",
                "sources": [
                    {"source": "rag.md", "score": 0.9},
                    {"source": "explain.md", "score": 0.8},
                ],
                "trace": ["registry explain trace"],
                "context": "",
                "flashcards": [],
                "used_context": bool(shared_context.get("rag_context") or shared_context.get("last_output")),
                "context_sources": ["rag_context"] if shared_context.get("rag_context") else [],
            }

        if self.name == "flashcard":
            return {
                "answer": "flashcard answer",
                "sources": [],
                "trace": ["registry flashcard trace"],
                "context": "",
                "flashcards": [
                    {
                        "front": "What is Agentic RAG?",
                        "back": "Agentic RAG uses agent behavior around retrieval.",
                        "tags": ["rag"],
                        "difficulty": "medium",
                    }
                ],
                "used_context": bool(shared_context.get("rag_context") or shared_context.get("last_output")),
                "context_sources": ["rag_context", "previous_step_output"],
            }

        if self.name == "quiz":
            return {
                "answer": "quiz answer",
                "sources": [],
                "trace": ["registry quiz trace"],
                "context": "",
                "flashcards": [],
                "used_context": bool(shared_context.get("rag_context") or shared_context.get("last_output")),
                "context_sources": ["previous_step_output"],
            }

        return {
            "answer": f"{self.name} answer",
            "sources": [],
            "trace": [],
            "context": "",
            "flashcards": [],
            "used_context": False,
            "context_sources": [],
        }


def make_fake_registry():
    return {
        name: FakeTool(name=name, description=f"{name} description")
        for name in ["chat", "rag", "explain", "summarize", "quiz", "flashcard"]
    }


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

    def _run_with_fake_registry(self, message):
        import backend.langgraph_demo as demo

        try:
            demo.build_demo_graph()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        fake_registry = make_fake_registry()

        with patch.object(demo, "TOOL_REGISTRY", fake_registry):
            result = demo.run_langgraph_demo(message)

        return result, fake_registry

    def _trace_path(self, result):
        path = []

        for item in result["trace"]:
            node = item.split(":", 1)[0]
            if node in {"planner", "rag", "explain", "flashcard", "quiz"}:
                if not path or path[-1] != node:
                    path.append(node)

        return path

    def test_plain_explain_routes_to_explain_then_end(self):
        result, registry = self._run_with_fake_registry("what is RAG")

        self.assertEqual(self._trace_path(result), ["planner", "explain"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["explain"])
        self.assertIn("explain answer", result["answer"])
        self.assertEqual(len(registry["explain"].calls), 1)
        self.assertEqual(len(registry["rag"].calls), 0)
        self.assertEqual(len(registry["quiz"].calls), 0)

    def test_knowledge_base_explain_routes_through_rag(self):
        result, registry = self._run_with_fake_registry("knowledge base explain agentic rag")

        self.assertEqual(self._trace_path(result), ["planner", "rag", "explain"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["rag", "explain"])
        self.assertEqual([source["source"] for source in result["sources"]], ["rag.md", "explain.md"])
        self.assertEqual(len(registry["rag"].calls), 1)
        self.assertEqual(len(registry["explain"].calls), 1)
        self.assertEqual(registry["explain"].calls[0]["shared_context"]["rag_context"], "mock rag context")

    def test_explain_and_quiz_routes_to_quiz(self):
        result, registry = self._run_with_fake_registry("explain RAG and quiz me")

        self.assertEqual(self._trace_path(result), ["planner", "explain", "quiz"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["explain", "quiz"])
        self.assertIn("quiz answer", result["answer"])
        self.assertEqual(len(registry["rag"].calls), 0)
        self.assertEqual(len(registry["quiz"].calls), 1)
        self.assertEqual(registry["quiz"].calls[0]["shared_context"]["last_output"], "explain answer")

    def test_knowledge_base_flashcard_routes_to_flashcard(self):
        result, registry = self._run_with_fake_registry("knowledge base generate agentic rag flashcard")

        self.assertEqual(self._trace_path(result), ["planner", "rag", "explain", "flashcard"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["rag", "explain", "flashcard"])
        self.assertEqual(len(result["flashcards"]), 1)
        self.assertEqual(result["flashcards"][0]["front"], "What is Agentic RAG?")
        self.assertEqual(len(registry["flashcard"].calls), 1)
        self.assertEqual(registry["flashcard"].calls[0]["shared_context"]["rag_context"], "mock rag context")

    def test_knowledge_base_flashcard_and_quiz_routes_through_all_nodes(self):
        message = "knowledge base explain agentic rag, generate flashcard, and quiz me"
        result, registry = self._run_with_fake_registry(message)

        self.assertEqual(self._trace_path(result), ["planner", "rag", "explain", "flashcard", "quiz"])
        self.assertEqual([step["tool"] for step in result["plan"]], ["rag", "explain", "flashcard", "quiz"])
        self.assertIn("explain answer", result["answer"])
        self.assertIn("flashcard answer", result["answer"])
        self.assertIn("quiz answer", result["answer"])
        self.assertEqual(len(result["flashcards"]), 1)
        self.assertEqual(len(registry["quiz"].calls), 1)
        self.assertEqual(registry["quiz"].calls[0]["shared_context"]["last_output"], "flashcard answer")

    def test_registry_tool_error_is_friendly(self):
        import backend.langgraph_demo as demo

        result = demo.run_registry_tool("missing", "hello", {"message": "hello"})

        self.assertFalse(result["tool_success"])
        self.assertIn("Unknown tool", result["error"])

    def test_trace_contains_registry_tool_metadata(self):
        result, _ = self._run_with_fake_registry("knowledge base explain agentic rag")

        self.assertTrue(any("rag: call tool=rag" in item for item in result["trace"]))
        self.assertTrue(any("rag: tool description=rag description" in item for item in result["trace"]))
        self.assertTrue(any("rag: tool success=yes" in item for item in result["trace"]))
        self.assertTrue(any("explain: used context=yes" in item for item in result["trace"]))


if __name__ == "__main__":
    unittest.main()
