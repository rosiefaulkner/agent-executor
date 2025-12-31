import os
import unittest
from unittest.mock import MagicMock, patch

# Set dummy environment variables before importing modules
os.environ["GEMINI_API_KEY"] = "dummy_gemini_key"
os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"

from langchain_core.tools import BaseTool

# Mock tool for type checking
class MockTavilySearch(BaseTool):
    name: str = "tavily_search_results_json"
    description: str = "A mock search tool"

    def _run(self, query: str) -> str:
        return "mocked search result"

    async def _arun(self, query: str) -> str:
        return "mocked search result"


# Patch external dependencies before importing our modules
with patch("langchain_google_genai.ChatGoogleGenerativeAI"), \
     patch("langchain_tavily.TavilySearch", new=MockTavilySearch), \
     patch("langgraph.graph.StateGraph.compile") as mock_compile:

    mock_app = MagicMock()
    mock_compile.return_value = mock_app

    import react
    import nodes
    import main

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    class ResourceExhausted(Exception):
        pass


class TestAgentUnit(unittest.TestCase):

    def test_react_triple_tool(self):
        """Test the triple tool functionality"""
        self.assertEqual(react.triple.invoke({"num": 5}), 15)
        self.assertEqual(react.triple.invoke({"num": 2.5}), 7.5)

    @patch("nodes.llm")
    def test_nodes_run_agent_reasoning(self, mock_llm):
        """Test the happy path for the agent reasoning node"""
        expected_response = AIMessage(content="Test response")
        mock_llm.invoke.return_value = expected_response
        state = {"messages": [HumanMessage(content="Test input")]}

        result = nodes.run_agent_reasoning(state)

        self.assertEqual(result, {"messages": [expected_response]})
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0]["role"], "system")
        self.assertEqual(call_args[1], state["messages"][0])

    @patch("nodes.llm")
    @patch("builtins.print")
    def test_run_agent_reasoning_quota_error(self, mock_print, mock_llm):
        """Test that ResourceExhausted errors are handled correctly"""
        error_message = "429 You exceeded your current quota"
        mock_llm.invoke.side_effect = ResourceExhausted(error_message)
        state = {"messages": [HumanMessage(content="Test input")]}

        with self.assertRaises(ResourceExhausted):
            nodes.run_agent_reasoning(state)

        printed_output = "\n".join(
            [call.args[0] for call in mock_print.call_args_list if call.args]
        )
        self.assertIn("---API QUOTA EXCEEDED---", printed_output)
        self.assertIn("You have exceeded your API quota", printed_output)
        self.assertIn(error_message, printed_output)

    @patch("nodes.llm")
    def test_run_agent_reasoning_other_error(self, mock_llm):
        """Test that other exceptions are propagated"""
        error_message = "Some other error"
        mock_llm.invoke.side_effect = Exception(error_message)
        state = {"messages": [HumanMessage(content="Test input")]}

        with self.assertRaises(Exception) as cm:
            nodes.run_agent_reasoning(state)
        self.assertEqual(str(cm.exception), error_message)

    def test_main_should_continue(self):
        """Test the routing logic of should_continue"""
        state_no_tool = {"messages": [AIMessage(content="...")]}
        self.assertEqual(main.should_continue(state_no_tool), main.END)

        tool_calls = [{"name": "t", "args": {}, "id": "1"}]
        state_with_tool = {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
        self.assertEqual(main.should_continue(state_with_tool), main.ACT)

    @patch("builtins.input")
    def test_run_interactive_session_approve(self, mock_input):
        """Test user approving a step in the interactive session"""
        mock_app = MagicMock()
        mock_input.return_value = "y"
        mock_snapshot = MagicMock()
        mock_snapshot.next = ("act",)
        mock_app.get_state.return_value = mock_snapshot
        mock_app.stream.side_effect = Exception("Loop broken")
        thread = {"configurable": {"thread_id": "1"}}

        with self.assertRaisesRegex(Exception, "Loop broken"):
            main.run_interactive_session(mock_app, thread)

        mock_input.assert_called_once()
        self.assertIn("approve", mock_input.call_args[0][0])
        mock_app.stream.assert_called_once_with(None, thread, stream_mode="values")

    @patch("builtins.input")
    def test_run_interactive_session_reject(self, mock_input):
        """Test user providing clarification in the interactive session"""
        mock_app = MagicMock()
        mock_input.return_value = "clarification"
        mock_snapshot = MagicMock()
        mock_snapshot.next = ("act",)
        mock_app.get_state.return_value = mock_snapshot
        mock_app.stream.side_effect = Exception("Loop broken")
        thread = {"configurable": {"thread_id": "1"}}

        with self.assertRaisesRegex(Exception, "Loop broken"):
            main.run_interactive_session(mock_app, thread)

        mock_app.update_state.assert_called_once()
        self.assertEqual(
            mock_app.update_state.call_args[0][1]["messages"][0].content,
            "clarification",
        )
        mock_app.stream.assert_called_once_with(None, thread, stream_mode="values")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_session_terminate(self, mock_print, mock_input):
        """Test user terminating the session with 'n'"""
        mock_app = MagicMock()
        mock_input.return_value = "n"
        mock_snapshot = MagicMock()
        mock_snapshot.next = ("act",)
        mock_app.get_state.return_value = mock_snapshot
        thread = {"configurable": {"thread_id": "1"}}

        main.run_interactive_session(mock_app, thread)

        printed_output = "\n".join(
            [call.args[0] for call in mock_print.call_args_list if call.args]
        )
        self.assertIn("---SESSION TERMINATED BY USER---", printed_output)
        mock_app.stream.assert_not_called()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_run_interactive_session_exit(self, mock_print, mock_input):
        """Test user exiting from clarification prompt"""
        mock_app = MagicMock()
        mock_input.return_value = "exit"
        mock_snapshot = MagicMock()
        mock_snapshot.next = None
        mock_app.get_state.return_value = mock_snapshot
        thread = {"configurable": {"thread_id": "1"}}

        main.run_interactive_session(mock_app, thread)

        mock_input.assert_called_once_with(
            "Please provide clarification (or type 'exit' to quit): "
        )
        printed_output = "\n".join(
            [call.args[0] for call in mock_print.call_args_list if call.args]
        )
        self.assertIn("---SESSION TERMINATED BY USER---", printed_output)
        mock_app.stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()