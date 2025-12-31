import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["GEMINI_API_KEY"] = "dummy_gemini_key"
os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"

from langchain_core.tools import BaseTool

class MockTavilySearch(BaseTool):
    name: str = "tavily_search"
    description: str = "A mock search tool"
    def _run(self, query: str) -> str: return "mocked search result"
    async def _arun(self, query: str) -> str:
        await asyncio.sleep(0.1)
        return "mocked search result"

# Patch dependencies to avoid side effects
with patch("langchain_google_genai.ChatGoogleGenerativeAI"), \
     patch("langchain_tavily.TavilySearch", new=MockTavilySearch), \
     patch("main.AsyncSqliteSaver", new=None): # Ensure we use MemorySaver
    import nodes
    import main
    import react

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    class ResourceExhausted(Exception): pass

class TestAgentExecution(unittest.IsolatedAsyncioTestCase):
    async def test_react_triple_tool_async(self):
        """Tests the async triple tool"""
        result = await react.triple.ainvoke({"num": 10})
        self.assertEqual(result, 30)

    @patch("nodes.llm")
    async def test_nodes_run_agent_reasoning_quota_error(self, mock_llm):
        """Tests quota error handling in the agent reasoning node"""
        error_message = "429 You exceeded your current quota"
        mock_llm.invoke.side_effect = ResourceExhausted(error_message)
        state = {"messages": [HumanMessage(content="Test input")]}

        with self.assertRaises(ResourceExhausted):
            nodes.run_agent_reasoning(state)

if __name__ == "__main__":
    unittest.main()
