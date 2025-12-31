import os
import unittest
from unittest.mock import MagicMock, patch

# Set dummy environment variables before importing modules to pass validation checks
os.environ["GEMINI_API_KEY"] = "dummy_gemini_key"
os.environ["TAVILY_API_KEY"] = "dummy_tavily_key"

# Patch external dependencies to avoid network calls and side effects during import
# We mock ChatGoogleGenerativeAI and TavilySearch to prevent connection attempts
# We mock StateGraph.compile to prevent the graph drawing in main.py from creating a file
with patch("langchain_google_genai.ChatGoogleGenerativeAI"), \
     patch("langchain_tavily.TavilySearch"), \
     patch("langgraph.graph.StateGraph.compile") as mock_compile:
    
    # Setup the mock app returned by compile()
    mock_app = MagicMock()
    mock_compile.return_value = mock_app
    
    # Import the modules to be tested
    # These imports trigger the top-level code in the files
    import react
    import nodes
    import main

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

class TestAgentUnit(unittest.TestCase):

    def test_react_triple_tool(self):
        """Test the triple tool functionality in react.py"""
        # Test with integer input
        result = react.triple.invoke({"num": 5})
        self.assertEqual(result, 15)
        
        # Test with float input
        result = react.triple.invoke({"num": 2.5})
        self.assertEqual(result, 7.5)

    @patch("nodes.llm")
    def test_nodes_run_agent_reasoning(self, mock_llm):
        """Test the run_agent_reasoning node in nodes.py"""
        # Setup mock response from LLM
        expected_response = AIMessage(content="Test response")
        mock_llm.invoke.return_value = expected_response
        
        # Create input state
        input_message = HumanMessage(content="Test input")
        state = {"messages": [input_message]}
        
        # Run the function
        result = nodes.run_agent_reasoning(state)
        
        # Verify the result structure
        self.assertEqual(result, {"messages": [expected_response]})
        
        # Verify LLM interaction
        mock_llm.invoke.assert_called_once()
        
        # Check arguments passed to invoke
        # invoke is called with a list of messages
        call_args = mock_llm.invoke.call_args[0][0]
        self.assertEqual(len(call_args), 2)
        
        # First message should be the system message
        self.assertIsInstance(call_args[0], dict)
        self.assertEqual(call_args[0]["role"], "system")
        self.assertIn("helpful assistant", call_args[0]["content"])
        
        # Second message should be our input message
        self.assertEqual(call_args[1], input_message)

    def test_main_should_continue(self):
        """Test the should_continue logic in main.py"""
        # Case 1: No tool calls -> END
        message_no_tool = AIMessage(content="Just chatting")
        state_no_tool = {"messages": [HumanMessage(content="hi"), message_no_tool]}
        
        next_node = main.should_continue(state_no_tool)
        self.assertEqual(next_node, main.END)
        
        # Case 2: Tool calls present -> ACT
        message_with_tool = AIMessage(
            content="", 
            tool_calls=[{"name": "triple", "args": {"num": 5}, "id": "call_1"}]
        )
        state_with_tool = {"messages": [HumanMessage(content="calc"), message_with_tool]}
        
        next_node = main.should_continue(state_with_tool)
        self.assertEqual(next_node, main.ACT)

if __name__ == "__main__":
    unittest.main()