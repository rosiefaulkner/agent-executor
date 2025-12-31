"""
Checks last message between agent and human and decides if a tool
should be called and if so, identifies which tool to call.
"""

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

from react import llm

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
You should always use the tools at your disposal to find the most up-to-date information.
When asked about a topic that may have recent developments or requires current information (like weather, news, or current events), you must use the search tool.
If a tool call fails, the error will be returned to you. You should analyze the error and try to call the tool again with corrected parameters.
Do not rely on your internal knowledge for such questions.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node
    """
    try:
        response = llm.invoke(
            [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]],
            config={"request_options": {"timeout": 60}},
        )
        return {"messages": [response]}
    except ResourceExhausted as e:
        print("\n---API QUOTA EXCEEDED---")
        print(
            "You have exceeded your API quota. Please check your plan and billing details."
        )
        print(
            "You can also try setting the GEMINI_MODEL environment variable to a different model."
        )
        print(f"Original error: {e}")
        # Re-raise the exception to stop the application
        raise
