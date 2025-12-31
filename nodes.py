"""
Checks last message between agent and human and decides if a tool
should be called and if so, identifies which tool to call.
"""

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
You should always use the tools at your disposal to find the most up-to-date information.
When asked about a topic that may have recent developments or requires current information (like weather, news, or current events), you must use the search tool.
If a tool call fails, you should analyze the error and try to call the tool again with corrected parameters. If you are still unable to find the answer after retrying, inform the user that you were unable to find the information.
Do not rely on your internal knowledge for such questions.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


def handle_tool_error(state: MessagesState) -> MessagesState:
    """
    If the last message is a tool call error, add a message to the state
    to encourage the agent to retry.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        if "Error:" in str(last_message.content):
            return {
                "messages": [
                    SystemMessage(
                        content="The previous tool call failed. Please analyze the error and try again. If the error persists, you may need to try a different approach."
                    )
                ]
            }
    return {}


tool_node = ToolNode(tools)
