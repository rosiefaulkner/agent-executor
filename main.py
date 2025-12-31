from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    """
    If the last message is a tool call because agent decided that
    a tool execution was required, then we want to execute the tool node.
    Else, end because the LLM was able to answer the question withput a tool.

    Args:
        state (MessagesState): message state

    Returns:
        str: next node to execute in graph
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct Langgraph with Function Calling")
    print("---INVOKING AGENT W HUMAN MESSAGE---")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the weather like in Tokyo? List THE CURRENT WEATHER INFO FOR TOKYO and then triple THE TEMPERATURE"
                )
            ]
        }
    )
    print("---DISPLAYING LAST MESSAGE---")
    print(res["messages"][LAST].content)
