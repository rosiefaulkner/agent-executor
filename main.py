from typing import TypedDict

from dotenv import load_dotenv
from langchain_core import messages
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

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
    if not getattr(state["messages"][LAST], "tool_calls", None):
        return END
    return ACT


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASON)

memory = MemorySaver()

app = flow.compile(checkpointer=memory, interrupt_before=[ACT])

graph = app.get_graph()

if "__interrupt__" in graph.nodes:
    graph.nodes["human in the loop"] = graph.nodes.pop("__interrupt__")
    new_edges = set()
    for source, target in graph.edges:
        if source == "__interrupt__":
            source = "human in the loop"
        if target == "__interrupt__":
            target = "human in the loop"
        new_edges.add((source, target))
    graph.edges = new_edges

    new_conditional_edges = []
    for source, cond, targets in graph.conditional_edges:
        new_targets = {}
        for k, v in targets.items():
            if v == "__interrupt__":
                v = "human in the loop"
            new_targets[k] = v
        new_conditional_edges.append((source, cond, new_targets))
    graph.conditional_edges = new_conditional_edges

graph.draw_mermaid_png(output_file_path="flow.png")


def run_interactive_session(app, thread):
    """
    Runs an interactive session with the agent.
    """
    while True:
        snapshot = app.get_state(thread)
        if snapshot.next:
            user_input = input(
                f"Do you approve the next step: '{snapshot.next[0]}'? Type 'y' to approve, or provide clarification: "
            )
            if user_input.strip().lower() == "y":
                for event in app.stream(None, thread, stream_mode="values"):
                    event["messages"][-1].pretty_print()
            else:
                app.update_state(
                    thread,
                    {"messages": [HumanMessage(content=user_input)]},
                    as_node=ACT,
                )
                for event in app.stream(None, thread, stream_mode="values"):
                    event["messages"][-1].pretty_print()
        else:
            clarification = input(
                "How can I help? Explain what you'd like me to do next:"
            )
            for event in app.stream(
                {"messages": [HumanMessage(content=clarification)]},
                thread,
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()


if __name__ == "__main__":
    print("---INITIALIZING GRAPH---")
    thread = {
        "configurable": {"thread_id": "1"}
    }  # TODO: update the thread_id to dynamic population rather than hardcoding

    print("---INVOKING AGENT W HUMAN MESSAGE---")
    initial_input = {"messages": [HumanMessage(content="Hi there")]}
    for event in app.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    run_interactive_session(app, thread)
