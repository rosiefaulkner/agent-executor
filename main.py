import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from nodes import run_agent_reasoning
from react import tavily_tool, triple

load_dotenv()

AGENT_REASON = "agent_reason"
LAST = -1


def should_continue(state: MessagesState) -> list[str]:
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return [END]
    else:
        return [tool_call["name"] for tool_call in last_message.tool_calls]


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node("tavily_search", ToolNode([tavily_tool]))
flow.add_node("triple", ToolNode([triple]))
flow.set_entry_point(AGENT_REASON)
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {"tavily_search": "tavily_search", "triple": "triple", END: END},
)
flow.add_edge("tavily_search", AGENT_REASON)
flow.add_edge("triple", AGENT_REASON)

tool_names = [t.name for t in [tavily_tool, triple]]


async def get_app():
    """
    Creates and returns the compiled LangGraph app with an async checkpointer.
    """
    memory = await AsyncSqliteSaver.acreate(
        db=":memory:"
    )  # Using in-memory for simplicity
    app = flow.compile(checkpointer=memory, interrupt_before=tool_names)
    return app


async def run_interactive_session(app, thread):
    """
    Runs an interactive asynchronous session with the agent.
    """
    while True:
        snapshot = await app.aget_state(thread)
        if snapshot.next:
            user_input = await asyncio.to_thread(
                input,
                f"Do you approve the next step: '{snapshot.next[0]}'? Type 'y' to approve, 'n' to terminate, or provide clarification: ",
            )
            if user_input.strip().lower() == "y":
                async for event in app.astream(None, thread, stream_mode="values"):
                    event["messages"][-1].pretty_print()
            elif user_input.strip().lower() == "n":
                print("---SESSION TERMINATED BY USER---")
                return
            else:
                async for event in app.astream(
                    {"messages": [HumanMessage(content=user_input)]},
                    thread,
                    stream_mode="values",
                ):
                    event["messages"][-1].pretty_print()
        else:
            clarification = await asyncio.to_thread(
                input, "Please provide clarification (or type 'exit' to quit): "
            )
            if clarification.strip().lower() == "exit":
                print("---SESSION TERMINATED BY USER---")
                return
            async for event in app.astream(
                {"messages": [HumanMessage(content=clarification)]},
                thread,
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()


async def main_async():
    """
    Main async function to run the agent.
    """
    print("---INITIALIZING APP---")
    app = await get_app()
    thread = {"configurable": {"thread_id": "1"}}

    print("---INVOKING AGENT W HUMAN MESSAGE---")
    initial_input = {"messages": [HumanMessage(content="Hi there")]}
    async for event in app.astream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    await run_interactive_session(app, thread)


if __name__ == "__main__":
    asyncio.run(main_async())
