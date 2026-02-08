from dotenv import load_dotenv

from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, START, MessagesState

from chains import revisor, first_responder
from tool_excutor import execute_tools
load_dotenv()

MAX_ITERATIONS = 2
def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """Revise the response."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal["execute_tools", END]: # type: ignore
    """Determine whether to execute tools or end the process."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    number_iterations = count_tool_visits
    if number_iterations < MAX_ITERATIONS:
        return "execute_tools"
    else:
        return END

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("revise", revise_node)
builder.add_node("execute_tools", execute_tools)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, {"execute_tools": "execute_tools", END: END})
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflextion_agent_graph.png")

if __name__ == "__main__":
    print("Hello Reflection agent")
    res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
    )
    # Extract the final answer from the last message with tool calls
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
  