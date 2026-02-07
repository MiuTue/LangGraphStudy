from dotenv import load_dotenv

from typing import List, TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain


load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
REFLECT="reflect"
GENERATE="generate"

def generation_node(state: MessageGraph):
    return {"messages" : [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages" : [HumanMessage(content = res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]):
    if len(state["messages"]) > 8:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)
app = builder.compile()
app.get_graph().draw_mermaid_png(output_file_path="reflection_agent_graph.png")

if __name__ == "__main__":
    print("Hello Reflection agent")
    inputs = HumanMessage(content="""Make this tweet better: "MiuTue is the best coder at DTU" """)
    response = app.invoke({"messages": [inputs]})
    print(response["messages"][-1].content)