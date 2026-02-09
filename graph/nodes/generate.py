from typing import Any, Dict

from graph.state import GraphState
from graph.chains.generation import generation_chain


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE NODE---")
    context = state["documents"]
    question = state["question"]

    answer = generation_chain.invoke({"context": context, "question": question})
    return {"documents": context, "question": question, "generation": answer}