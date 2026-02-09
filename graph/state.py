from typing import List, TypedDict


class GraphState(TypedDict):
    """State of the graph during execution.

    Attributes:
        question: question
        generation: LLm generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
