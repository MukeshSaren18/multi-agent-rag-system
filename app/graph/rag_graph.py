import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from app.agents.query_decomposer import QueryDecomposerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.synthesizer import SynthesizerAgent
from app.models.schemas import SubQuery, RetrievedDocument

logger = logging.getLogger(__name__)


# ── Graph State ───────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    collection: Optional[str]
    top_k: Optional[int]
    sub_queries: list[SubQuery]
    documents: list[RetrievedDocument]
    answer: str
    error: Optional[str]


# ── Node functions ────────────────────────────────────────────────────────────

decomposer = QueryDecomposerAgent()
retriever  = RetrieverAgent()
synthesizer = SynthesizerAgent()


async def decompose_node(state: RAGState) -> RAGState:
    try:
        sub_queries = await decomposer.decompose(state["query"])
        return {**state, "sub_queries": sub_queries}
    except Exception as e:
        logger.error("Decompose node error: %s", e)
        return {**state, "error": str(e), "sub_queries": []}


async def retrieve_node(state: RAGState) -> RAGState:
    if state.get("error") or not state["sub_queries"]:
        return {**state, "documents": []}
    try:
        docs = await retriever.retrieve(
            sub_queries=state["sub_queries"],
            top_k=state.get("top_k"),
            collection=state.get("collection"),
        )
        return {**state, "documents": docs}
    except Exception as e:
        logger.error("Retrieve node error: %s", e)
        return {**state, "error": str(e), "documents": []}


async def synthesize_node(state: RAGState) -> RAGState:
    if state.get("error"):
        return {**state, "answer": f"Pipeline error: {state['error']}"}
    try:
        answer = await synthesizer.synthesize(
            query=state["query"],
            documents=state["documents"],
        )
        return {**state, "answer": answer}
    except Exception as e:
        logger.error("Synthesize node error: %s", e)
        return {**state, "answer": "Failed to generate answer.", "error": str(e)}


# ── Graph Definition ──────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve",  retrieve_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve",  "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# Compiled graph singleton
rag_pipeline = build_rag_graph()
