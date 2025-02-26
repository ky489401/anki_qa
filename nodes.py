from typing import List, Optional

# --- LangGraph / LangChain imports ---
from langchain_core.messages import AnyMessage, trim_messages
from langchain_openai import ChatOpenAI
from retrieval import faiss_manager, bm25_manager

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeInterrupt

# Pydantic for runtime validation
from pydantic import BaseModel

# ------------------------
# 1) Define RAGState (Pydantic)
# ------------------------
class RAGState(BaseModel):
    messages: List[AnyMessage]  # conversation history
    query: str
    expanded_query: Optional[str] = None
    retrieved_docs: Optional[List[str]] = None
    final_answer: Optional[str] = None
    rerank_needed: bool = False
    summarization_needed: bool = False
    retrieval_method: Optional[str] = None


# ------------------------
# 2) Instantiate Models / Tools
# ------------------------
llm = ChatOpenAI(model="gpt-4o-mini")
# embeddings = OpenAIEmbeddings()
# faiss_db = FAISS.load_local("faiss_index", embeddings)
# bm25_retriever = BM25Retriever()


# ------------------------
# 5) Query Clarification (Step 1)
# ------------------------
def query_clarification(state: RAGState) -> RAGState:
    """
    The LLM checks if the query is clear.
    If it needs clarification, it might ask user to refine the question.
    """

    #TODO This should be a yes no answer
    response = llm.invoke(
        state.messages + [
            {"role": "system", "content": "Is the query clear? If not, ask for clarification."}
        ]
    )
    # Example: if LLM mentions 'clarification', we might return or proceed
    if "clarification" in response.content.lower():
        # TODO implement the node interupt here ...
        # Could also do a NodeInterrupt for human-in-the-loop
        return RAGState(**state.dict(), messages=state.messages + [response])
    return RAGState(**state.dict(), messages=state.messages + [response])

# ------------------------
# 6) Query Expansion (Step 2)
# ------------------------
def query_expansion(state: RAGState) -> RAGState:
    expansion_decision = llm.invoke([
        {"role": "system", "content": f"Should the following query be expanded? {state.query} (yes/no)"}
    ]).content.lower()

    if "yes" in expansion_decision:
        expanded_query_resp = llm.invoke([
            {"role": "system", "content": f"Expand this query: {state.query}"}
        ])
        return RAGState(**state.dict(), expanded_query=expanded_query_resp.content)
    return state

# ------------------------
# 7) Retrieval Decision (Step 3)
# ------------------------
def retrieval_decision(state: RAGState) -> str:
    """
    The LLM decides which retrieval method to use: BM25, FAISS, or BOTH.
    We'll return a simple string so we can do conditional edges.
    """
    decision_resp = llm.invoke([
        {"role": "system", "content": f"Should this query use BM25, FAISS, or both? {state.query}"}
    ]).content.lower()
    if "bm25" in decision_resp:
        return "bm25"
    elif "faiss" in decision_resp:
        return "faiss"
    return "both"

def retrieval_step(state: RAGState, method: str) -> RAGState:
    """
    Actually run BM25, FAISS, or BOTH to get docs, store them in state.
    """
    if method == "bm25":
        docs = bm25_retriever.get_relevant_documents(state.query, k=5)
    elif method == "faiss":
        docs = faiss_db.similarity_search(state.query, k=5)
    else:  # both
        docs = bm25_retriever.get_relevant_documents(state.query, k=3) \
             + faiss_db.similarity_search(state.query, k=3)

    return RAGState(**state.dict(),
                    retrieved_docs=[doc.page_content for doc in docs],
                    retrieval_method=method)

# ------------------------
# 8) Optionally check retrieval quality for multi-step retrieval
# ------------------------
def check_retrieval_quality(state: RAGState) -> str:
    """
    The LLM might decide if the retrieved docs are good enough
    or if we need to re-run retrieval with a bigger 'k' or synonyms.
    Return a node name or key.
    """
    quality_resp = llm.invoke([
        {"role": "system", "content": "Are the retrieved docs sufficient? (yes/no/try again)"}
    ]).content.lower()

    # If the LLM suggests re-running, we can jump back to retrieval_step
    if "try again" in quality_resp:
        return "retrieval_step"
    elif "no" in quality_resp:
        # Could do a NodeInterrupt or ask user for clarifications
        return "query_expansion"
    return "rerank"  # proceed to rerank

# ------------------------
# 9) Rerank (Step 4)
# ------------------------
def rerank(state: RAGState) -> RAGState:
    """
    The LLM decides if we should reorder the docs. Simple example:
    sort by length descending if the LLM says 'yes'.
    """
    decision = llm.invoke([
        {"role": "system", "content": "Should the retrieved results be reranked? (yes/no)"}
    ]).content.lower()

    #TODO implement the actual rerank function ...
    if "yes" in decision:
        sorted_docs = sorted(state.retrieved_docs, key=len, reverse=True) if state.retrieved_docs else []
    else:
        sorted_docs = state.retrieved_docs

    return RAGState(**state.dict(), retrieved_docs=sorted_docs)

# ------------------------
# 10) Summarization or Direct Answer (Step 5)
# ------------------------
def summarization(state: RAGState) -> RAGState:
    """
    Final step: either produce a summarized answer or direct snippet.
    """
    decision = llm.invoke([
        {"role": "system", "content": "Should the answer be summarized? (yes/no)"}
    ]).content.lower()

    if "yes" in decision:
        summary_prompt = f"Summarize the following content: {' '.join(state.retrieved_docs or [])}"
        final_answer = llm.invoke([
            {"role": "system", "content": summary_prompt}
        ]).content
        return RAGState(**state.dict(), summarization_needed=True, final_answer=final_answer)
    else:
        if state.retrieved_docs:
            final_answer = state.retrieved_docs[0]
        else:
            final_answer = "No relevant documents found."
        return RAGState(**state.dict(), summarization_needed=False, final_answer=final_answer)

# ------------------------
# 11) Optional: Human-in-the-Loop Breakpoint
# ------------------------
def human_approval(state: RAGState) -> RAGState:
    """
    Example node where we might pause for user input.
    Raises NodeInterrupt to let a UI or console ask for manual approval.
    """
    raise NodeInterrupt("Manual approval or editing needed. You can check and update the state here.")




# ------------------------
# 12) Build the Agentic RAG Workflow
# ------------------------
builder = StateGraph(RAGState)

#TODO add START
# builder.add_node("human_feedback", human_feedback)
# builder.add_edge(START, "human_feedback")


builder.add_node("query_clarification", query_clarification)
builder.add_edge("filter_messages", "query_clarification")

builder.add_node("query_expansion", query_expansion)
builder.add_edge("query_clarification", "query_expansion")

# retrieval decision
builder.add_node("retrieval_decision", retrieval_decision)
builder.add_edge("query_expansion", "retrieval_decision")

# add conditional edges from retrieval_decision
# e.g. "bm25" => retrieval_step, "faiss" => retrieval_step, "both" => retrieval_step
builder.add_conditional_edges("retrieval_decision", retrieval_decision,
    {"bm25": "retrieval_step", "faiss": "retrieval_step", "both": "retrieval_step"})

builder.add_node("retrieval_step", retrieval_step)

# optionally check retrieval quality
builder.add_node("check_retrieval_quality", check_retrieval_quality)
builder.add_edge("retrieval_step", "check_retrieval_quality")

# add conditional edges from check_retrieval_quality
builder.add_conditional_edges(
    "check_retrieval_quality",
    check_retrieval_quality,
    {
        "try again": "retrieval_step",
        "query_expansion": "query_expansion",
        "rerank": "rerank"
    }
)

builder.add_node("rerank", rerank)
builder.add_edge("rerank", "summarization")

builder.add_node("summarization", summarization)
builder.add_edge("summarization", END)

# (Optional) Add a node for human_approval if you want manual intervention
# builder.add_node("human_approval", human_approval)
# builder.add_edge("retrieval_step", "human_approval")
# etc.

# ------------------------
# 13) Compile with Checkpointer
# ------------------------
agentic_rag = builder.compile()

# ------------------------
# 14) Run the Agentic RAG System
# ------------------------
if __name__ == "__main__":
    # Sample user message
    messages = [{"role": "user", "content": "Explain supervised learning"}]

    # Provide a unique thread_id for memory tracking, e.g. "session-1"
    result = agentic_rag.invoke(
        RAGState(messages=messages, query=messages[0]["content"]),
        config={"configurable": {"thread_id": "session-1"}}
    )

    print("\n==== Final Agentic RAG Resul`t ====")
    print(result)
