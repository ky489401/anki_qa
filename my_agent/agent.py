import os
import re
from typing import Optional, List

# --- LangGraph / LangChain Imports ---
# Importing classes and functions for messaging, validation, and LLM operations
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# LangGraph specific imports for graph building and tracing
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langsmith import traceable

# Import configuration and retrieval modules from the project
from my_agent.config import (
    OPENAI_API_KEY,
    embedding_model,
    working_directory_path,
    anki_query,
    langchain_api_key,
)
from my_agent.retrieval.faiss_manager import FAISSManager

# Set environment variables for LangChain tracing and API keys
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enables LangSmith tracing
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Instantiate the language model (LLM) with a specific model and temperature settings.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load the FAISS index using the FAISSManager based on the current environment
new_faiss_mgr = FAISSManager(model_name=embedding_model)

# Check if running inside a Docker container to set the appropriate file paths.
if os.path.exists("/.dockerenv"):  # inside docker
    new_faiss_mgr.load_index(
        f"/deps/__outer_my_agent/my_agent/artifacts/faiss.index",
        f"/deps/__outer_my_agent/my_agent/artifacts/metadata.pkl",
    )
else:
    new_faiss_mgr.load_index(
        f"{working_directory_path}/artifacts/faiss_{anki_query}.index",
        f"{working_directory_path}/artifacts/metadata_{anki_query}.pkl",
    )


def combine_card_fields_into_string(retrieved_cards, columns):
    """
    Combine specified fields from each retrieved card into a single formatted string.
    Each card's fields are joined by newlines and separated by a delimiter.
    """
    card_fmt = ""
    for card in retrieved_cards:
        # Format each field of the card using the provided columns list.
        card_fmt += "\n".join([f"{col}: {card[col]}" for col in columns])
        card_fmt += "\n" + "*****" * 5 + "\n" * 3
    return card_fmt


class RAGState(BaseModel):
    """
    Data model to hold the state for the Retrieval-Augmented Generation (RAG) pipeline.
    It contains the original query, LLM answer, and details of retrieved and re-ranked cards.
    """

    query: Optional[str] = None
    answer: Optional[str] = None
    retrieved_cards_fmt: Optional[str] = None
    retrieved_cards: Optional[List] = None
    is_retrieved_card_relevant: Optional[List] = None
    is_retrieved_card_relevant_simple: Optional[str] = None
    is_answer_satisfactory: Optional[str] = ""

    class Config:
        # Allow arbitrary types for compatibility with Pydantic validation
        arbitrary_types_allowed = True


@traceable
def get_question(state: RAGState):
    """
    Node to simply return the current state which holds the user's query.
    """
    return state


@traceable
def get_anki_cards(state: RAGState):
    """
    Query the FAISS index with the user query to retrieve the top 5 Anki cards.
    Updates the state with the retrieved cards.
    """
    results = new_faiss_mgr.query(state.query, top_k=5)
    return state.copy(update={"retrieved_cards": results})


@traceable
def rerank_retrieved_cards(state: RAGState):
    """
    Node to re-rank the retrieved cards using a structured LLM output.
    The LLM is prompted with both the query and formatted card details.
    """

    # Define the structure of each reranked card item
    class RerankItem(BaseModel):
        id: str = Field(..., description="Anki card id")
        card_title: str = Field(..., description="Anki card title")
        is_card_relevant: bool = Field(
            ..., description="Flag indicating if the card is relevant to the query"
        )
        is_card_relevant_reason: str = Field(
            ...,
            description="Explanation why this card is considered relevant or irrelevant",
        )

    # Define the expected structure of the LLM's response
    class RerankResponse(BaseModel):
        results: List[RerankItem] = Field(
            default_factory=list,
            description="List of Anki card ids and reasons they are relevant to the query",
        )

    # Set up the LLM to produce a structured output
    structured_llm = llm.with_structured_output(RerankResponse)

    # Define the columns to extract from the card details
    columns = ["id", "card_title", "summary"]
    card_fmt = combine_card_fields_into_string(state.retrieved_cards, columns)

    # Build the prompt messages for the LLM with system and user messages.
    messages = [
        SystemMessage(content="Identify cards that is useful to the query."),
        HumanMessage(content=f"User query: {state.query}. Anki cards {card_fmt}"),
    ]

    # Invoke the structured LLM and update state with the re-ranked results.
    response = structured_llm.invoke(messages)

    return state.copy(update={"is_retrieved_card_relevant": response.results})


@traceable
def rerank_retrieved_cards_simple(state: RAGState):
    """
    Node to re-rank the retrieved cards using a simpler approach.
    The LLM returns a plain text response listing relevant card IDs.
    """
    columns = ["id", "card_title", "summary"]
    card_fmt = combine_card_fields_into_string(state.retrieved_cards, columns)

    # Build the prompt messages with detailed instructions for the LLM.
    messages = [
        SystemMessage(
            content="""Identify cards that is useful to the query. Only include card ids for relevant cards. Skip ids for irrelevant Cards. Follow the response format of this example:

        Relevant Cards:
        id: 123456
        card title: xxxxxxx
        is_relevant_reason: xxxxxx

        id: .....

        Irrelevant Cards:
        (Do Not Put An Id Here)
        card title: xxxxxxx
        is_irrelevant_reason: xxxxxx
        """
        ),
        HumanMessage(content=f"User query: {state.query}. Anki cards {card_fmt}"),
    ]

    # Invoke the LLM to get the simple re-ranking result.
    response = llm.invoke(messages).content

    return state.copy(update={"is_retrieved_card_relevant_simple": response})


@traceable
def answer_query(state: RAGState):
    """
    Node to generate the final answer.
    It uses the refined query and relevant Anki cards to prompt the LLM.
    """
    final_query = state.query

    # Determine relevant card ids using either the simple or structured re-ranking result.
    if state.is_retrieved_card_relevant_simple:
        relevant_card_id_lst = set(
            re.findall(r"id:\s*(\d+)", state.is_retrieved_card_relevant_simple)
        )
    elif state.is_retrieved_card_relevant:
        # Extract ids from the structured re-ranking results
        relevant_card_id_lst = set(
            [
                card.id
                for card in state.is_retrieved_card_relevant
                if card.is_card_relevant
            ]
        )

    # Filter the retrieved cards based on the relevant card ids
    retrieved_cards = [
        r for r in state.retrieved_cards if r["id"] in relevant_card_id_lst
    ]

    # Format the retrieved cards for inclusion in the prompt
    retrieved_cards_fmt = combine_card_fields_into_string(
        retrieved_cards, columns=["card_title", "summary", "text"]
    )

    # Construct the messages for final answer generation
    messages = [
        SystemMessage(
            content="Produce a clear and concise answers based on the provided anki cards. "
            "Then, List all the card titles which are referenced in the answer."
        ),
        HumanMessage(
            content=f"User's refined query: {final_query}. Anki cards as follows:{retrieved_cards_fmt}"
        ),
    ]

    # Invoke the LLM to generate the final answer
    response = llm.invoke(messages).content

    return state.copy(
        update={
            "answer": response,
            # "retrieved_cards_fmt": retrieved_cards_fmt  # Uncomment if you wish to store formatted cards in the state
        }
    )


@traceable
def human_feedback_node(state: RAGState):
    """
    Node to collect human feedback regarding the generated answer.
    It loops until a valid yes/no response is received.
    """
    while True:
        user_feedback = (
            interrupt({"question": "Do you approve of the output?"}).lower().strip()
        )
        if user_feedback in ["yes", "no"]:
            return state.copy(update={"is_answer_satisfactory": user_feedback})


@traceable
def human_feedback_node_simple(state: RAGState):
    """
    Alternative human feedback node that uses direct input() for feedback.
    Returns END if approved, or routes back to 'answer_query' if not.
    """
    while True:
        user_feedback = input("do you approve of this answer?").lower().strip()
        if user_feedback == "yes":
            return END
        elif user_feedback == "no":
            return "answer_query"
        else:
            print("answer must be yes/no")


# Build the state graph by adding nodes (functions) and defining transitions between them.
builder = StateGraph(RAGState)

builder.add_node("get_question", get_question)
builder.add_node("get_anki_cards", get_anki_cards)
builder.add_node("rerank_retrieved_cards", rerank_retrieved_cards)
builder.add_node("rerank_retrieved_cards_simple", rerank_retrieved_cards_simple)
builder.add_node("answer_query", answer_query)
builder.add_node("human_feedback_node", human_feedback_node)

builder.add_edge(START, "get_question")
builder.add_edge("get_question", "get_anki_cards")
builder.add_edge("get_anki_cards", "rerank_retrieved_cards_simple")
builder.add_edge("rerank_retrieved_cards_simple", "answer_query")
builder.add_edge("answer_query", "human_feedback_node")

# Define a conditional edge based on human feedback
builder.add_conditional_edges(
    "human_feedback_node",
    lambda st: "answer_query" if st.is_answer_satisfactory == "no" else END,
)

# Set up a memory saver checkpoint for the state graph execution
checkpointer = MemorySaver()

# Compile the state graph with the checkpoint
graph = builder.compile(checkpointer=checkpointer)

# Main entry point for running the graph; streams output based on the RAGState.
if __name__ == "__main__":
    for chunk in graph.stream(RAGState(query="what is the meaning of life")):
        print(chunk)
