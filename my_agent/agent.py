import os
import re
from typing import Optional, List

# --- LangGraph / LangChain imports ---
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langsmith import traceable

from my_agent.config import (
    OPENAI_API_KEY,
    embedding_model,
    working_directory_path,
    anki_query,
    langchain_api_key,
)
from my_agent.retrieval.faiss_manager import FAISSManager

os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enables LangSmith tracing
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Pydantic for runtime validation

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load index
new_faiss_mgr = FAISSManager(model_name=embedding_model)

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


@traceable
def combine_card_fields_into_string(retrieved_cards, columns):
    card_fmt = ""
    for card in retrieved_cards:
        card_fmt += "\n".join([f"{col}: {card[col]}" for col in columns])
        card_fmt += "\n" + "*****" * 5 + "\n" * 3
    return card_fmt


class RAGState(BaseModel):
    query: Optional[str] = None
    answer: Optional[str] = None
    retrieved_cards_fmt: Optional[str] = None
    retrieved_cards: Optional[List] = None
    is_retrieved_card_relevant: Optional[List] = None
    is_retrieved_card_relevant_simple: Optional[str] = None
    is_answer_satisfactory: Optional[str] = ""

    class Config:
        arbitrary_types_allowed = True  # Still needed for Pydantic compatibility


@traceable
def get_question(state: RAGState):
    return state


@traceable
def get_anki_cards(state: RAGState):
    results = new_faiss_mgr.query(state.query, top_k=5)
    return state.copy(update={"retrieved_cards": results})


@traceable
# --- Node 1: Analyze the query for ambiguity ---
def rerank_retrieved_cards(state: RAGState):
    class RerankItem(BaseModel):
        id: str = Field(..., description="Anki card id")
        card_title: str = Field(..., description="Anki card title")
        is_card_relevant: bool = Field(
            ..., description="Anki card id which is relevant to the query"
        )
        is_card_relevant_reason: str = Field(
            ...,
            description="Explanation why this card id is considered relevant or irrelevant",
        )

    class RerankResponse(BaseModel):
        results: List[RerankItem] = Field(
            default_factory=list,
            description="List of anki card ids and reasons they are relevant to the query",
        )

    structured_llm = llm.with_structured_output(RerankResponse)

    columns = ["id", "card_title", "summary"]

    card_fmt = combine_card_fields_into_string(state.retrieved_cards, columns)

    # Prompt the LLM to analyze ambiguity
    messages = [
        SystemMessage(content="Identify cards that is useful to the query."),
        HumanMessage(content=f"User query: {state.query}. Anki cards {card_fmt}"),
    ]

    response = structured_llm.invoke(messages)

    return state.copy(
        update={"is_retrieved_card_relevant": response.results}
    )  # Proceed with the given query


# --- Node 1: Analyze the query for ambiguity ---
@traceable
def rerank_retrieved_cards_simple(state: RAGState):

    columns = ["id", "card_title", "summary"]

    card_fmt = combine_card_fields_into_string(state.retrieved_cards, columns)

    # Prompt the LLM to analyze ambiguity
    messages = [
        SystemMessage(
            content="""Identify cards that is useful to the query. Only include card ids for relevant cards. Skip ids for irrelevant Cards. Follow the response format of this example:
            
        Relevant Cards:
        id: 123456
        card title: xxxxxxx
        is_relevant_reason: xxxxxx
        
        id : .....
        
        Irrelevant Cards:
        (Do Not Put An Id Here)
        card title: xxxxxxx
        is_irrelevant_reason: xxxxxx
        
        """
        ),
        HumanMessage(content=f"User query: {state.query}. Anki cards {card_fmt}"),
    ]

    response = llm.invoke(messages).content

    return state.copy(
        update={"is_retrieved_card_relevant_simple": response}
    )  # Proceed with the given query


# --- Node 3: Final Answer Generation ---
@traceable
def answer_query(state: RAGState):
    final_query = state.query

    if state.is_retrieved_card_relevant_simple:
        relevant_card_id_lst = set(
            re.findall(r"id:\s*(\d+)", state.is_retrieved_card_relevant_simple)
        )

    elif state.is_retrieved_card_relevant:
        # get ids for set of relevant cards
        relevant_card_id_lst = set(
            [
                card.id
                for card in state.is_retrieved_card_relevant
                if card.is_card_relevant
            ]
        )

    retrieved_cards = [
        r for r in state.retrieved_cards if r["id"] in relevant_card_id_lst
    ]

    retrieved_cards_fmt = combine_card_fields_into_string(
        retrieved_cards, columns=["card_title", "summary", "text"]
    )

    # Ask the LLM to generate an answer based on the refined query
    messages = [
        SystemMessage(
            content="Produce a clear and concise answers based on the provided anki cards. "
            "Then, List all the card titles which are referenced in the answer."
        ),
        HumanMessage(
            content=f"User's refined query: {final_query}. Anki cards as follows:{retrieved_cards_fmt}"
        ),
    ]

    response = llm.invoke(messages).content

    return state.copy(
        update={
            "answer": response,
            # "retrieved_cards_fmt": retrieved_cards_fmt
        }
    )  # Proceed with the given query


@traceable
def human_feedback_node(state: RAGState):
    """Human feedback node that processes user input and updates the state."""
    while True:
        user_feedback = (
            interrupt({"question": "Do you approve of the output?"}).lower().strip()
        )
        if user_feedback in ["yes", "no"]:
            return state.copy(update={"is_answer_satisfactory": user_feedback})


@traceable
def human_feedback_node_simple(state: RAGState):
    """Human feedback node that processes user input and updates the state."""
    while True:
        user_feedback = input("do you approve of this answer?").lower().strip()
        # user_feedback = interrupt("Do you approve of the output?\n")
        if user_feedback == "yes":
            return END
        elif user_feedback == "no":
            return "answer_query"
        else:
            print("answer must be yes/no")


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

builder.add_conditional_edges(
    "human_feedback_node",
    lambda st: "answer_query" if st.is_answer_satisfactory == "no" else END,
)
checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    for chunk in graph.stream(RAGState(query="what is the meaning of life")):
        print(chunk)
