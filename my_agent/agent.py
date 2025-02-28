import os
from typing import Optional, List

# --- LangGraph / LangChain imports ---
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from my_agent.config import OPENAI_API_KEY, embedding_model, working_directory_path
from my_agent.retrieval.faiss_manager import FAISSManager

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
        f"{working_directory_path}/artifacts/faiss.index",
        f"{working_directory_path}/artifacts/metadata.pkl",
    )


def combine_card_fields_into_string(retrieved_cards, columns):
    card_fmt = ""
    for card in retrieved_cards:
        card_fmt += "\n".join([f"{col}: {card[col]}" for col in columns])
        card_fmt += "\n" + "*****" * 5 + "\n" * 3
    return card_fmt


class RAGState(BaseModel):
    answer: Optional[str] = None
    query: Optional[str] = None
    retrieved_cards_fmt: Optional[str] = None
    retrieved_cards: Optional[List] = None
    is_retrieved_card_relevant: Optional[List] = None

    class Config:
        arbitrary_types_allowed = True  # Still needed for Pydantic compatibility


def get_question(state: RAGState):
    # state = state.copy(update={"query": user_response})
    return state


def get_anki_cards(state: RAGState):
    results = new_faiss_mgr.query(state.query, top_k=5)
    return state.copy(update={"retrieved_cards": results})


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


# --- Node 3: Final Answer Generation ---
def answer_query(state: RAGState):
    final_query = state.query

    # get ids for set of relevant cards
    relevant_card_id_lst = set(
        [card.id for card in state.is_retrieved_card_relevant if card.is_card_relevant]
    )

    retrieved_cards = [
        r for r in state.retrieved_cards if r["id"] in relevant_card_id_lst
    ]

    retrieved_cards_fmt = combine_card_fields_into_string(
        retrieved_cards, columns=["card_title", "summary"]
    )

    # Ask the LLM to generate an answer based on the refined query
    messages = [
        SystemMessage(
            content="You are an assistant providing clear and concise answers based on the provided anki cards. "
            "quote the source the answer is based on"
        ),
        HumanMessage(
            content=f"User's refined query: {final_query}. Anki cards as follows:{retrieved_cards_fmt}"
        ),
    ]

    response = llm.invoke(messages).content

    return state.copy(
        update={"answer": response, "retrieved_cards_fmt": retrieved_cards_fmt}
    )  # Proceed with the given query


def dummy_node(state: RAGState):
    pass


builder = StateGraph(RAGState)

builder.add_node("get_question", get_question)
builder.add_node("get_anki_cards", get_anki_cards)
# builder.add_node("rerank_retrieved_cards", rerank_retrieved_cards)
builder.add_node("answer_query", answer_query)
builder.add_node("dummy_node", dummy_node)

builder.add_edge(START, "get_question")
builder.add_edge("get_question", "get_anki_cards")
# builder.add_edge("get_anki_cards", "rerank_retrieved_cards")
# builder.add_edge("rerank_retrieved_cards", "answer_query")
builder.add_edge("get_anki_cards", "answer_query")
builder.add_edge("answer_query", END)

graph = builder.compile()

if __name__ == "__main__":
    for chunk in graph.stream(RAGState(query="what is the meaning of life")):
        print(chunk)
