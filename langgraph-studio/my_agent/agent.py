import os
from typing import List, Optional

# --- LangGraph / LangChain imports ---
from langchain_core.messages import AnyMessage, AIMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

# Pydantic for runtime validation
from pydantic import BaseModel, field_validator

from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class RAGState(BaseModel):
    query: str
    messages: Optional[List[AnyMessage]] = None
    is_query_clear: Optional[bool] = None
    is_query_clear_reason: Optional[str] = None
    clarification_question: Optional[str] = None
    answer: Optional[str] = None

    @field_validator("messages", mode="before")
    def validate_messages(cls, v):
        if not isinstance(v, list) or not all(
            isinstance(m, HumanMessage)
            or isinstance(m, SystemMessage)
            or isinstance(m, AIMessage)
            for m in v
        ):
            raise ValueError("messages must be a list of Message objects")
        return v

    class Config:
        arbitrary_types_allowed = True  # Still needed for Pydantic compatibility


def get_question(state: RAGState):
    # user_response = input("How can i help?")

    user_response = interrupt("How can i help sir?")
    state = state.copy(update={"query": user_response})
    return state


# --- Node 1: Analyze the query for ambiguity ---
def analyze_query(state: RAGState):
    class analyze_query_response(BaseModel):
        is_clear: bool = Field(description="is query clear or not")
        reason: str = Field(description="reason for clear/not clear")
        clarification_question: str = Field(description="clarification_question")

    user_query = state.query

    structured_llm = llm.with_structured_output(analyze_query_response)

    # Prompt the LLM to analyze ambiguity
    messages = [
        SystemMessage(
            content="You are an assistant that determines if a query is ambiguous. "
            "If the query is clear, respond with 'CLEAR'. Otherwise, rephrase it "
            "into a clarifying question."
        ),
        HumanMessage(content=f"User query: {user_query}"),
    ]

    response = structured_llm.invoke(messages)

    return state.copy(
        update={
            "is_query_clear": response.is_clear,
            "is_query_clear_reason": response.reason,
            "clarification_question": response.clarification_question,
        }
    )  # Proceed with the given query


# --- Node 2: Get clarification from the user (human-in-the-loop) ---
def clarify_query(state: RAGState):
    # Interrupt the graph execution to ask the user a question
    # user_response = input(state.clarification_question)
    user_response = interrupt({"question": state.clarification_question})
    return state.copy(
        update={"query": user_response}
    )  # Update the query with user's clarification


# --- Node 3: Final Answer Generation ---
def answer_query(state: RAGState):
    final_query = state.query

    # Ask the LLM to generate an answer based on the refined query
    messages = [
        SystemMessage(
            content="You are an assistant providing clear and concise answers."
        ),
        HumanMessage(content=f"User's refined query: {final_query}"),
    ]

    response = llm.invoke(messages).content

    return state.copy(update={"answer": response})  # Proceed with the given query


builder = StateGraph(RAGState)

builder.add_node("get_question", get_question)
builder.add_node("analyze_query", analyze_query)
builder.add_node("clarify_query", clarify_query)
builder.add_node("answer_query", answer_query)

builder.add_edge(START, "get_question")
builder.add_edge("get_question", "analyze_query")
builder.add_conditional_edges(
    "analyze_query",
    lambda st: "clarify_query" if not st.is_query_clear else "answer_query",
)
builder.add_edge("clarify_query", "analyze_query")
builder.add_edge("analyze_query", END)

graph = builder.compile()

if __name__ == "__main__":
    for chunk in graph.stream(RAGState(query="what is the meaning of life")):
        print(chunk)
