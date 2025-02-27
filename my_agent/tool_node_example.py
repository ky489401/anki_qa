# weather_agent.py

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from my_agent.config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Define the state structure
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define the tool to fetch weather information
@tool
def get_weather(location: str) -> str:
    """Fetches the current weather for a given location."""
    # Placeholder implementation
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


# Initialize the ToolNode with the defined tool
tools = [get_weather]
tool_node = ToolNode(tools)

# Set up the language model and bind the tool to it
model = ChatOpenAI(model_name="gpt-4o-mini")
model_with_tools = model.bind_tools(tools)


# Function to determine the next step in the workflow
def should_continue(state: State) -> Literal["tools", END]:
    messages = state["messages"]
    print(messages)
    last_message = messages[-1]
    # Check if the last message includes a tool call
    if last_message.tool_calls:
        return "tools"
    return END


# Function to call the language model
def call_model(state: State):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    # Return the updated messages
    return {"messages": [response]}


# Create the workflow graph
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Define the edges between nodes
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Compile the workflow into an executable application
graph = workflow.compile()


if __name__ == "__main__":

    # Define the initial state with a user query
    initial_state = {
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]
    }

    # Execute the workflow with the initial state
    result = graph.invoke(initial_state)

    # Print the final response from the agent
    # print(result["messages"][-1]["content"])
