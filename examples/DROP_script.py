import os
import re
import math
import json

# HYPOTHESIS: Combining knowledge graph construction with a more focused ReAct agent will improve multi-hop reasoning.
# The ReAct agent will focus on traversing the knowledge graph and answering the specific question.
# Validation will check for both answer correctness and proper graph traversal.

import openai
from openai import OpenAI

def call_llm(prompt, system_instruction=None):


    # Set your API key (keep this safe and secure)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Call the chat completion endpoint
    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
    )

    # Print the response content
    print(response.output[0].content[0].text)
    return response.output[0].content[0].text

def construct_knowledge_graph(question, documents):
    """Construct a knowledge graph from the documents. This is a simplified representation, focusing on key entities and relationships."""

    graph_construction_prompt = f"""
    Given the question and supporting documents, extract key entities and relationships to construct a knowledge graph.

    Question: {question}
    Documents: {documents}

    Example:
    Question: In what city did Brian May defend his dissertation?
    Documents: "Brian May is an English musician... May defended his dissertation in 2007 at Imperial College London..."
    Graph:
    {{
      "nodes": ["Brian May", "Imperial College London"],
      "edges": [["Brian May", "defended dissertation at", "Imperial College London"]]
    }}

    Now construct the graph for the current question:
    """

    graph_data = call_llm(graph_construction_prompt, "You are a knowledge graph expert.")
    return graph_data

def reason_and_act(question, graph_data, max_iterations=5):
    """Reason about the question and graph, taking actions to find the answer."""
    system_instruction = "You are a ReAct agent that traverses a knowledge graph to answer questions."
    react_prompt = f"""
    You are an agent reasoning about how to answer a question using a knowledge graph.
    The knowledge graph and question are provided below.

    Question: {question}
    Knowledge Graph: {graph_data}

    Follow the ReAct pattern:
    1. Reason about the current state and what action to take.
    2. Take an action: (FindNode[entity_name], FindRelation[entity1, relation, entity2], Finish[answer])
    3. Observe the result of the action.
    4. Repeat until you reach the Finish action.

    Example:
    Question: In what city did Brian May defend his dissertation?
    Knowledge Graph:
    {{
      "nodes": ["Brian May", "Imperial College London", "London"],
      "edges": [["Brian May", "defended dissertation at", "Imperial College London"], ["Imperial College London", "is in", "London"]]
    }}
    Thought 1: I need to find the city where Brian May defended his dissertation. I can start by finding the node related to "Brian May" and the "defended dissertation at" relation.
    Action 1: FindRelation[Brian May, defended dissertation at, ?]
    Observation 1: Imperial College London

    Thought 2: Now I need to find the city that "Imperial College London" is in.
    Action 2: FindRelation[Imperial College London, is in, ?]
    Observation 2: London

    Thought 3: I have found the city.
    Action 3: Finish[London]

    Now, for this question: {question}
    Start reasoning:
    """
    for i in range(max_iterations):
        response = call_llm(react_prompt, system_instruction)
        # Simplified action extraction - more robust parsing needed in real system.
        if "Action" in response and "Finish" in response:
            answer = response.split("Finish[")[1].split("]")[0]
            return answer
        react_prompt += response + "\n"

    return "Could not determine the final answer."

def main(question):
    documents = "[The Supporting Documents go here - but the execution environment populates this already.]"

    # Step 1: Construct Knowledge Graph
    graph_data = construct_knowledge_graph(question, documents)

    # Step 2: Reason and Act with the Knowledge Graph
    answer = reason_and_act(question, graph_data)
    return answer