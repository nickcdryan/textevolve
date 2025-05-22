import os
import re
import math

# New Approach: Knowledge Graph Traversal with LLM-Guided Relation Extraction and Focused Answer Validation
# Hypothesis: Constructing a simplified in-memory knowledge graph from the question and LLM-simulated search results, then traversing it with LLM guidance to extract the answer, will improve accuracy by enabling structured reasoning.
# This approach combines information extraction and structured reasoning. It tests whether a structured intermediate representation improves performance.

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types

        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Call the API with system instruction if provided
        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"

def main(question, max_attempts=3):
    """Solve factual questions using Knowledge Graph Traversal with LLM-Guided Relation Extraction."""

    # Step 1: Simulated Search and Initial Information Extraction
    search_query = call_llm(f"Generate a concise search query for the question: {question}", system_instruction="You are an expert search query generator.")
    search_results = call_llm(f"Simulated web search results for: {search_query}. Focus on concise and relevant results.", "You are a helpful search engine.")

    # Step 2: Knowledge Graph Construction (Simplified, In-Memory)
    kg_construction_prompt = f"""
    Extract entities and relationships from the question and search results to build a simplified knowledge graph.

    Example:
    Question: What is the capital of Australia?
    Search Results: Canberra is the capital of Australia.
    Knowledge Graph:
    {{
        "Australia": {{"relation": "capital", "target": "Canberra"}}
    }}

    Question: {question}
    Search Results: {search_results}
    Knowledge Graph:
    """
    knowledge_graph_str = call_llm(kg_construction_prompt, system_instruction="You are an expert knowledge graph builder.")
    print(f"Initial Knowledge Graph:{knowledge_graph_str}")

    # Step 3: Knowledge Graph Traversal (LLM-Guided)
    traversal_prompt = f"""
    Traverse the knowledge graph to find the answer to the question. Follow relationships to reach the target information.

    Example:
    Question: What is the capital of Australia?
    Knowledge Graph:
    {{
        "Australia": {{"relation": "capital", "target": "Canberra"}}
    }}
    Answer: Canberra

    Question: {question}
    Knowledge Graph: {knowledge_graph_str}
    Answer:
    """
    answer = call_llm(traversal_prompt, system_instruction="You are an expert knowledge graph traversal agent.")
    print(f"Answer after Traversal:{answer}")

    # Step 4: Focused Answer Validation
    validation_prompt = f"""
    Validate if the answer is a correct and complete response to the question, given the knowledge graph and original information.

    Example:
    Question: What is the capital of Australia?
    Answer: Canberra
    Validation: VALID - Canberra is the capital of Australia.

    Question: {question}
    Answer: {answer}
    Validation:
    """
    validation_result = call_llm(validation_prompt, system_instruction="You are a strict answer validator.")

    if "VALID" in validation_result:
        return answer
    else:
        return "Could not be validated."