import os
import re
import math

# EXPLORATION: Knowledge Graph-Based Transformation with Explicit Coordinate References and Multi-Example Learning
# HYPOTHESIS: By having the LLM build a "knowledge graph" representing relationships between grid elements, and explicitly referencing coordinates in transformation rules, we can improve generalization and spatial reasoning.
# We will use a multi-example learning approach to teach the LLM how to construct and utilize this knowledge graph. This approach addresses the
# previously identified weaknesses of failing to capture spatial transformations, incorrect rule applications, and pattern generalization.

def solve_grid_transformation(question):
    """Solves grid transformation problems by constructing a knowledge graph and applying coordinate-based transformations."""
    try:
        # Step 1: Construct Knowledge Graph
        knowledge_graph_result = construct_knowledge_graph(question)
        if not knowledge_graph_result["is_valid"]:
            return f"Error: Could not construct knowledge graph. {knowledge_graph_result['error']}"
        knowledge_graph = knowledge_graph_result["knowledge_graph"]

        # Step 2: Apply Transformation using Knowledge Graph
        transformed_grid = apply_transformation(question, knowledge_graph)
        return transformed_grid
    except Exception as e:
        return f"Error in solve_grid_transformation: {str(e)}"

def construct_knowledge_graph(question):
    """Constructs a knowledge graph representing relationships between grid elements."""
    system_instruction = "You are an expert at constructing knowledge graphs from grid transformation problems. Your goal is to represent the relationships between grid elements in a structured format."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and construct a knowledge graph that represents the relationships between grid elements.
    The knowledge graph should include nodes representing grid elements (with their coordinates and values) and edges representing relationships between them. Use explicit coordinate references.

    Example 1:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2], [3, 4]]
    Output Grid:
    [[4, 3], [2, 1]]
    Knowledge Graph:
    {{
      "nodes": [
        {{"id": "0,0", "value": 1}},
        {{"id": "0,1", "value": 2}},
        {{"id": "1,0", "value": 3}},
        {{"id": "1,1", "value": 4}}
      ],
      "edges": [
        {{"source": "0,0", "target": "1,1", "relation": "becomes"}},
        {{"source": "0,1", "target": "1,0", "relation": "becomes"}},
        {{"source": "1,0", "target": "0,1", "relation": "becomes"}},
        {{"source": "1,1", "target": "0,0", "relation": "becomes"}}
      ]
    }}

    Example 2:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 1, 1], [0, 0, 0], [2, 2, 2]]
    Output Grid:
    [[3, 3, 3], [0, 0, 0], [4, 4, 4]]
    Knowledge Graph:
    {{
      "nodes": [
        {{"id": "0,0", "value": 1}},
        {{"id": "0,1", "value": 1}},
        {{"id": "0,2", "value": 1}},
        {{"id": "1,0", "value": 0}},
        {{"id": "1,1", "value": 0}},
        {{"id": "1,2", "value": 0}},
        {{"id": "2,0", "value": 2}},
        {{"id": "2,1", "value": 2}},
        {{"id": "2,2", "value": 2}}
      ],
      "edges": [
        {{"source": "0,0", "target": "0,0", "relation": "add 2"}},
        {{"source": "0,1", "target": "0,1", "relation": "add 2"}},
        {{"source": "0,2", "target": "0,2", "relation": "add 2"}},
        {{"source": "2,0", "target": "2,0", "relation": "add 2"}},
        {{"source": "2,1", "target": "2,1", "relation": "add 2"}},
        {{"source": "2,2", "target": "2,2", "relation": "add 2"}}
      ]
    }}

    Problem:
    {question}

    Knowledge Graph:
    """

    knowledge_graph = call_llm(prompt, system_instruction)

    # Simple validation to ensure that *something* was output
    if knowledge_graph and knowledge_graph.strip():
        return {"is_valid": True, "knowledge_graph": knowledge_graph, "error": None}
    else:
        return {"is_valid": False, "knowledge_graph": None, "error": "Failed to construct knowledge graph."}

def apply_transformation(question, knowledge_graph):
    """Applies the transformation rules to the test input grid, using the knowledge graph."""
    system_instruction = "You are an expert at applying transformation rules using a knowledge graph. Your goal is to transform the test input grid based on the relationships represented in the knowledge graph."

    prompt = f"""
    Given the following grid transformation problem and the knowledge graph, apply the transformations to the test input grid. Provide ONLY the transformed grid as a list of lists.
    Use explicit coordinate references from the training examples in order to help with transformation.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2], [3, 4]]
    Knowledge Graph:
    {{
      "nodes": [
        {{"id": "0,0", "value": 1}},
        {{"id": "0,1", "value": 2}},
        {{"id": "1,0", "value": 3}},
        {{"id": "1,1", "value": 4}}
      ],
      "edges": [
        {{"source": "0,0", "target": "1,1", "relation": "becomes"}},
        {{"source": "0,1", "target": "1,0", "relation": "becomes"}},
        {{"source": "1,0", "target": "0,1", "relation": "becomes"}},
        {{"source": "1,1", "target": "0,0", "relation": "becomes"}}
      ]
    }}
    Output Grid:
    [[4, 3], [2, 1]]

    Problem:
    {question}
    Knowledge Graph:
    {knowledge_graph}
    Output Grid:
    """

    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"