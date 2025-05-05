import os
import re
import math

# Hypothesis: By focusing on identifying specific transformation types (e.g., arithmetic, mirroring, neighbor-based),
# and then using specialized prompts for each type, we can improve pattern recognition. Also we will focus on making
# sure the output is in the correct format by adding some format validation.
# The script will attempt to identify the transform type and then apply a special function with different examples
# to address this transformation type.
# This is substantially different than previous approaches as it's classifying the transformation.

def main(question):
    """Transforms a grid based on identified pattern type and specialized transformation."""
    try:
        # 1. Identify the transformation type
        transformation_type = identify_transformation_type(question)

        # 2. Apply specialized transformation based on the type
        if transformation_type == "arithmetic":
            transformed_grid = apply_arithmetic_transformation(question)
        elif transformation_type == "mirroring":
            transformed_grid = apply_mirroring_transformation(question)
        elif transformation_type == "neighbor_based":
            transformed_grid = apply_neighbor_based_transformation(question)
        else:
            transformed_grid = "Unknown transformation type."

        # Verify that the output has proper formatting
        if not (transformed_grid.startswith("[[") and transformed_grid.endswith("]]")):
            transformed_grid = "[[ERROR: The returned grid did not start with '[[' and end with ']]']]"
        return transformed_grid

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def identify_transformation_type(question):
    """Identifies the type of transformation applied to the grid."""
    system_instruction = "You are an expert in classifying grid transformations."
    prompt = f"""
    Classify the type of transformation applied to the grid based on the examples.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Transformation Type: arithmetic (addition of 1 to each element)

    Example 2:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 1], [4, 3]]
    Transformation Type: mirroring (horizontal mirroring)

    Example 3:
    Input Grid: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid: [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    Transformation Type: neighbor_based (each cell becomes the average of its neighbors, then rounded)

    Question: {question}
    Transformation Type:
    """
    response = call_llm(prompt, system_instruction)
    return response.strip().lower()

def apply_arithmetic_transformation(question):
    """Applies an arithmetic transformation to the grid."""
    system_instruction = "You are an expert in arithmetic grid transformations."
    prompt = f"""
    Apply an arithmetic transformation to the grid based on the examples. You MUST return a grid starting with '[[' and ending with ']]'.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Transformed Grid: [[2, 3], [4, 5]]

    Example 2:
    Input Grid: [[5, 6], [7, 8]]
    Output Grid: [[10, 11], [12, 13]]
    Transformed Grid: [[10, 11], [12, 13]]

    Question: {question}
    Transformed Grid:
    """
    response = call_llm(prompt, system_instruction)
    return response.strip()

def apply_mirroring_transformation(question):
    """Applies a mirroring transformation to the grid."""
    system_instruction = "You are an expert in mirroring grid transformations."
    prompt = f"""
    Apply a mirroring transformation to the grid based on the examples. You MUST return a grid starting with '[[' and ending with ']]'.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 1], [4, 3]]
    Transformed Grid: [[2, 1], [4, 3]]

    Example 2:
    Input Grid: [[5, 6, 7], [8, 9, 10]]
    Output Grid: [[7, 6, 5], [10, 9, 8]]
    Transformed Grid: [[7, 6, 5], [10, 9, 8]]

    Question: {question}
    Transformed Grid:
    """
    response = call_llm(prompt, system_instruction)
    return response.strip()

def apply_neighbor_based_transformation(question):
    """Applies a neighbor-based transformation to the grid."""
    system_instruction = "You are an expert in neighbor-based grid transformations."
    prompt = f"""
    Apply a neighbor-based transformation to the grid based on the examples. You MUST return a grid starting with '[[' and ending with ']]'.

    Example 1:
    Input Grid: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid: [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
    Transformed Grid: [[5, 5, 5], [5, 5, 5], [5, 5, 5]]

    Example 2:
    Input Grid: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Output Grid: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    Transformed Grid: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]

    Question: {question}
    Transformed Grid:
    """
    response = call_llm(prompt, system_instruction)
    return response.strip()

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