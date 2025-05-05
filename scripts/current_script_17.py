import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Spatial Relation Encoding and Contextual Modification" approach.
# This is a radically different approach. Instead of directly transforming the grid, we will:
# 1. Encode the spatial relationships between key numbers within the grid.
# 2. Use the LLM to understand and modify these spatial relationships based on training examples.
# 3. Reconstruct the grid based on the modified spatial relationships.
# The core idea is that the *relationships* between the numbers are more important than their absolute values.

def main(question):
    """Transforms a grid by encoding spatial relationships and applying contextual modifications."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Encode spatial relationships in the test input
        spatial_encoding = encode_spatial_relationships(test_input)

        # 3. Apply contextual modifications to the spatial encoding based on training examples
        modified_encoding = apply_contextual_modifications(spatial_encoding, training_examples)

        # 4. Reconstruct the grid from the modified spatial encoding
        transformed_grid = reconstruct_grid(modified_encoding, test_input)

        return transformed_grid

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def preprocess_question(question):
    """Extract training examples and test input from the question string using regex."""
    try:
        training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)

        training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
        test_input = test_input_match.group(1).strip() if test_input_match else ""

        return training_examples, test_input
    except Exception as e:
        return "", ""

def encode_spatial_relationships(test_input):
    """Encodes the spatial relationships between key numbers (e.g., 8s, 1s) in the test input using LLM."""
    system_instruction = "You are an expert in encoding spatial relationships between numbers in a grid."
    prompt = f"""
    You are an expert in encoding spatial relationships between numbers in a grid. Given a test input grid, identify key numbers (e.g., 8, 1, etc.) and encode their spatial relationships (e.g., distance, direction, adjacency).

    Example:
    Test Input: [[0, 8, 0], [8, 0, 8], [0, 8, 0]]
    Encoded Relationships:
    {{
      "8_center": (1,1),  # row, col
      "8_north": (0,1),
      "8_south": (2,1),
      "8_west": (1,0),
      "8_east": (1,2)
    }}

    Test Input:
    {test_input}
    Encoded Relationships:
    """
    spatial_encoding = call_llm(prompt, system_instruction)
    return spatial_encoding

def apply_contextual_modifications(spatial_encoding, training_examples):
    """Applies contextual modifications to the spatial encoding based on training examples using LLM."""
    system_instruction = "You are an expert in applying contextual modifications to spatial encodings based on training examples."
    prompt = f"""
    You are an expert in applying contextual modifications to spatial encodings based on training examples. Given a spatial encoding and training examples, identify how the spatial relationships change between the input and output grids, and apply similar modifications to the given spatial encoding.

    Example:
    Training Examples:
    Input Grid: [[0, 8, 0], [8, 0, 8], [0, 8, 0]]
    Output Grid: [[2, 8, 2], [8, 0, 8], [2, 8, 2]]
    Spatial Encoding:
    {{
      "8_center": (1,1),
      "8_north": (0,1),
      "8_south": (2,1),
      "8_west": (1,0),
      "8_east": (1,2)
    }}
    Modified Encoding: The cells N, S, E, W of 8_center are now 2.

    Training Examples:
    {training_examples}
    Spatial Encoding:
    {spatial_encoding}
    Modified Encoding:
    """
    modified_encoding = call_llm(prompt, system_instruction)
    return modified_encoding

def reconstruct_grid(modified_encoding, test_input):
    """Reconstructs the grid from the modified spatial encoding using LLM."""
    system_instruction = "You are an expert in reconstructing grids from modified spatial encodings, ensuring proper format."
    prompt = f"""
    You are an expert in reconstructing grids from modified spatial encodings, ensuring proper format. Given a test input and a modified spatial encoding, reconstruct the output grid such that it aligns with the new encodings, and has proper double brackets for the grid object.

    Example:
    Test Input: [[0, 8, 0], [8, 0, 8], [0, 8, 0]]
    Modified Encoding: The cells N, S, E, W of 8_center are now 2.
    Reconstructed Grid: [[2, 8, 2], [8, 0, 8], [2, 8, 2]]

    Test Input:
    {test_input}
    Modified Encoding:
    {modified_encoding}
    Reconstructed Grid:
    """
    reconstructed_grid = call_llm(prompt, system_instruction)
    return reconstructed_grid

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types
        import os

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