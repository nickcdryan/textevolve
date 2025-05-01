import os
import re
import math
import json

# New approach: Decompose the problem into identifying grid features,
# translating the rule, and applying the rule with a verification loop.
# Hypothesis: Explicitly focusing on grid features will improve pattern recognition.

# Function to call the LLM (using the provided template)
def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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

def identify_grid_features(question):
    """Identify key features of the input grids using LLM."""
    prompt = f"""
    Analyze the following question and identify key features of the input and output grids.
    Focus on identifying patterns, dimensions, and value distributions.

    Example:
    Question: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[0, 1], [1, 0]]
    Identified Features:
    - Input grid dimensions: 2x2
    - Output grid dimensions: 4x4
    - Values present: 0, 1
    - Transformation: Each cell expands into a 2x2 block on the diagonal.

    Question: {question}
    Identified Features:
    """
    return call_llm(prompt, system_instruction="You are an expert grid feature identifier.")

def translate_transformation_rule(question, grid_features):
    """Translate the transformation rule from examples into a textual description."""
    prompt = f"""
    Based on the grid features and the examples in the question, translate the transformation rule into a clear, step-by-step textual description.

    Example:
    Question: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[0, 1], [1, 0]]
    Grid Features:
    - Input grid dimensions: 2x2
    - Output grid dimensions: 4x4
    - Values present: 0, 1
    Transformation Description:
    Each cell in the input grid is expanded to occupy a 2x2 block in the output grid, positioned diagonally. All other positions are zero.

    Question: {question}
    Grid Features: {grid_features}
    Transformation Description:
    """
    return call_llm(prompt, system_instruction="You are an expert grid transformation translator.")

def apply_transformation_rule(question, transformation_description):
    """Apply the transformation rule to the test input and generate the transformed grid."""
    prompt = f"""
    Apply the following transformation rule to the test input grid provided in the question and generate the transformed grid.
    Output the grid in a list-of-lists format.

    Example:
    Question: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[0, 1], [1, 0]]
    Transformation Description:
    Each cell in the input grid is expanded to occupy a 2x2 block in the output grid, positioned diagonally. All other positions are zero.
    Transformed Grid:
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]

    Question: {question}
    Transformation Description: {transformation_description}
    Transformed Grid:
    """
    return call_llm(prompt, system_instruction="You are an expert grid transformer.")

def verify_transformed_grid(question, transformed_grid):
    """Verify the transformed grid is correct."""
    prompt = f"""
    Verify if the transformed grid is correct according to the transformation rule in the provided question.
    Return 'Correct' if it is, otherwise return 'Incorrect' and explain why.

    Example:
    Question: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[0, 1], [1, 0]]
    Transformed Grid:
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    Verification: Correct

    Question: {question}
    Transformed Grid: {transformed_grid}
    Verification:
    """
    return call_llm(prompt, system_instruction="You are an expert grid transformation verifier.")

def main(question):
    """Transform the test input grid according to patterns shown in training examples, with feature extraction and verification."""
    try:
        # Step 1: Identify grid features
        grid_features = identify_grid_features(question)
        print(f"Grid Features: {grid_features}")

        # Step 2: Translate the transformation rule
        transformation_description = translate_transformation_rule(question, grid_features)
        print(f"Transformation Description: {transformation_description}")

        # Step 3: Apply the transformation rule
        transformed_grid = apply_transformation_rule(question, transformation_description)
        print(f"Transformed Grid: {transformed_grid}")

        # Step 4: Verify the transformed grid
        verification_result = verify_transformed_grid(question, transformed_grid)
        print(f"Verification Result: {verification_result}")

        return transformed_grid

    except Exception as e:
        return f"An error occurred: {str(e)}"