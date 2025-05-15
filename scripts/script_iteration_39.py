import os
import re
import math

# EXPLORATION: LLM-Driven Transformation with Visual Pattern Encoding and Iterative Refinement
# HYPOTHESIS: We can improve grid transformation accuracy by encoding visual patterns into a simplified representation that the LLM can manipulate more effectively. This representation will focus on relative positions and value changes, and we'll use iterative refinement based on pattern consistency.
# This approach differs from previous attempts by focusing on a simplified pattern encoding.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by encoding visual patterns and iteratively refining the output."""
    try:
        # 1. Encode Visual Patterns
        encoded_patterns_result = encode_visual_patterns(question)
        if not encoded_patterns_result["is_valid"]:
            return f"Error: Could not encode visual patterns. {encoded_patterns_result['error']}"
        encoded_patterns = encoded_patterns_result["encoded_patterns"]

        # 2. Apply Transformation with Pattern-Aware Refinement
        transformed_grid = apply_transformation_with_refinement(question, encoded_patterns, max_attempts)
        return transformed_grid

    except Exception as e:
        return f"Error in solve_grid_transformation: {str(e)}"

def encode_visual_patterns(question):
    """Encodes visual patterns from the training examples into a simplified representation."""
    system_instruction = "You are an expert at encoding visual patterns from grid transformation problems into simplified representations."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and encode the visual patterns into a simplified representation focusing on relative positions and value changes.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[1, 1, 1],
     [0, 0, 0],
     [1, 1, 1]]
    Encoded Patterns:
    "The row with 1s swaps positions with the row above and below it. Values in new rows become 1."

    Problem:
    {question}
    Encoded Patterns:
    """

    encoded_patterns = call_llm(prompt, system_instruction)

    # Validation: Ensure patterns are present
    if encoded_patterns and encoded_patterns.strip():
        return {"is_valid": True, "encoded_patterns": encoded_patterns, "error": None}
    else:
        return {"is_valid": False, "encoded_patterns": None, "error": "Failed to extract transformation patterns."}

def apply_transformation_with_refinement(question, encoded_patterns, max_attempts):
    """Applies the transformation rules to the test input grid with iterative refinement."""
    system_instruction = "You are an expert at applying transformation patterns to grids and iteratively refining the result."

    prompt = f"""
    Given the following grid transformation problem and encoded visual patterns, apply the patterns to the test input grid. After the transformation, validate that the generated grid consistently reflects the encoded patterns. If there are inconsistencies, refine the output.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[1, 1, 1],
     [0, 0, 0],
     [1, 1, 1]]
    Encoded Patterns: "The row with 1s swaps positions with the row above and below it. Values in new rows become 1."
    Test Input:
    [[0, 0, 0],
     [2, 2, 2],
     [0, 0, 0]]
    Completed Grid:
    [[2, 2, 2],
     [0, 0, 0],
     [2, 2, 2]]
    Refinement Reasoning: The pattern is followed.

    Problem:
    {question}
    Encoded Patterns: {encoded_patterns}
    Completed Grid:
    """

    completed_grid = call_llm(prompt, system_instruction)
    return completed_grid

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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"