import os
import re
import math

# EXPLORATION: Grid Transformation using Local Pattern Analysis with Adaptive Context and Iterative Refinement
# HYPOTHESIS: We can improve grid transformation accuracy by analyzing local patterns around each cell and then iteratively refining the transformation based on those local patterns. The approach attempts to model what is "nearby" the target cell and adjust accordingly.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by analyzing local patterns and iteratively refining the transformation."""
    try:
        # Step 1: Analyze local patterns around each cell in the input grid.
        local_patterns_result = analyze_local_patterns(question)
        if not local_patterns_result["is_valid"]:
            return f"Error: Could not analyze local patterns. {local_patterns_result['error']}"
        local_patterns = local_patterns_result["patterns"]

        # Step 2: Apply transformation based on local patterns and original values
        transformed_grid = apply_transformation(question, local_patterns)
        return transformed_grid

    except Exception as e:
        return f"Error in solve_grid_transformation: {str(e)}"

def analyze_local_patterns(question):
    """Analyzes local patterns around each cell in the input grid using a LLM for pattern recognition."""
    system_instruction = "You are an expert at identifying local patterns in grid transformation problems."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and identify local patterns around each cell in the grid. Focus on the immediate neighborhood of each cell.

    Example 1:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[2, 2, 2],
     [1, 1, 1],
     [2, 2, 2]]
    Local Patterns:
    - A '0' surrounded by other '0's becomes a '2'.
    - A '1' remains a '1'.

    Problem:
    {question}
    Local Patterns:
    """

    patterns = call_llm(prompt, system_instruction)

    # Validation: Check if patterns are not empty and a sensible statement
    if patterns and patterns.strip():
        return {"is_valid": True, "patterns": patterns, "error": None}
    else:
        return {"is_valid": False, "patterns": None, "error": "Failed to identify local patterns."}

def apply_transformation(question, local_patterns):
    """Applies the transformation rules to the test input grid."""
    system_instruction = "You are an expert at applying transformation rules to grids based on local patterns."

    prompt = f"""
    Given the following grid transformation problem and the local patterns, apply the patterns to the test input grid. Focus the transformation on the information captured by the 'Local Patterns' portion of the prompt.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[2, 2, 2],
     [1, 1, 1],
     [2, 2, 2]]
    Local Patterns:
    - A '0' surrounded by other '0's becomes a '2'.
    - A '1' remains a '1'.
    Test Input:
    [[5, 5, 5],
     [6, 6, 6],
     [7, 7, 7]]
    Completed Grid:
    [[2, 2, 2],
     [6, 6, 6],
     [2, 2, 2]]

    Problem:
    {question}
    Local Patterns: {local_patterns}
    Completed Grid:
    """

    completed_grid = call_llm(prompt, system_instruction)
    return completed_grid

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