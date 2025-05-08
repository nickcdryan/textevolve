import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by first extracting the transformation rule and then applying it."""

    system_instruction = "You are an expert at identifying grid transformation patterns and applying them."
    
    # STEP 1: Extract the transformation rule with embedded examples
    rule_extraction_prompt = f"""
    You are tasked with identifying the transformation rule applied to grids. Study the examples and explain the logic.

    Example 1:
    Input Grid:
    [[1, 0], [0, 1]]
    Output Grid:
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Explanation: Each element in the input grid becomes a diagonal in a larger grid.

    Example 2:
    Input Grid:
    [[2, 8], [8, 2]]
    Output Grid:
    [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Explanation: Each element is expanded to a 2x2 block with the element's value.

    Example 3:
    Input Grid:
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Output Grid:
    [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    Explanation: The input grid is overlaid onto a grid of zeros; 1 replaces 0; 0 remains as 0.

    Now, explain the transformation rule applied to this example. Respond with ONLY the explanation:
    Test Example:
    {problem_text}
    """
    
    # Attempt to extract the rule
    extracted_rule = call_llm(rule_extraction_prompt, system_instruction)

    # STEP 2: Apply the extracted rule to the test input with embedded examples
    application_prompt = f"""
    You have extracted this transformation rule:
    {extracted_rule}

    Now, apply this rule to the following test input grid:
    {problem_text}

    Example Application:
    Extracted Rule: Each number is replaced with its modular inverse with respect to 10.
    Input Grid: [[7, 7, 3, 2, 2], [7, 7, 3, 2, 2], [3, 3, 3, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
    Transformed Grid: [[3, 3, 7, 8, 8], [3, 3, 7, 8, 8], [7, 7, 7, 8, 8], [8, 8, 8, 8, 8], [8, 8, 8, 8, 8]]

    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.
    """
    
    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(application_prompt, system_instruction)
            # Basic validation - check if it looks like a grid
            if "[" in transformed_grid_text and "]" in transformed_grid_text:
                return transformed_grid_text
            else:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # STEP 3: Fallback approach if all attempts fail
    fallback_prompt = f"""
    Apply a simple pattern replacement where each 1 becomes 0 and each 0 becomes 1.
    Input Grid: {problem_text}
    
    Example:
    Input: [[1, 0], [0, 1]]
    Output: [[0, 1], [1, 0]]
    
    What is the output for the given input grid? Respond ONLY with the grid.
    """
    try:
        fallback_grid = call_llm(fallback_prompt, system_instruction)
        return fallback_grid
    except Exception as e:
        print(f"Fallback failed with error: {e}")
        return "[[0,0,0],[0,0,0],[0,0,0]]"

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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