import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven analysis and coordinate-based transformation."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by analyzing coordinate patterns and applying transformations."""

    system_instruction = "You are an expert at identifying grid transformation patterns based on coordinate analysis. You analyze how element values CHANGE based on their ORIGINAL COORDINATES."

    # STEP 1: Analyze coordinate patterns with embedded examples
    coordinate_analysis_prompt = f"""
    You are tasked with identifying transformation rules based on the coordinates of grid elements. Analyze the input and output grids to determine how element values change as a function of their row and column indices.

    Example 1:
    Input Grid:
    [[1, 0], [0, 1]]
    Output Grid:
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Analysis: The element at (r, c) in the input grid is transformed to a diagonal line in the output grid. If input[r][c] == 1, then output[r+c][r+c] = 1. All other elements in the output grid are 0.

    Example 2:
    Input Grid:
    [[2, 8], [8, 2]]
    Output Grid:
    [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Analysis: The element at (r, c) in the input grid is expanded to a 2x2 block in the output grid. output[2r:2r+2][2c:2c+2] = input[r][c].

    Now, analyze the coordinate patterns in this example. Respond with ONLY the analysis:
    Test Example:
    {problem_text}
    """

    # Attempt to analyze coordinate patterns
    coordinate_analysis = call_llm(coordinate_analysis_prompt, system_instruction)

    # STEP 2: Apply the coordinate-based transformation with embedded examples
    transformation_application_prompt = f"""
    You have analyzed the coordinate patterns and determined this transformation rule:
    {coordinate_analysis}

    Now, apply this rule to the following test input grid:
    {problem_text}

    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.

    Example Application:
    Analyzed Rule: The element at (r, c) becomes a 2x2 block with the element's value.
    Input Grid: [[1, 2], [3, 4]]
    Transformed Grid: [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    """

    # Attempt to generate the transformed grid
    transformed_grid_text = call_llm(transformation_application_prompt, system_instruction)

    # STEP 3: Validation - check for format
    if "[" in transformed_grid_text and "]" in transformed_grid_text:
        return transformed_grid_text
    else:
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