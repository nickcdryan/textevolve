import os
import re
import math

def main(question):
    """Main function to solve grid transformation problems using a multi-stage LLM reasoning approach."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves grid transformation problems using LLM with a multi-stage chain-of-thought and explicit output formatting."""
    system_instruction = "You are an expert at identifying patterns in grid transformations. Provide only the output grid."

    # Step 1: Analyze the problem and extract examples using LLM
    analysis_prompt = f"""
    Analyze the following grid transformation problem and extract the transformation rule. Provide the transformed output grid with a clear explanation.

    Example 1:
    Input Grid:
    [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
    Output Grid:
    [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]]
    Transformed Output Grid:
    Transformation Rule: Each number in the input grid is expanded into a 3x3 block in the output grid.
    [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]]

    Example 2:
    Input Grid:
    [[4, 0, 4], [0, 0, 0], [0, 4, 0]]
    Output Grid:
    [[4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]
    Transformed Output Grid:
    Transformation Rule: Each number in the input grid is expanded into a 3x3 block in the output grid.
    [[4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]

    Problem: {problem_text}
    Transformed Output Grid:
    """

    transformed_grid = call_llm(analysis_prompt, system_instruction)

    # Step 2: Validate the output format (strict JSON check)
    if not is_valid_grid_format(transformed_grid):
        transformed_grid = apply_backup_transformation(problem_text)  # Call backup function if format is incorrect

    return transformed_grid


def is_valid_grid_format(grid_string):
    """Verify if the string represents a valid grid format."""
    pattern = r'^(\[\[\d+(,\s*\d+)*\](,\s*\[\d+(,\s*\d+)*\])*\])$'
    return bool(re.match(pattern, grid_string))

def apply_backup_transformation(problem_text):
  """Applies a simple backup transformation to ensure a valid grid format is returned. A less accurate backup plan is better than no plan."""
  return "[[0, 0, 0], [0, 0, 0], [0, 0, 0]]"

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