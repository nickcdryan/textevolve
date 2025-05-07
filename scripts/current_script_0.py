import os
import re
import math

def main(question):
    """Main function to solve grid transformation problems using LLM-driven approach."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves grid transformation problems using LLM for pattern recognition and transformation."""
    system_instruction = "You are an expert at identifying patterns in grid transformations and applying them."

    # Step 1: Analyze the problem and extract examples using LLM
    analysis_prompt = f"""
    Analyze the following grid transformation problem and extract the transformation rule based on the provided examples.

    Example 1:
    Input Grid:
    [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
    Output Grid:
    [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]]
    Transformation Rule: Each number in the input grid is expanded into a 3x3 block in the output grid.

    Example 2:
    Input Grid:
    [[4, 0, 4], [0, 0, 0], [0, 4, 0]]
    Output Grid:
    [[4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]
    Transformation Rule: Each number in the input grid is expanded into a 3x3 block in the output grid.
    
    Problem: {problem_text}
    Transformation Rule:
    """

    transformation_rule = call_llm(analysis_prompt, system_instruction)

    # Step 2: Apply the transformation rule to the test input using LLM
    application_prompt = f"""
    Apply the following transformation rule to the test input grid.

    Transformation Rule: {transformation_rule}

    Example Input Grid:
    [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    Expected Output Grid:
    [[0, 2, 0], [2, 2, 0], [0, 2, 0], [0, 2, 2], [0, 2, 0], [2, 2, 0], [0, 2, 0], [0, 2, 2], [0, 2, 0]]
    
    Test Problem: {problem_text}
    Transformed Output Grid:
    """

    transformed_grid = call_llm(application_prompt, system_instruction)

    # Step 3: Validate the output format (basic check)
    if not re.match(r"\[.*\]", transformed_grid):
        transformed_grid = "Error: Invalid output format. Could not transform."

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