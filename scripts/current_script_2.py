import os
import re
import math

# Hypothesis: Instead of trying to code the logic, let the LLM directly transform the grid by learning from examples.
# We will use a direct transformation approach with enhanced examples to guide the LLM.
# We will use multiple examples AND validation loop on intermediate and final outputs to improve reliability.

def main(question):
    """Transforms a grid based on examples, using LLM for direct transformation."""
    try:
        # 1. Direct Grid Transformation with Validation Loop
        transformed_grid = transform_grid_with_validation(question)
        return transformed_grid
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def transform_grid_with_validation(question, max_attempts=3):
    """Transforms the grid using a validation loop to ensure correctness."""
    system_instruction = "You are an expert grid transformer."
    prompt = f"""
    You are a grid transformation expert. Analyze the training examples and transform the test input accordingly.
    Return ONLY the transformed grid.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

    Example 2:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[2, 1], [1, 2]]

    Now transform the input grid below:
    {question}
    """

    for attempt in range(max_attempts):
        # Get the transformed grid
        transformed_grid = call_llm(prompt, system_instruction)

        # Verify the output - is it well-formed, and consistent?
        validation_result = verify_grid_format(question, transformed_grid)
        if validation_result["is_valid"]:
            return transformed_grid  # Return valid result immediately
        else:
            # Refine prompt with specific feedback if possible.
            print(f"Validation failed: {validation_result['feedback']}")
            prompt += f"\nYour previous output had formatting problems: {validation_result['feedback']}. Please correct it and retry."

    return "Failed to transform grid correctly after multiple attempts."  # Give up

def verify_grid_format(question, transformed_grid):
    """Verifies the output grid format using regex."""
    try:
        # Check if the output looks like a grid
        if not (transformed_grid.startswith("[[") and transformed_grid.endswith("]]")):
            return {"is_valid": False, "feedback": "Output should start with '[[' and end with ']]'."}
        # More robust check that it is a grid, and also get grid dimensions:
        grid_rows = transformed_grid.strip("[]").split("],[")
        num_rows = len(grid_rows)
        if num_rows == 0:
            return {"is_valid": False, "feedback": "Grid is empty."}
        num_cols = len(grid_rows[0].split(","))
        for row in grid_rows:
            if len(row.split(",")) != num_cols:
                return {"is_valid": False, "feedback": "Rows have inconsistent number of columns."}

        return {"is_valid": True} # It's a grid, looks good

    except Exception as e:
        return {"is_valid": False, "feedback": f"General error: {str(e)}"}

# Helper function for calling the LLM - DO NOT MODIFY
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