import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by inferring patterns from examples and applying them."""

    system_instruction = "You are an expert at identifying grid transformation patterns from examples and applying them to new grids."
    
    # Improved prompt with multiple examples and clear instructions
    prompt = f"""
    You are tasked with transforming a test input grid based on the patterns observed in the training examples. Study the examples carefully to infer the transformation logic.

    Example 1:
    Input Grid:
    [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
    Output Grid:
    [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]]
    Reasoning: The transformation expands the grid. Each original cell's value is used to populate a 3x3 block in the output.

    Example 2:
    Input Grid:
    [[4, 0, 4], [0, 0, 0], [0, 4, 0]]
    Output Grid:
    [[4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]
    Reasoning: The grid is expanded, and values are used to populate a 3x3 block based on the original cell's coordinates.

    Example 3:
    Input Grid:
    [[0, 0, 0], [0, 0, 2], [2, 0, 2]]
    Output Grid:
    [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 2, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 2], [2, 0, 2, 0, 0, 0, 2, 0, 2]]
    Reasoning: Similar expansion pattern, but with different original values and block placements.
    
    Now, transform this test input grid according to the patterns:
    Test Input Grid:
    {problem_text}
    
    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.
    """
    
    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(prompt, system_instruction)
            # Basic validation - check if it looks like a grid
            if "[" in transformed_grid_text and "]" in transformed_grid_text:
                return transformed_grid_text
            else:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # Fallback approach if all attempts fail
    return "[[0,0,0],[0,0,0],[0,0,0]]"

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