import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach uses a "Contextual Value Mapping with Iterative Neighborhood Analysis" strategy.

    Hypothesis: By explicitly analyzing the neighboring values of each cell and their contextual relationships within the grid, the LLM can better identify transformation rules.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by analyzing neighboring values and applying transformations."""

    system_instruction = "You are an expert at identifying grid transformation patterns based on contextual value mappings. Analyze the relationships between neighboring values to determine the transformation rules."
    
    # STEP 1: Analyze Contextual Value Mappings - with examples!
    contextual_mapping_prompt = f"""
    Analyze the training examples and identify contextual value mappings. Focus on how the value of a cell changes based on the values of its neighbors (up, down, left, right, and diagonals). 

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Contextual Value Mappings:
    - A cell with value '1' in the input grid results in a '1' on the diagonal in the output grid. All surrounding cells become '0'.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Contextual Value Mappings:
    - Each cell expands to a 2x2 block of the same value. No contextual dependencies apparent here, the original value just propagates outwards in two dimensions.

    Problem: {problem_text}
    Contextual Value Mappings:
    """
    
    extracted_mappings = call_llm(contextual_mapping_prompt, system_instruction)
    print(f"Extracted Contextual Mappings: {extracted_mappings}") # Diagnostic

    # STEP 2: Iterative Neighborhood Analysis - with examples!
    neighborhood_analysis_prompt = f"""
    Perform an iterative neighborhood analysis. Based on the contextual value mappings, transform the test input grid by considering the immediate neighbors of each cell. If a cell's neighbors influence its value, apply the identified mapping. Otherwise keep as the original value.

    Contextual Value Mappings: {extracted_mappings}
    Input Grid: {problem_text}

    Example 1:
    Contextual Value Mappings: A cell with value '1' results in a '1' on the diagonal. All surrounding cells become '0'.
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Contextual Value Mappings: Each cell expands to a 2x2 block of the same value.
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Transformed Grid:
    """
    
    # Iteratively transform the grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(neighborhood_analysis_prompt, system_instruction)
            print(f"Transformed Grid Text: {transformed_grid_text}") # Diagnostic

            # STEP 3: Basic validation: does the output look like a grid?
            if "[" not in transformed_grid_text or "]" not in transformed_grid_text:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
                continue

            return transformed_grid_text

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