import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach uses a "Grid Decomposition and Local Transformation" strategy.
    Hypothesis: By decomposing the grid into smaller subgrids and identifying local transformations within those subgrids, we can improve the LLM's ability to generalize to different grid sizes and patterns. This approach differs from previous ones by focusing on localized analysis and transformations rather than global patterns.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by decomposing the grid and applying local transformations."""

    system_instruction = "You are an expert at identifying local grid transformation patterns and applying them to new grids. Focus on subgrids and their transformations."
    
    # STEP 1: Decompose Grid into Subgrids - with examples!
    decomposition_prompt = f"""
    Decompose the input grid into smaller, overlapping subgrids. Analyze how these subgrids transform in the training examples.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Subgrid Analysis:
    - A 2x2 subgrid [[1,0],[0,1]] transforms into a 4x4 grid where '1's are placed on the diagonal.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Subgrid Analysis:
    - A 2x2 subgrid [[2,8],[8,2]] transforms into a 4x4 grid where each element expands to a 2x2 block of the same value.

    Problem: {problem_text}
    Subgrid Analysis:
    """
    
    # Attempt to decompose the grid and analyze subgrids
    extracted_subgrid_analysis = call_llm(decomposition_prompt, system_instruction)
    print(f"Extracted Subgrid Analysis: {extracted_subgrid_analysis}") # Diagnostic

    # STEP 2: Apply Local Transformations - with examples!
    transformation_prompt = f"""
    Apply the local transformations identified in the subgrid analysis to the input grid.

    Subgrid Analysis: {extracted_subgrid_analysis}
    Input Grid: {problem_text}

    Example 1:
    Subgrid Analysis: A 2x2 subgrid [[1,0],[0,1]] transforms into a 4x4 grid where '1's are placed on the diagonal.
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Subgrid Analysis: A 2x2 subgrid [[2,8],[8,2]] transforms into a 4x4 grid where each element expands to a 2x2 block of the same value.
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Transformed Grid:
    """
    
    # Attempt to apply local transformations
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(transformation_prompt, system_instruction)
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