import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach uses a "Transformation Pattern Codebook and Selection" strategy.

    Hypothesis: By having the LLM select a transformation pattern from a pre-defined codebook of possible transformations, we can improve the consistency and accuracy of transformations. This is different because it constrains the LLM to choose from explicit and valid transformation patterns rather than freely hallucinating. We also include verification to ensure that the output follows basic grid rules.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by selecting a transformation pattern from a codebook and then applying it."""

    system_instruction = "You are an expert at identifying grid transformation patterns. You have access to a codebook of possible transformations and must select the most appropriate one. Focus on selecting the correct pattern from the codebook."
    
    # STEP 1: Select a Transformation Pattern from the Codebook - with examples!
    pattern_selection_prompt = f"""
    You are given a grid transformation problem and must select the most appropriate transformation pattern from the following codebook:

    Codebook:
    1. Element Expansion: Each element in the input grid is expanded into a block in the output grid (e.g., 2x2, 3x3).
    2. Diagonal Placement: Input elements are placed along the diagonal of a larger output grid.
    3. Value Replacement: Certain values in the input grid are replaced with other values based on their location or neighboring values.
    4. Grid Reversal: The grid is reversed either horizontally or vertically.
    5. Extract Unique Values: Extract unique values from the input grid to form output grid.
    6. Shift and Fill: Elements are shifted, and empty spaces filled according to a clear pattern.

    For each pattern, also generate an explanation of parameters that need to be applied. Parameters are information like grid expansion size.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Selected Transformation Pattern: Diagonal Placement.
    Parameters: N/A.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Selected Transformation Pattern: Element Expansion.
    Parameters: Expansion size: 2x2

    Problem: {problem_text}
    Selected Transformation Pattern:
    """
    
    # Attempt to select the transformation pattern
    extracted_pattern_selection = call_llm(pattern_selection_prompt, system_instruction)
    print(f"Extracted Pattern Selection: {extracted_pattern_selection}") # Diagnostic

    # STEP 2: Apply the Selected Transformation Pattern - with examples!
    application_prompt = f"""
    You have selected this transformation pattern: {extracted_pattern_selection}
    Apply this transformation to the following input grid. If parameters are needed, extract them from the training examples.
    Input Grid: {problem_text}

    Example 1:
    Selected Transformation Pattern: Diagonal Placement.
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Selected Transformation Pattern: Element Expansion. Parameters: Expansion size: 2x2
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Transformed Grid:
    """
    
    # Attempt to apply the transformation
    transformed_grid_text = call_llm(application_prompt, system_instruction)
    print(f"Transformed Grid Text: {transformed_grid_text}") # Diagnostic

    # STEP 3: Validation of Grid Format
    validation_prompt = f"""
    You have generated the following grid: {transformed_grid_text}
    The problem input is {problem_text}
    Examine the grid format, and provide a basic output that only contains the grid output. If the provided grid is unformatted, you must properly format the solution as nested lists representing the grid, with proper quotation marks.

    Example 1:
    Input Grid: [[1,0],[0,1]]
    Generated Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Validated Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Input Grid: [[1,0],[0,1]]
    Generated Grid: [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    Validated Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Validated Grid:
    """

    # Call to validator
    for attempt in range(max_attempts):
        try:
            validated_grid_text = call_llm(validation_prompt, system_instruction)
            # STEP 4: Basic validation: does the output look like a grid?
            if "[" not in validated_grid_text or "]" not in validated_grid_text:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
                continue

            return validated_grid_text

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