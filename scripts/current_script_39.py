import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.

    This approach uses a "Transformation Pattern Codebook with Iterative Value Adjustment" strategy.

    Hypothesis: By combining a constrained codebook approach with an iterative process that focuses on adjusting individual cell values based on their neighbors and the codebook transformation, we can improve the transformation's accuracy. This strategy aims to leverage both structured pattern selection and local refinement. The different part about this is the added step of iteratively re-evaluating each cell to fine-tune its value based on its neighbor, while remaining within the codebook constraints for the high-level pattern.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by selecting a transformation pattern from a codebook and iteratively refining value placements."""

    system_instruction = "You are an expert at identifying grid transformation patterns and refining individual cell values. Select a pattern from the codebook and then adjust values iteratively."
    
    # STEP 1: Select a Transformation Pattern from the Codebook - with examples!
    pattern_selection_prompt = f"""
    Select the most appropriate transformation pattern from the following codebook, also generating a parameter explanation:

    Codebook:
    1. Element Expansion: Each element is expanded (e.g., 2x2, 3x3). Parameter: Expansion size.
    2. Diagonal Placement: Elements are placed along the diagonal. Parameter: N/A.
    3. Value Replacement: Values are replaced based on location. Parameter: Value mappings.
    4. Grid Reversal: The grid is reversed. Parameter: Direction (horizontal/vertical).
    5. Shift and Fill: Elements are shifted, and spaces filled. Parameter: Shift direction and fill value.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Selected Transformation Pattern: Diagonal Placement. Parameters: N/A

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Selected Transformation Pattern: Element Expansion. Parameters: Expansion size: 2x2

    Problem: {problem_text}
    Selected Transformation Pattern:
    """
    
    extracted_pattern_selection = call_llm(pattern_selection_prompt, system_instruction)
    print(f"Extracted Pattern Selection: {extracted_pattern_selection}") # Diagnostic

    # STEP 2: Apply Initial Transformation
    initial_transformation_prompt = f"""
    Apply the selected transformation pattern to the input grid:

    Selected Transformation Pattern: {extracted_pattern_selection}
    Input Grid: {problem_text}

    Example:
    Selected Transformation Pattern: Diagonal Placement. Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Transformed Grid:
    """

    transformed_grid_text = call_llm(initial_transformation_prompt, system_instruction)
    print(f"Initial Transformed Grid: {transformed_grid_text}")

    # STEP 3: Iterative Value Adjustment
    value_adjustment_prompt = f"""
    Iteratively adjust cell values in the transformed grid based on the selected pattern and neighboring values:

    Selected Transformation Pattern: {extracted_pattern_selection}
    Transformed Grid: {transformed_grid_text}
    Input Grid: {problem_text}

    Example:
    Selected Transformation Pattern: Diagonal Placement. Transformed Grid: [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], Input Grid: [[1, 0], [0, 1]]
    Adjusted Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Adjusted Grid:
    """

    for attempt in range(max_attempts):
        try:
            adjusted_grid_text = call_llm(value_adjustment_prompt, system_instruction)
            print(f"Adjusted Grid Text: {adjusted_grid_text}")

            if "[" not in adjusted_grid_text or "]" not in adjusted_grid_text:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
                continue

            # STEP 4: Verify if the result conforms to the selected pattern:
            verification_prompt = f"""
            Verify if the final result is a valid grid by the Selected Transformaton Pattern

            Selected Transformaton Pattern: {extracted_pattern_selection}
            Final Transformed Grid: {adjusted_grid_text}

            Valid Grid:
            """
            verified_grid_text = call_llm(verification_prompt, system_instruction)

            return verified_grid_text

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