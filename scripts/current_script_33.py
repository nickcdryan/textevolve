import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach leverages a "Transformation by Analogy and Iterative Refinement" strategy.
    Hypothesis: By identifying analogous transformations from a set of known transformations, and then iteratively refining the application of that transformation, we can improve accuracy. This approach is different because it explicitly tries to find known transformation "families" to bootstrap off of.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by finding analogous transformations and iteratively refining the application."""

    system_instruction = "You are an expert at identifying grid transformation patterns by analogy and iteratively refining the result. Focus on explicit, step-by-step reasoning and validation."
    
    # STEP 1: Identify Analogous Transformation - with examples!
    analogy_prompt = f"""
    Identify the analogous transformation from a set of known transformations.
    Known Transformations:
    1. Element Expansion: Each element in the input grid is expanded into a block in the output grid.
    2. Diagonal Placement: Input elements become the diagonal of a larger grid.
    3. Value Replacement: Certain values in the input grid are replaced with other values based on their location or neighboring values.
    4. Grid Reversal: The grid is reversed either horizontally or vertically.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Analogous Transformation: Diagonal Placement

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Analogous Transformation: Element Expansion

    Problem: {problem_text}
    Analogous Transformation:
    """
    
    # Attempt to identify the analogous transformation
    extracted_analogy = call_llm(analogy_prompt, system_instruction)
    print(f"Extracted Analogy: {extracted_analogy}") # Diagnostic

    # STEP 2: Apply the Analogous Transformation - with examples!
    application_prompt = f"""
    You have identified the analogous transformation as: {extracted_analogy}
    Apply this transformation to the following input grid.
    Input Grid: {problem_text}

    Example 1:
    Analogous Transformation: Diagonal Placement
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Analogous Transformation: Element Expansion
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Transformed Grid:
    """
    
    # Attempt to apply the transformation
    transformed_grid_text = call_llm(application_prompt, system_instruction)
    print(f"Transformed Grid Text: {transformed_grid_text}") # Diagnostic

    # STEP 3: Iterative Refinement - with examples!
    refinement_prompt = f"""
    You have applied the transformation and generated the following grid: {transformed_grid_text}
    However, it may not be perfect. Examine the original problem and the generated grid.
    What could be wrong? Refine the grid to address potential errors.

    Problem: {problem_text}
    Generated Grid: {transformed_grid_text}

    Example 1:
    Problem: Input Grid: [[1, 0], [0, 1]]. Expected Diagonal Placement. The transformation is almost correct, but it needs mirroring.
    Generated Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Refinement: No refinement needed

    Example 2:
    Problem: Input Grid: [[2, 8], [8, 2]]. Expected Element Expansion. The 2x2 blocks are present, but they appear in the wrong order
    Generated Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Refinement: No refinement needed

    Refined Grid:
    """
    
    # Attempt to refine the transformation
    for attempt in range(max_attempts):
      try:
          refined_grid_text = call_llm(refinement_prompt, system_instruction)
          print(f"Refined Grid Text: {refined_grid_text}") # Diagnostic

          # STEP 4: Basic validation: does the output look like a grid?
          if "[" not in refined_grid_text or "]" not in refined_grid_text:
              print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
              continue

          return refined_grid_text

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