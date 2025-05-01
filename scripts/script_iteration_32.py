import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach uses a "Transformation Rule Decomposition and Guided Synthesis" strategy.
    Hypothesis: By explicitly decomposing the transformation rule into smaller, more manageable sub-rules, and then guiding the LLM to synthesize the output grid step-by-step using those sub-rules, we can improve accuracy.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by explicitly decomposing transformation rules and guiding synthesis."""

    system_instruction = "You are an expert at identifying grid transformation patterns, decomposing them into sub-rules, and applying them systematically. Focus on explicit, step-by-step synthesis."
    
    # STEP 1: Decompose the Transformation Rule into Sub-Rules - with examples!
    decomposition_prompt = f"""
    Decompose the grid transformation rule into a set of explicit sub-rules that govern how the output grid is generated from the input grid. Each sub-rule should describe a specific transformation operation, such as value mapping, element shifting, or replication.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Decomposed Sub-Rules:
    1. The output grid is larger than the input grid.
    2. The value '1' from the input grid is placed on the diagonal of the output grid.
    3. All other cells in the output grid are assigned the value '0'.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Decomposed Sub-Rules:
    1. Each element in the input grid is expanded into a 2x2 block in the output grid.
    2. The value of each element in the input grid is copied to all cells within its corresponding 2x2 block in the output grid.

    Problem: {problem_text}
    Decomposed Sub-Rules:
    """
    
    # Attempt to decompose the transformation rule
    extracted_sub_rules = call_llm(decomposition_prompt, system_instruction)
    print(f"Extracted Sub-Rules: {extracted_sub_rules}") # Diagnostic

    # STEP 2: Guided Synthesis of the Output Grid - with examples!
    synthesis_prompt = f"""
    You are provided with a set of sub-rules that govern how to generate the output grid from the input grid. Follow these sub-rules step-by-step to synthesize the output grid.

    Sub-Rules:
    {extracted_sub_rules}

    Test Input Grid:
    {problem_text}

    Example 1:
    Sub-Rules:
    1. The output grid is larger than the input grid.
    2. The value '1' from the input grid is placed on the diagonal of the output grid.
    3. All other cells in the output grid are assigned the value '0'.
    Input Grid: [[1, 0], [0, 1]]
    Synthesized Output Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Sub-Rules:
    1. Each element in the input grid is expanded into a 2x2 block in the output grid.
    2. The value of each element in the input grid is copied to all cells within its corresponding 2x2 block in the output grid.
    Input Grid: [[2, 8], [8, 2]]
    Synthesized Output Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Now, apply the sub-rules to synthesize the output grid for the following input:
    """
    
    # Attempt to synthesize the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(synthesis_prompt, system_instruction)
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