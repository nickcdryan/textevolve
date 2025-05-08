import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach uses a "Transformation by Dissection and Assembly" strategy.
    The grid is 'dissected' into its composite values, patterns of assembly determined by matching of the components.

    Hypothesis: By focusing the LLM's analysis on dissecting the grids into basic components
    (unique values and their arrangement), we can shift the reasoning process from holistic
    transformation to component-based assembly. The script then applies transformation rules to these components before reassembling the output grid.

    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by dissecting into key values, and assembling based on these values."""

    system_instruction = "You are an expert at identifying grid transformation patterns by dissecting the grid into its key components."

    # STEP 1: Extract Key Values and Components - with examples!
    key_components_prompt = f"""
    Analyze the problem and extract the key values and arrangement of components from the grid. Identify the unique values, and their relative positions.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Key Values and Arrangement: The key values are '1' and '0'. The '1's form a diagonal arrangement and the output extends this arrangement in the form of diagonals.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Key Values and Arrangement: The key values are '2' and '8'. These numbers are repeated in blocks of 2x2.

    Problem: {problem_text}
    Key Values and Arrangement:
    """

    extracted_key_components = call_llm(key_components_prompt, system_instruction)
    print(f"Extracted Key Components: {extracted_key_components}")  # Diagnostic

    # STEP 2: Determine the assembly rules of these key components
    assembly_rules_prompt = f"""
    Given the key values and their arrangement, extract the assembly rules from training data.

    Key Values and Arrangement: {extracted_key_components}
    Problem: {problem_text}

    Example 1:
    Key Values and Arrangement: The key values are '1' and '0'. The '1's form a diagonal arrangement and the output extends this arrangement in the form of diagonals.
    Assembly Rules: Take original grid and expand the location of '1's in a diagonal format.

    Example 2:
    Key Values and Arrangement: The key values are '2' and '8'. These numbers are repeated in blocks of 2x2.
    Assembly Rules: Expand the grid size so each cell becomes a 2x2 block with same initial value.

    Assembly Rules:
    """
    assembly_rules = call_llm(assembly_rules_prompt, system_instruction)

    # STEP 3: Apply assembly Rules and return results
    apply_rules_prompt = f"""
    Apply extracted assmebly rules to reconstruct the grid.

    Key Components: {extracted_key_components}
    Assembly Rules: {assembly_rules}
    Input Grid: {problem_text}

    Example 1:
    Key Components: The key values are '1' and '0'. The '1's form a diagonal arrangement and the output extends this arrangement in the form of diagonals.
    Assembly Rules: Take original grid and expand the location of '1's in a diagonal format.
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Key Components: The key values are '2' and '8'. These numbers are repeated in blocks of 2x2.
    Assembly Rules: Expand the grid size so each cell becomes a 2x2 block with same initial value.
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Transformed Grid:
    """
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(apply_rules_prompt, system_instruction)
            print(f"Transformed Grid Text: {transformed_grid_text}")

            # STEP 4: Basic validation: does the output look like a grid?
            if "[" not in transformed_grid_text or "]" not in transformed_grid_text:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
                continue

            return transformed_grid_text

        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # Fallback approach if all attempts fail
    return "[[0,0,0],[0,0,0],[0,0,0]]"


def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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