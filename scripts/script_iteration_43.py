import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by identifying and applying patterns using a "Visual Attention and Transformation Synthesis" strategy."""

    system_instruction = "You are an expert at visually attending to relevant parts of the grid and synthesizing transformations. Identify key elements and their interactions within the grid."
    
    # STEP 1: Visual Attention and Key Element Identification - with examples!
    attention_prompt = f"""
    Visually attend to the grid and identify the key elements that influence the transformation. Focus on patterns, symmetries, and unique values that stand out.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Key Elements: The '1's form a diagonal pattern. The output expands the grid while maintaining this diagonal.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Key Elements: The '2' and '8' values are repeated in 2x2 blocks. The grid expands with this repetition.

    Problem: {problem_text}
    Key Elements:
    """
    
    extracted_elements = call_llm(attention_prompt, system_instruction)
    print(f"Extracted Key Elements: {extracted_elements}")  # Diagnostic

    # STEP 2: Transformation Synthesis - with examples!
    synthesis_prompt = f"""
    Synthesize the transformation rule based on the key elements identified. How do these elements interact to create the output grid?

    Key Elements: {extracted_elements}
    Problem: {problem_text}

    Example 1:
    Key Elements: The '1's form a diagonal pattern. The output expands the grid while maintaining this diagonal.
    Transformation Rule: Expand the grid, placing '1's along the diagonal and '0's elsewhere.

    Example 2:
    Key Elements: The '2' and '8' values are repeated in 2x2 blocks. The grid expands with this repetition.
    Transformation Rule: Expand each cell into a 2x2 block with the original cell's value.

    Transformation Rule:
    """
    
    transformation_rule = call_llm(synthesis_prompt, system_instruction)
    print(f"Transformation Rule: {transformation_rule}") # Diagnostic
    
    # STEP 3: Grid Reconstruction - with examples!
    reconstruction_prompt = f"""
    Reconstruct the transformed grid based on the identified transformation rule. Apply this rule to create the output grid.

    Transformation Rule: {transformation_rule}
    Problem: {problem_text}

    Example 1:
    Transformation Rule: Expand the grid, placing '1's along the diagonal and '0's elsewhere.
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Transformation Rule: Expand each cell into a 2x2 block with the original cell's value.
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Transformed Grid:
    """
    #Attempt to refine rule and apply
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(reconstruction_prompt, system_instruction)
            print(f"Transformed Grid Text: {transformed_grid_text}") # Diagnostic

            # STEP 4: Basic validation: does the output look like a grid?
            if "[" not in transformed_grid_text or "]" not in transformed_grid_text:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
                continue

            # STEP 5: Validation using example from training set
            validation_prompt = f"""You will be given a transformed grid and a training example to validate
            Transformed Grid Text: {transformed_grid_text}
            Problem: {problem_text}

            Is the transformed grid a valid transformation based on the test problem given.
            Respond with VALID if correct or INVALID if incorrect
            """
            validation_output = call_llm(validation_prompt, system_instruction)
            if validation_output == "VALID":
              return transformed_grid_text
            else:
              print(f"Attempt {attempt+1} failed: Output is an invalid grid transformation. Retrying...")
              continue
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")
    # Fallback approach if all attempts fail
    return "[[0,0,0],[0,0,0],[0,0,0]]"

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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