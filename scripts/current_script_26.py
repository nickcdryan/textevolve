import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach focuses on identifying minimal transformation sets and applying them.
    Hypothesis: By focusing on minimal changes, we can improve generalization.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by identifying a minimal transformation set and applying it."""

    system_instruction = "You are an expert at identifying minimal grid transformation sets and applying them. Focus on the FEWEST changes needed."
    
    # STEP 1: Identify the minimal transformation set
    transformation_set_prompt = f"""
    Identify the MINIMAL transformation set that explains the grid transformations. Provide the transformation set as a bulleted list of changes, focusing on the fewest changes possible.

    Example:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n\n=== TEST INPUT ===\n[[2, 8], [8, 2]]\n
    Minimal Transformation Set:
    - Expand grid to 4x4.
    - Place original values along the diagonal.
    - Fill non-diagonal values with 0.

    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Minimal Transformation Set:
    - Expand each value to a 2x2 block of the same value.

    Problem: {problem_text}
    Minimal Transformation Set:
    """
    
    # Attempt to identify the transformation set
    extracted_transformation_set = call_llm(transformation_set_prompt, system_instruction)
    print(f"Extracted Transformation Set: {extracted_transformation_set}") # Diagnostic

    # STEP 2: Apply the extracted transformation set
    application_prompt = f"""
    Apply the following minimal transformation set to the test input grid:
    {extracted_transformation_set}

    Test Input Grid:
    {problem_text}

    Example:
    Minimal Transformation Set:
    - Expand each value to a 2x2 block of the same value.
    Input Grid: [[1, 2], [3, 4]]
    Transformed Grid: [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

    Now apply the transformation set. Provide the transformed grid as a 2D array formatted as a string.
    """
    
    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(application_prompt, system_instruction)
            print(f"Transformed Grid Text: {transformed_grid_text}") # Diagnostic

            # STEP 3: Verify the generated grid using an example.
            verification_prompt = f"""
            You have extracted the transformation set:\n{extracted_transformation_set}\nand generated the transformed grid:\n{transformed_grid_text}\n

            Is this transformed grid a valid application of the transformation set to the problem:\n{problem_text}?\n
            Respond with ONLY 'VALID' or 'INVALID' followed by a brief explanation.
            """

            validation_result = call_llm(verification_prompt, system_instruction)
            print(f"Validation Result: {validation_result}")

            if "VALID" in validation_result:
                if "[" in transformed_grid_text and "]" in transformed_grid_text:  # Basic grid check
                    return transformed_grid_text
                else:
                    print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
            else:
                print(f"Attempt {attempt+1} failed validation: {validation_result}. Retrying...")
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