import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach uses a "Transformation Propagation Network" to identify and apply the transformation.
    Hypothesis: By focusing on how transformations propagate through the grid, we can improve generalization.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by identifying and propagating transformations."""

    system_instruction = "You are an expert at identifying grid transformation patterns and applying them. Focus on how transformations PROPAGATE through the grid."
    
    # STEP 1: Identify the transformation propagation network
    propagation_network_prompt = f"""
    Identify the transformation propagation network that explains how changes in one part of the grid affect other parts. Focus on identifying "source" elements and how their values influence "destination" elements.  Output as a series of source-destination mappings and rules.

    Example:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    Transformation Propagation Network:
    - Source: Diagonal elements (1s)
    - Destination: Corresponding diagonals in the larger grid
    - Rule: Copy the source value (1) to the destination diagonal.

    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    Transformation Propagation Network:
    - Source: Each element
    - Destination: A 2x2 block surrounding the element
    - Rule: Copy the source value to all elements in the destination block.

    Problem: {problem_text}
    Transformation Propagation Network:
    """
    
    # Attempt to identify the propagation network
    extracted_propagation_network = call_llm(propagation_network_prompt, system_instruction)
    print(f"Extracted Propagation Network: {extracted_propagation_network}") # Diagnostic

    # STEP 2: Apply the transformation propagation network
    application_prompt = f"""
    Apply the following transformation propagation network to the test input grid:
    {extracted_propagation_network}

    Test Input Grid:
    {problem_text}

    Example:
    Transformation Propagation Network:
    - Source: Each element
    - Destination: A 2x2 block surrounding the element
    - Rule: Copy the source value to all elements in the destination block.
    Input Grid: [[1, 2], [3, 4]]
    Transformed Grid: [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

    Now apply the transformation network, showing your reasoning. Provide the transformed grid as a 2D array formatted as a string.
    """
    
    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(application_prompt, system_instruction)
            print(f"Transformed Grid Text: {transformed_grid_text}") # Diagnostic

            # STEP 3: Basic validation:  does the output look like a grid?
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