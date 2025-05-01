import os
import re
import math

# This script uses a novel approach: analyzing the distribution of values within the grid
# and predicting transformations based on common distributions observed in training data.
# The hypothesis is that by understanding the value landscape, the LLM can infer transformations.
# A verification step is added to check value distribution before and after transformation.

def main(question):
    """Transforms a grid by analyzing and applying value distribution patterns with LLM."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by analyzing value distributions and applying transformations."""

    system_instruction = "You are an expert at identifying value distribution patterns in grids and applying transformations based on these patterns."

    # STEP 1: Analyze value distribution in training data
    distribution_analysis_prompt = f"""
    Analyze the value distribution in the input and output grids to understand how values are transformed.

    Example 1:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Analysis: The input has a sparse distribution of 1s and 0s. The output maintains this sparsity but expands the distribution diagonally.

    Example 2:
    Input Grid: [[2, 8], [8, 2]]
    Output Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Analysis: The input contains two distinct values (2 and 8) with a balanced distribution. The output expands each value into a 2x2 block.

    Analyze the value distribution in this example:
    {problem_text}
    """

    value_distribution_analysis = call_llm(distribution_analysis_prompt, system_instruction)

    # STEP 2: Apply transformation based on value distribution patterns
    transformation_application_prompt = f"""
    Based on the value distribution analysis: {value_distribution_analysis}, apply a transformation to the input grid.

    Example:
    Analysis: Input is sparse with mostly 0s, output expands the non-zero values diagonally.
    Input Grid: [[3, 0], [0, 3]]
    Transformed Grid: [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]

    Apply the transformation based on this distribution analysis to the given input:
    {problem_text}

    Output the transformed grid as a 2D array formatted as a string.
    """

    transformed_grid_text = call_llm(transformation_application_prompt, system_instruction)

    # STEP 3: Verify the value distribution is maintained
    verification_prompt = f"""
    Verify that the value distribution in the transformed grid is consistent with the initial analysis.

    Original Analysis: {value_distribution_analysis}
    Transformed Grid: {transformed_grid_text}

    Example:
    Analysis: Input sparse with mostly 0s, output expands non-zero values diagonally.
    Transformed Grid: [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]
    Result: Value distribution is maintained.

    Is the value distribution maintained in the transformed grid? Answer 'Yes' or 'No'.
    """

    verification_result = call_llm(verification_prompt, system_instruction)

    if "Yes" in verification_result:
        return transformed_grid_text
    else:
        return "[[0,0,0],[0,0,0],[0,0,0]]"  # Fallback if value distribution is not maintained

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