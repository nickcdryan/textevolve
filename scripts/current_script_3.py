import os
import re
import math

def main(question):
    """
    Transforms a grid based on patterns in training examples using LLM-driven spatial reasoning and iterative refinement.
    Uses a different approach by focusing on spatial relationships between numbers and using a combination of pattern extraction and grid manipulation.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem, max_attempts=3):
    """Solve grid transformation problems using spatial reasoning, pattern extraction, and grid manipulation."""

    # Hypothesis: By explicitly defining spatial relationships and using a combination of pattern extraction and grid manipulation, we can better guide the LLM.
    system_instruction = "You are an expert in spatial reasoning and grid manipulation. You will identify patterns based on the location of grid elements."

    # Step 1: Extract the training examples and the test input grid from the problem description.
    extraction_prompt = f"""
    Extract the training examples and the test input grid from the problem description.

    Example:
    Problem: Grid Transformation Task... Input Grid: [[1,2],[3,4]] ... Output Grid: [[5,6],[7,8]] ... TEST INPUT: [[9,10],[11,12]]
    Extracted: {{"examples": ["Input Grid: [[1,2],[3,4]] ... Output Grid: [[5,6],[7,8]]"], "test_input": "[[9,10],[11,12]]"}}

    Problem: {problem}
    Extracted:
    """
    extracted_info = call_llm(extraction_prompt, system_instruction)
    print(f"Extracted Info: {extracted_info}")  # Diagnostic output

    # Step 2: Analyze the training examples to identify spatial relationships.
    spatial_analysis_prompt = f"""
    Analyze the training examples to identify spatial relationships between numbers.
    Focus on identifying rules based on adjacent cells or patterns.

    Example:
    Examples: Input Grid: [[1, 0], [0, 1]] ... Output Grid: [[2, 0], [0, 2]]
    Spatial Relationships: If a cell has value 1, transform it to 2. Otherwise, maintain cell values.

    Examples: {extracted_info}
    Spatial Relationships:
    """
    spatial_relationships = call_llm(spatial_analysis_prompt, system_instruction)
    print(f"Spatial Relationships: {spatial_relationships}")  # Diagnostic output

    # Step 3: Apply transformation based on spatial relationships to the test input
    transformation_prompt = f"""
    Apply the identified spatial relationships to transform the test input grid.

    Spatial Relationships: {spatial_relationships}
    Test Input Grid: {extracted_info}

    Example:
    Spatial Relationships: If a cell has value 1, transform it to 2. Otherwise, maintain cell values.
    Test Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[2, 0], [0, 2]]

    Transformed Grid:
    """
    transformed_grid = call_llm(transformation_prompt, system_instruction)
    print(f"Transformed Grid: {transformed_grid}")  # Diagnostic output

    # Step 4: Verify the transformed grid.
    verification_prompt = f"""
    Verify the transformed grid based on the extracted spatial relationships.

    Spatial Relationships: {spatial_relationships}
    Test Input Grid: {extracted_info}
    Transformed Grid: {transformed_grid}

    Example:
    Spatial Relationships: If a cell has value 1, transform it to 2. Otherwise, maintain cell values.
    Test Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[2, 0], [0, 2]]
    Verification: The grid transformation follows the spatial relationships defined.

    Verification: Does the transformed grid follow the identified spatial relationships? Answer 'yes' or 'no'.
    """
    verification_result = call_llm(verification_prompt, system_instruction)
    print(f"Verification Result: {verification_result}")  # Diagnostic output

    if "yes" in verification_result.lower():
        return transformed_grid
    else:
        return "Unable to transform the grid correctly."

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