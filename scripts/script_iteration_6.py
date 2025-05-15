import os
import re
import math

# HYPOTHESIS: A hierarchical decomposition approach will improve the LLM's ability to solve grid transformation problems.
# The LLM will first identify the overall transformation type (e.g., mirroring, rotation, value substitution), then extract specific parameters, and finally apply the transformation.
# This script implements a hierarchical decomposition of the grid transformation problem.
# It first identifies the overall transformation type. It then extracts specific parameters for that transformation.
# Finally, it applies the transformation.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems using a hierarchical decomposition approach."""

    # Step 1: Identify Transformation Type
    transformation_type_result = identify_transformation_type(question, max_attempts=max_attempts)
    if not transformation_type_result["is_valid"]:
        return f"Error: Could not identify transformation type. {transformation_type_result['error']}"

    transformation_type = transformation_type_result["transformation_type"]

    # Step 2: Extract Transformation Parameters
    transformation_parameters_result = extract_transformation_parameters(question, transformation_type, max_attempts=max_attempts)
    if not transformation_parameters_result["is_valid"]:
        return f"Error: Could not extract transformation parameters. {transformation_parameters_result['error']}"

    transformation_parameters = transformation_parameters_result["transformation_parameters"]

    # Step 3: Apply Transformation
    transformed_grid = apply_transformation(question, transformation_type, transformation_parameters)
    return transformed_grid

def identify_transformation_type(question, max_attempts=3):
    """Identifies the overall transformation type (e.g., mirroring, rotation, value substitution)."""
    system_instruction = "You are an expert at identifying the overall type of transformation applied to a grid."
    prompt = f"""
    Given the following grid transformation problem, identify the overall type of transformation applied.
    Possible transformation types include: mirroring, rotation, value substitution, expansion, contraction.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    Transformation Type: mirroring

    Example 2:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Transformation Type: value substitution

    Problem:
    {question}

    Transformation Type:
    """
    transformation_type = call_llm(prompt, system_instruction)
    return {"is_valid": True, "transformation_type": transformation_type, "error": None}

def extract_transformation_parameters(question, transformation_type, max_attempts=3):
    """Extracts the specific parameters for the identified transformation type."""
    system_instruction = "You are an expert at extracting parameters for grid transformations."
    prompt = f"""
    Given the following grid transformation problem and the identified transformation type, extract the specific parameters required to apply the transformation.

    Example 1:
    Transformation Type: mirroring
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    Transformation Parameters: horizontal

    Example 2:
    Transformation Type: value substitution
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Transformation Parameters: increment by 1

    Problem:
    {question}
    Transformation Type: {transformation_type}

    Transformation Parameters:
    """
    transformation_parameters = call_llm(prompt, system_instruction)
    return {"is_valid": True, "transformation_parameters": transformation_parameters, "error": None}

def apply_transformation(question, transformation_type, transformation_parameters):
    """Applies the transformation to the test input grid."""
    system_instruction = "You are an expert at applying transformations to grids."
    prompt = f"""
    Given the following grid transformation problem, the transformation type, and the transformation parameters, apply the transformation to the test input grid.

    Problem: {question}
    Transformation Type: {transformation_type}
    Transformation Parameters: {transformation_parameters}

    Generate the output grid.
    """
    output_grid = call_llm(prompt, system_instruction)
    return output_grid

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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"