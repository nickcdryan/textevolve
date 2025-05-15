import os
import re
import math

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by analyzing visual features and applying transformations."""

    feature_analysis_result = analyze_visual_features(question, max_attempts=max_attempts)
    if not feature_analysis_result["is_valid"]:
        return f"Error: Could not analyze visual features. {feature_analysis_result['error']}"

    transformation_description = feature_analysis_result["transformation_description"]

    transformed_grid = apply_transformation(question, transformation_description)
    return transformed_grid

def analyze_visual_features(question, max_attempts=3):
    """Analyzes visual features of the grid transformation problem."""
    system_instruction = "You are an expert at analyzing visual features in grid transformations."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and identify key visual features
    and describe the transformation in terms of those features. Visual features can include lines, shapes, repetition,
    patterns, symmetries, etc.

    Example:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[1, 1, 1],
     [0, 0, 0],
     [1, 1, 1]]
    Transformation Description: The transformation involves swapping the rows with '1' with adjacent rows.

    Problem:
    {question}

    Transformation Description:
    """

    transformation_description = call_llm(prompt, system_instruction)

    #Improved verification with more specific instructions.
    verification_prompt = f"""
    Verify that the given transformation description is clear, concise, describes a valid transformation,
    and provides enough detail to perform the transformation.
    Transformation Description: {transformation_description}
    Is the description valid? (VALID/INVALID). Provide a brief explanation after the VALID/INVALID label.
    """
    validation_result = call_llm(verification_prompt)

    if "VALID" in validation_result:
        return {"is_valid": True, "transformation_description": transformation_description, "error": None}
    else:
        return {"is_valid": False, "transformation_description": None, "error": f"Invalid feature description: {validation_result}"}

def apply_transformation(question, transformation_description):
    """Applies the described transformation to the test input grid."""
    system_instruction = "You are an expert at applying transformations to grids based on a feature description."
    prompt = f"""
    Given the following grid transformation problem and the transformation description, apply the transformation to the test input grid.

    Problem: {question}
    Transformation Description: {transformation_description}

    Example:
    Problem: Input Grid: [[0, 0], [1, 1]] Output Grid: [[1, 1], [0, 0]]
    Transformation Description: Swap the rows.
    Test Input: [[2, 2], [3, 3]]
    Output Grid: [[3, 3], [2, 2]]

    Generate the output grid. Ensure output is a python list of lists.
    """
    output_grid = call_llm(prompt, system_instruction)

    #Adding simple validation to see if the output is list of lists.
    if not (output_grid.startswith("[[") and output_grid.endswith("]]")):
        return "Error: Output grid is not in the correct format (list of lists)."

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