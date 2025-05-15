import os
import re
import math

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by analyzing features, identifying type, & applying transformations."""
    transformation_type_result = identify_transformation_type(question)
    if not transformation_type_result["is_valid"]:
        return f"Error: Could not identify transformation type."

    feature_analysis_result = analyze_visual_features(question, transformation_type_result["transformation_type"])
    if not feature_analysis_result["is_valid"]:
        return f"Error: Could not analyze visual features."

    transformed_grid = apply_transformation(question, feature_analysis_result["transformation_description"])
    return transformed_grid

def identify_transformation_type(question):
    """Identifies the type of transformation (e.g., mirroring, rotation, value replacement). Includes example for the LLM."""
    system_instruction = "You are an expert in identifying transformation types in grid patterns."
    prompt = f"""
    Given the following grid transformation problem, identify the *type* of transformation being applied.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    Transformation Type: Mirroring (along both diagonals)

    Reasoning: The elements are reflected across both diagonals.

    Example 2:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[3, 4], [1, 2]]
    Transformation Type: Row Swapping

    Reasoning: The first and second rows are swapped.

    Problem: {question}
    Transformation Type:
    """
    transformation_type = call_llm(prompt, system_instruction)
    return {"is_valid": True, "transformation_type": transformation_type}

def analyze_visual_features(question, transformation_type):
    """Analyzes visual features of the grid transformation problem. Includes example for the LLM."""
    system_instruction = "You are an expert at analyzing visual features in grid transformations."
    prompt = f"""
    Given the following grid transformation problem (of type: {transformation_type}), analyze the examples and describe the transformation.

    Example 1:
    === TRAINING EXAMPLES ===
    Input Grid: [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    Output Grid: [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
    Transformation Description: The transformation involves swapping the rows with '1's with the adjacent rows. The '1's row moves to the top, the original top row moves to the center, the original center row moves to the bottom.

    Reasoning: Identify the row with all '1's and move to top.

    Problem: {question}
    Transformation Description:
    """
    transformation_description = call_llm(prompt, system_instruction)
    return {"is_valid": True, "transformation_description": transformation_description}

def apply_transformation(question, transformation_description):
    """Applies the transformation to the test input grid. Includes validation."""
    system_instruction = "You are an expert at applying transformations to grids based on a feature description and validating the result."
    prompt = f"""
    Given the following problem and transformation description, apply the transformation to the test input. Ensure the output is a valid grid.

    Problem: {question}
    Transformation Description: {transformation_description}

    Example Output Format: The output grid should be a nested list of numbers like this: [[1, 2], [3, 4]].

    Reasoning: Take the input grid from the problem and follow the transformation description to complete the transform to the test input grid. Make sure all elements of the grid are correct, and that the correct format is used, and that it follows all aspects of the transformation and reasoning. Ensure that the output grid is a valid grid. Check output grid size for consistency with what is expected.

    Output Grid:
    """
    output_grid = call_llm(prompt, system_instruction)
    # Basic validation (can be enhanced)
    if not isinstance(output_grid, str):
        return "Error: Output grid must be a string."

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