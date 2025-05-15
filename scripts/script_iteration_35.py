import os
import re
import math

# This script improves grid transformation by adding multi-example prompting,
# a new intermediate step (identify_transformation_type), and a stricter validation process.
# It also adds a more explicit output format verification step to address the output size issue.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by analyzing features, identifying type, & applying transformations."""
    transformation_type_result = identify_transformation_type(question)
    if not transformation_type_result["is_valid"]:
        return f"Error: Could not identify transformation type."

    feature_analysis_result = analyze_visual_features(question, transformation_type_result["transformation_type"])
    if not feature_analysis_result["is_valid"]:
        return f"Error: Could not analyze visual features."

    transformed_grid = apply_transformation(question, feature_analysis_result["transformation_description"])
    # Add validation
    if not is_valid_grid(transformed_grid):
      return "Error: Invalid output grid format."
    return transformed_grid

def is_valid_grid(grid_string):
  """Verifies that the grid string represents a 2D array (nested list of numbers)."""
  try:
    # Check if the string starts and ends with brackets
    if not (grid_string.startswith("[[") and grid_string.endswith("]]")):
        return False

    # Check if each row in the grid is a list of numbers
    rows = grid_string.replace("[[", "").replace("]]", "").split("],[")
    for row in rows:
        numbers = row.split(",")
        for number in numbers:
            try:
                int(number.strip())  # Check if it is an integer value
            except ValueError:
                return False
    return True
  except:
    return False

def identify_transformation_type(question):
    """Identifies the type of transformation (e.g., mirroring, rotation, value replacement)."""
    system_instruction = "You are an expert in identifying transformation types in grid patterns."
    prompt = f"""
    Given the following grid transformation problem, identify the *type* of transformation being applied.

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[4, 3], [2, 1]]
    Transformation Type: Mirroring

    Reasoning: The output grid is a mirror image of the input grid.

    Example 2:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[3, 4], [1, 2]]
    Transformation Type: Row Swapping

    Reasoning: The rows of the input grid have been swapped to create the output grid.

    Problem: {question}
    Transformation Type:
    """
    transformation_type = call_llm(prompt, system_instruction)
    return {"is_valid": True, "transformation_type": transformation_type}

def analyze_visual_features(question, transformation_type):
    """Analyzes visual features of the grid transformation problem."""
    system_instruction = "You are an expert at analyzing visual features in grid transformations."
    prompt = f"""
    Given the following grid transformation problem (of type: {transformation_type}), analyze the examples and describe the transformation.

    Example 1:
    === TRAINING EXAMPLES ===
    Input Grid: [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    Output Grid: [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
    Transformation Description: The transformation involves swapping the rows with '1' with adjacent rows.

    Reasoning:
    1. The row containing '1' values in the input is moved to the top in the output
    2. The row containing '1' values from input is moved to the bottom in output, with other rows padded.

    Problem: {question}
    Transformation Description:
    """
    transformation_description = call_llm(prompt, system_instruction)
    return {"is_valid": True, "transformation_description": transformation_description}

def apply_transformation(question, transformation_description):
    """Applies the transformation to the test input grid."""
    system_instruction = "You are an expert at applying transformations to grids based on a feature description."
    prompt = f"""
    Given the following problem and transformation description, apply the transformation to the test input.

    Problem: {question}
    Transformation Description: {transformation_description}

    Example:
    Problem: Input Grid: [[1, 2], [3, 4]] Output Grid: [[4, 3], [2, 1]]. Test Input: [[5, 6], [7, 8]]
    Transformation Description: Mirror the grid horizontally and vertically.

    Output Grid: [[8, 7], [6, 5]]

    Generate the output grid. The output grid should be a nested list of numbers like this: [[1, 2], [3, 4]].
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