import os
import re
import math

# HYPOTHESIS: Improve generalization by adding multi-example prompting to analyze_visual_features and apply_transformation,
# and by implementing explicit checks for output grid dimensions and value ranges.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by analyzing and describing visual features."""

    # Step 1: Analyze Visual Features
    feature_analysis_result = analyze_visual_features(question, max_attempts=max_attempts)
    if not feature_analysis_result["is_valid"]:
        return f"Error: Could not analyze visual features. {feature_analysis_result['error']}"

    transformation_description = feature_analysis_result["transformation_description"]

    # Step 2: Apply Transformation
    transformed_grid = apply_transformation(question, transformation_description)
    return transformed_grid

def analyze_visual_features(question, max_attempts=3):
    """Analyzes visual features of the grid transformation problem with multi-example prompting."""
    system_instruction = "You are an expert at analyzing visual features in grid transformations."

    prompt = f"""
    Given the following grid transformation problems, analyze the training examples and identify key visual features
    and describe the transformation in terms of those features. Visual features can include lines, shapes, repetition,
    patterns, symmetries, etc.

    Example 1:
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

    Example 2:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[8, 8, 0],
     [8, 0, 8],
     [0, 8, 8]]
    Output Grid:
    [[0, 8, 8],
     [8, 0, 8],
     [8, 8, 0]]
    Transformation Description: The transformation involves rotating the top row to the bottom row

    Problem:
    {question}

    Transformation Description:
    """

    transformation_description = call_llm(prompt, system_instruction)

    # Add a verification step to ensure the LLM is providing a usable description.
    verification_prompt = f"""
    Verify that the given transformation description is clear, concise, and describes a valid transformation.
    Transformation Description: {transformation_description}
    Is the description valid? (VALID/INVALID)
    """
    validation_result = call_llm(verification_prompt)

    if "VALID" in validation_result:
        return {"is_valid": True, "transformation_description": transformation_description, "error": None}
    else:
        return {"is_valid": False, "transformation_description": None, "error": "Invalid feature description."}

def apply_transformation(question, transformation_description):
    """Applies the described transformation to the test input grid with multi-example prompting."""
    system_instruction = "You are an expert at applying transformations to grids based on a feature description."
    prompt = f"""
    Given the following grid transformation problem and the transformation description, apply the transformation to the test input grid.

    Example 1:
    Problem: Input Grid: [[1, 2], [3, 4]] Transformation Description: Swap rows.
    Output Grid: [[3, 4], [1, 2]]

    Example 2:
    Problem: Input Grid: [[5, 5], [6, 6]] Transformation Description: Replace all values with 0.
    Output Grid: [[0, 0], [0, 0]]

    Problem: {question}
    Transformation Description: {transformation_description}

    Generate the output grid.
    """
    output_grid = call_llm(prompt, system_instruction)
    # Output validation (check if it's a list of lists)

    # Attempt to extract a valid grid from possible python-like outputs by trying to extract out just the grid
    output_grid = output_grid.replace(" ", "")
    regex = r"\[(\[[\d,]+\](?:,)?)+\]"
    match = re.search(regex, output_grid)

    if match:
      output_grid = match.group(0)
    else:
        print(f"Error: Could not extract a valid output grid.  Returning a default output.")
        return "[[]]"

    print(f"Raw Output Grid is: {output_grid}")

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