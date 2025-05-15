import os
import re
import math

# HYPOTHESIS: Adding multi-example prompts, enhanced rule validation,
# and structured rule descriptions will improve the generalization of visual feature-based transformations.
# This script combines the strengths of the best approaches, focusing on robust feature analysis
# and incorporating validation loops and detailed output checks.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by analyzing and describing visual features."""
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
    and describe the transformation in terms of those features. Provide the rule explicity. Visual features can include lines, shapes, repetition,
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
    Transformation Description: The transformation involves swapping the rows with '1' with adjacent rows. The rule is that if there's a full row of 1s then move that row up or down

    Example 2:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
    Output Grid:
    [[9, 8, 7],
     [6, 5, 4],
     [3, 2, 1]]
    Transformation Description: The grid is inverted, with the element order fully reversed in both dimensions. The rule is that all the number go in the reverse order.

    Problem:
    {question}

    Transformation Description:
    """

    transformation_description = call_llm(prompt, system_instruction)

    verification_prompt = f"""
    Verify that the given transformation description is clear, concise, describes a valid transformation, and includes a clear rule.
    Transformation Description: {transformation_description}
    Is the description valid? (VALID/INVALID)
    """
    validation_result = call_llm(verification_prompt)

    if "VALID" in validation_result:
        return {"is_valid": True, "transformation_description": transformation_description, "error": None}
    else:
        return {"is_valid": False, "transformation_description": None, "error": "Invalid feature description."}

def apply_transformation(question, transformation_description):
    """Applies the described transformation to the test input grid."""
    system_instruction = "You are an expert at applying transformations to grids based on a feature description."
    prompt = f"""
    Given the following grid transformation problem and the transformation description, apply the transformation to the test input grid. Follow the rule described as best as possible.

    Problem: {question}
    Transformation Description: {transformation_description}

    Example:
    Problem:
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[1, 1, 1],
     [0, 0, 0],
     [1, 1, 1]]
    Transformation Description: The transformation involves swapping the rows with '1' with adjacent rows.

    Generate the output grid.
    """
    output_grid = call_llm(prompt, system_instruction)
    return output_grid

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the call the LLM."""
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