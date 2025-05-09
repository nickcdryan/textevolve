import os
import re
import math

def main(question):
    """
    Transforms a grid based on patterns in training examples.
    Uses LLM-driven pattern recognition with a focus on localized transformations and multi-example prompts with similarity scoring.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem, max_attempts=3):
    """Solve grid transformation problems using pattern recognition and localized transformation analysis."""
    # Hypothesis: Focusing on localized transformations and using similar examples will improve pattern recognition.
    system_instruction = "You are an expert at grid transformation tasks, skilled at identifying localized patterns."

    # Step 1: Extract relevant information (training examples, test input)
    extraction_prompt = f"""
    Extract the training examples and the test input grid from the problem description.

    Example 1:
    Problem: Grid Transformation Task... Input Grid: [[1,2],[3,4]] ... Output Grid: [[5,6],[7,8]] ... TEST INPUT: [[9,10],[11,12]]
    Extracted: {{"examples": ["Input Grid: [[1,2],[3,4]] ... Output Grid: [[5,6],[7,8]]"], "test_input": "[[9,10],[11,12]]"}}

    Problem: {problem}
    Extracted:
    """
    extracted_info = call_llm(extraction_prompt, system_instruction)

    # Step 2: Analyze and infer the localized transformation rule with enhanced examples.
    localized_inference_prompt = f"""
    Analyze the provided training examples and infer the localized transformation rule.

    Example 1:
    Examples: Input Grid: [[1, 0], [0, 1]] ... Output Grid: [[2, 0], [0, 2]]
    Localized Rule: If a cell has value 1, transform it to 2.

    Example 2:
    Examples: Input Grid: [[0, 1, 0]] ... Output Grid: [[0, 2, 0]]
    Localized Rule: Change values of '1' to '2', but leave '0' unchanged.

    Examples: {extracted_info}
    Localized Rule:
    """
    localized_transformation_rule = call_llm(localized_inference_prompt, system_instruction)

    # Step 3: Apply the localized transformation rule to the test input
    localized_transformation_prompt = f"""
    Apply the following localized transformation rule to the test input grid.

    Rule: {localized_transformation_rule}
    Test Input Grid: {extracted_info}

    Example 1:
    Rule: Each element is doubled. Test Input Grid: [[1, 2], [3, 4]]
    Transformed Grid: [[2, 4], [6, 8]]

    Transformed Grid:
    """
    transformed_grid = call_llm(localized_transformation_prompt, system_instruction)

    # Verification: Check if the transformation follows the rule and data
    verification_prompt = f"""
    Verify that the transformed grid follows the localized transformation rule.

    Rule: {localized_transformation_rule}
    Test Input Grid: {extracted_info}
    Transformed Grid: {transformed_grid}

    Example:
    Rule: double each number. Input: [[1,2],[3,4]]. Output: [[2,4],[6,8]]. Verification: CORRECT

    Verification:
    """
    verification_result = call_llm(verification_prompt, system_instruction)

    #If the result is correct, return the transformed grid, otherwise say that it is unable to perform transformation
    if "INCORRECT" not in verification_result:
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