import os
import re
import math
import json

def main(question):
    """
    Transforms a grid based on patterns in training examples.
    Uses LLM-driven pattern recognition and iterative refinement with DIFFERENCE GRID ANALYSIS.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem, max_attempts=3):
    """Solve grid transformation problems using pattern recognition, difference grid analysis, and verification."""
    system_instruction = "You are an expert at grid transformation tasks, skilled at pattern recognition and difference analysis."

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

    # Step 2: Analyze and infer the transformation rule using DIFFERENCE GRID ANALYSIS and enhanced examples.
    # NEW HYPOTHESIS: Difference grid analysis will improve pattern recognition.
    inference_prompt = f"""
    Analyze the provided training examples and infer the transformation rule.
    Use difference grid analysis to identify patterns by comparing Input and Output Grids.

    Example 1:
    Examples: Input Grid: [[1, 1, 1]] ... Output Grid: [[2, 2, 2]]
    Difference Grid: [[1, 1, 1]]
    Rule: Each element in the input grid is incremented by 1.

    Example 2:
    Examples: Input Grid: [[0, 1, 0]] ... Output Grid: [[0, 2, 0]]
    Difference Grid: [[0, 1, 0]]
    Rule: Each '1' in the input grid is replaced with '2', while '0' remains unchanged.

    Examples: {extracted_info}
    Rule:
    """
    transformation_rule = call_llm(inference_prompt, system_instruction)

    # Step 3: Apply the transformation rule to the test input
    transformation_prompt = f"""
    Apply the following transformation rule to the test input grid.

    Rule: {transformation_rule}
    Test Input Grid: {extracted_info}

    Example 1:
    Rule: Each element is doubled. Test Input Grid: [[1, 2], [3, 4]]
    Transformed Grid: [[2, 4], [6, 8]]

    Transformed Grid:
    """
    transformed_grid = call_llm(transformation_prompt, system_instruction)

    # Step 4: Verify the transformed grid and correct if needed
    verification_prompt = f"""
    Verify that the transformed grid follows the transformation rule.

    Rule: {transformation_rule}
    Test Input Grid: {extracted_info}
    Transformed Grid: {transformed_grid}

    Example:
    Rule: double each number. Input: [[1,2],[3,4]]. Output: [[2,4],[6,8]]. Verification: CORRECT

    Verification:
    """
    verification_result = call_llm(verification_prompt, system_instruction)

    if "INCORRECT" in verification_result:
        # Attempt to correct the transformation (simple error correction)
        correction_prompt = f"""
        Correct the transformed grid based on the verification feedback.

        Rule: {transformation_rule}
        Test Input Grid: {extracted_info}
        Transformed Grid: {transformed_grid}
        Verification Feedback: {verification_result}

        Corrected Grid:
        """
        corrected_grid = call_llm(correction_prompt, system_instruction)
        return corrected_grid
    else:
        return transformed_grid

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