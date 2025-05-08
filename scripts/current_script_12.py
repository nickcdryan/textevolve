import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition and explicit rule extraction."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by first extracting the transformation rule and then applying it with verification."""

    system_instruction = "You are an expert at identifying grid transformation patterns and applying them. You first explain, then transform, and then verify."
    
    # STEP 1: Extract the transformation rule with examples
    rule_extraction_prompt = f"""
    You are tasked with identifying the transformation rule applied to grids. Study the examples carefully and explain the transformation logic in plain English.

    Example 1:
    Input Grid:
    [[1, 0], [0, 1]]
    Output Grid:
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Explanation: Each element in the input grid becomes a diagonal in a larger grid.

    Example 2:
    Input Grid:
    [[2, 8], [8, 2]]
    Output Grid:
    [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Explanation: Each element is expanded to a 2x2 block with the element's value.

    Input Grid:
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Output Grid:
    [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    Explanation: The input grid is overlaid onto a grid of zeros; the value of 1 replaces 0; the values of 0 remain as 0.

    Now, explain the transformation rule applied to this example. Respond with ONLY the explanation:
    Test Example:
    {problem_text}
    """
    
    # Attempt to extract the rule
    extracted_rule = call_llm(rule_extraction_prompt, system_instruction)

    # STEP 2: Apply the extracted rule to the test input with examples
    application_prompt = f"""
    You have extracted this transformation rule:
    {extracted_rule}

    Now, apply this rule to the following test input grid:
    {problem_text}

    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.

    Example:
    Extracted rule: Each element is doubled.
    Input Grid: [[1, 2], [3, 4]]
    Transformed Grid: [[2, 4], [6, 8]]
    """
    
    # Attempt to generate the transformed grid
    transformed_grid_text = call_llm(application_prompt, system_instruction)

    # STEP 3: Verification Step
    verification_prompt = f"""
    You extracted the following rule and then transformed a grid:

    Rule: {extracted_rule}
    Original Grid: {problem_text}
    Transformed Grid: {transformed_grid_text}

    Does the transformed grid follow the extracted rule? Answer 'Yes' or 'No'.
    """

    verification_result = call_llm(verification_prompt, system_instruction)

    if "Yes" in verification_result:
        return transformed_grid_text
    else:
        return "[[0,0,0],[0,0,0],[0,0,0]]" # Fallback if the transformation doesn't verify

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