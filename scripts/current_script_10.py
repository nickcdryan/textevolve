import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition and explicit rule extraction with verification."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by first extracting the transformation rule, then applying it, and finally verifying the result."""

    system_instruction = "You are an expert at identifying grid transformation patterns from examples and applying them to new grids. You first EXPLAIN the rule before applying it and verify the results."
    
    # STEP 1: Extract the transformation rule with example
    rule_extraction_prompt = f"""
    You are tasked with identifying the transformation rule applied to grids. Study the examples carefully and explain the transformation logic in plain English.

    Example:
    Input Grid:
    [[1, 0], [0, 1]]
    Output Grid:
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Explanation: Each element in the input grid becomes a diagonal in a larger grid.

    Now, explain the transformation rule applied to this example. Respond with ONLY the explanation:
    Test Example:
    {problem_text}
    """
    
    # Attempt to extract the rule
    extracted_rule = call_llm(rule_extraction_prompt, system_instruction)

    # STEP 2: Apply the extracted rule to the test input with example
    application_prompt = f"""
    You have extracted this transformation rule:
    {extracted_rule}

    Example:
    Extracted Rule: Each element is expanded to a 2x2 block with the element's value.
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Now, apply this rule to the following test input grid:
    {problem_text}

    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.
    """
    
    # Attempt to generate the transformed grid
    transformed_grid_text = call_llm(application_prompt, system_instruction)

    # STEP 3: Verify the transformed grid with example
    verification_prompt = f"""
    You extracted the following transformation rule: {extracted_rule}
    And you applied it to this input: {problem_text}
    To produce this transformed grid: {transformed_grid_text}

    Example:
    Extracted Rule: Each element in the input grid becomes a diagonal in a larger grid.
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Is the above a valid transformation? Yes.

    Now, is the transformed grid valid according to your rule? Respond with Yes or No.
    """

    # Attempt to verify the generated grid
    is_valid_transformation = call_llm(verification_prompt, system_instruction)

    if "Yes" in is_valid_transformation:
        return transformed_grid_text
    else:
        return "[[0,0,0],[0,0,0],[0,0,0]]"


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