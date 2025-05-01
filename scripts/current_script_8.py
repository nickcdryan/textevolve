import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition and explicit rule extraction."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by first extracting the transformation rule and then applying it with verification."""

    system_instruction = "You are an expert at identifying grid transformation patterns from examples and applying them to new grids. You first EXPLAIN the rule before applying it, then apply the rule and verify the output."
    
    # STEP 1: Extract the transformation rule with embedded examples
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

    Example 3:
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

    # STEP 2: Apply the extracted rule to the test input with embedded examples
    application_prompt = f"""
    You have extracted this transformation rule:
    {extracted_rule}

    Now, apply this rule to the following test input grid:
    {problem_text}

    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.

    Example Application:
    Extracted Rule: Each element is doubled.
    Input Grid: [[1, 2], [3, 4]]
    Transformed Grid: [[2, 4], [6, 8]]
    """
    
    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(application_prompt, system_instruction)

            # STEP 3: Verification - check if the output looks like a grid and is plausibly correct
            verification_prompt = f"""
            You are validating a transformed grid against the original input and the extracted transformation rule.

            Extracted Rule: {extracted_rule}
            Original Grid: {problem_text}
            Transformed Grid: {transformed_grid_text}

            Based on the extracted rule, is the transformed grid a plausible result of applying the rule to the original grid? 
            Respond with 'YES' if it is plausible, and 'NO' if it is not. If the output is not a grid, respond with 'INVALID'.
            """
            verification_result = call_llm(verification_prompt, system_instruction)
            if "YES" in verification_result and "[" in transformed_grid_text and "]" in transformed_grid_text:
                return transformed_grid_text #If conditions are met the transformed grid is returned

            else:
                print(f"Attempt {attempt+1} failed verification: {verification_result}. Retrying...")

        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # Fallback approach if all attempts fail
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