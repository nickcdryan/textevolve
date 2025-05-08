import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition and explicit rule extraction."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by first extracting the transformation rule and then applying it."""

    system_instruction = "You are an expert at identifying grid transformation patterns from examples and applying them to new grids. You first EXPLAIN the rule before applying it."
    
    # STEP 1: Extract the transformation rule with an example
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
    print(f"Extracted Rule: {extracted_rule}") #Diagnostic statement

    # STEP 2: Verify the extracted rule's quality (is it understandable, not just gibberish)
    rule_verification_prompt = f"""
    You extracted this rule: {extracted_rule}
    Is the extracted rule understandable in plain English? Does it describe the transformation in a clear way, or is it nonsensical?
    Answer "Yes" or "No"
    """
    rule_is_valid = call_llm(rule_verification_prompt, system_instruction).startswith("Yes")
    print(f"Rule Valid: {rule_is_valid}") #Diagnostic statement

    if not rule_is_valid:
        print("Extracted rule is not valid, using fallback.")
        return "[[0,0,0],[0,0,0],[0,0,0]]" # Return default

    # STEP 3: Apply the extracted rule to the test input - add a well formatted example for more consistent output
    application_prompt = f"""
    You have extracted this transformation rule:
    {extracted_rule}

    Example:
    Input Grid:
    [[1, 2], [3, 4]]
    Extracted Rule: Each number is doubled
    Transformed Grid:
    [[2, 4], [6, 8]]

    Now, apply this rule to the following test input grid:
    {problem_text}

    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.
    """
    
    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(application_prompt, system_instruction)
            print(f"Transformed Grid Text: {transformed_grid_text}") #Diagnostic statement
            # Basic validation - check if it looks like a grid
            if "[" in transformed_grid_text and "]" in transformed_grid_text:
                return transformed_grid_text
            else:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
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