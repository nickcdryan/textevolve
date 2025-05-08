import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition and explicit rule extraction."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by first extracting the transformation rule and then applying it."""

    system_instruction = "You are an expert at identifying grid transformation patterns from examples and applying them to new grids. You first EXPLAIN the rule before applying it."
    
    # STEP 1: Extract the transformation rule
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

    # STEP 2: Apply the extracted rule to the test input
    application_prompt = f"""
    You have extracted this transformation rule:
    {extracted_rule}

    Now, apply this rule to the following test input grid:
    {problem_text}

    Example 1:
    Extracted Rule: Each element in the input grid becomes a diagonal in a larger grid.
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Extracted Rule: Each element is expanded to a 2x2 block with the element's value.
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    
    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.
    """
    
    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(application_prompt, system_instruction)
            # Basic validation - check if it looks like a grid
            if "[" in transformed_grid_text and "]" in transformed_grid_text:

                # Validate the format and content of the result
                validation_prompt = f"""
                You generated this transformed grid: {transformed_grid_text}
                Original Input:{problem_text}

                Is this valid 2D array? Are the values in the array valid numbers? Does the grid transformation reasonably align with the source input and examples provided.

                Respond with ONLY "VALID" or "INVALID".
                """
                validation_result = call_llm(validation_prompt)

                if "VALID" in validation_result:
                    return transformed_grid_text
                else:
                    print(f"Attempt {attempt+1} failed: Output invalid. Retrying...")
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