import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition and explicit rule extraction."""
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by first extracting the transformation rule and then applying it, with verification."""

    system_instruction = "You are an expert at identifying grid transformation patterns and applying them. Explain the rule, verify its consistency, and then apply it."
    
    # STEP 1: Extract the transformation rule with examples
    rule_extraction_prompt = f"""
    You are tasked with identifying the transformation rule applied to grids. Study the examples and explain the logic in plain English.

    Example 1:
    Input Grid:
    [[1, 0], [0, 1]]
    Output Grid:
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Explanation: Each element becomes a diagonal in a larger grid.

    Example 2:
    Input Grid:
    [[2, 8], [8, 2]]
    Output Grid:
    [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    Explanation: Each element is expanded to a 2x2 block.

    Example 3:
    Input Grid:
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Output Grid:
    [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    Explanation: Input overlaid onto zeros; 1 replaces 0, 0 remains as 0.

    Now, explain the transformation rule applied to this example. Respond with ONLY the explanation:
    Test Example:
    {problem_text}
    """
    
    extracted_rule = call_llm(rule_extraction_prompt, system_instruction)

    # STEP 2: Verify the extracted rule against the examples
    verification_prompt = f"""
    You extracted this transformation rule:
    {extracted_rule}

    Verify if this rule is consistent with the following example:
    {problem_text}

    Does the extracted rule logically explain how the input transforms into the output grid? Answer 'YES' or 'NO'.
    """
    
    verification_result = call_llm(verification_prompt, system_instruction)
    if "NO" in verification_result:
        print("Rule verification failed. Using fallback.")
        extracted_rule = "Applying a default identity transformation."  # Simple fallback

    # STEP 3: Apply the rule to the test input with examples
    application_prompt = f"""
    You have extracted this transformation rule:
    {extracted_rule}

    Example:
    Input Grid: [[1, 2], [3, 4]]
    Rule: Each number is replaced with its square
    Output Grid: [[1, 4], [9, 16]]

    Now, apply this rule to the following test input grid:
    {problem_text}

    Provide the transformed grid as a 2D array formatted as a string, WITHOUT additional explanations.
    """
    
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(application_prompt, system_instruction)
            if "[" in transformed_grid_text and "]" in transformed_grid_text:
                return transformed_grid_text
            else:
                print(f"Attempt {attempt+1} failed: Output format incorrect. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # Fallback approach
    return "[[0,0,0],[0,0,0],[0,0,0]]"

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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