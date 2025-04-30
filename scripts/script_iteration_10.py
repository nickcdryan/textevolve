import os
import re
import math

# New approach: Decompose into rule extraction, matrix transformation with specific dimensions, and result validation.
# This approach tests the hypothesis that explicitly providing example dimensions will constrain and improve the result.
# Each LLM interaction will include examples.
def main(question):
    """
    Orchestrates the grid transformation process, extracting rules, applying them, and validating results.
    """
    try:
        # 1. Extract transformation rule and matrix dimensions using LLM with example.
        rule_and_dimensions = extract_rule_and_dimensions(question)
        if "Error" in rule_and_dimensions:
            return f"Rule and dimensions extraction error: {rule_and_dimensions}"

        # 2. Apply transformation rule to the test input using LLM.
        transformed_grid = apply_transformation_with_dimensions(question, rule_and_dimensions)
        if "Error" in transformed_grid:
            return f"Transformation application error: {transformed_grid}"

        # 3. Validate the transformed grid using LLM.
        validation_result = validate_transformation(question, transformed_grid)
        if "Error" in validation_result:
            return f"Transformation validation error: {validation_result}"

        return transformed_grid

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_rule_and_dimensions(question):
    """
    Extracts the transformation rule and matrix dimensions from the question using LLM.
    """
    system_instruction = "You are an expert at extracting transformation rules and matrix dimensions."
    prompt = f"""
    Extract the transformation rule and the input and output matrix dimensions from the question.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]]
    Extracted Rule and Dimensions:
    {{
        "rule": "Reflect the grid along both diagonals. The element at (i, j) is swapped with the element at (N-1-i, N-1-j).",
        "input_rows": 2,
        "input_cols": 2,
        "output_rows": 2,
        "output_cols": 2
    }}

    Question: {question}
    Extracted Rule and Dimensions:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting rule and dimensions: {str(e)}"

def apply_transformation_with_dimensions(question, rule_and_dimensions):
    """
    Applies the transformation rule to the test input with the matrix dimensions to generate the transformed grid.
    """
    system_instruction = "You are an expert at applying transformation rules to grid data."
    prompt = f"""
    Apply the following transformation rule to the test input, considering the provided matrix dimensions.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]]
    Rule and Dimensions:
    {{
        "rule": "Reflect the grid along both diagonals. The element at (i, j) is swapped with the element at (N-1-i, N-1-j).",
        "input_rows": 2,
        "input_cols": 2,
        "output_rows": 2,
        "output_cols": 2
    }}
    Transformed Grid: [[8, 7], [6, 5]]

    Question: {question}
    Rule and Dimensions: {rule_and_dimensions}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying transformation: {str(e)}"

def validate_transformation(question, transformed_grid):
    """
    Validates the transformed grid against the question and returns whether the result is correct.
    """
    system_instruction = "You are an expert at validating grid transformations."
    prompt = f"""
    Validate the following transformation result against the question.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]]
    Transformed Grid: [[8, 7], [6, 5]]
    Validation: Correct

    Question: {question}
    Transformed Grid: {transformed_grid}
    Validation:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error validating transformation: {str(e)}"

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