import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks by decomposing the problem into three distinct steps: 
    data extraction and validation, rule inference with example-based priming, and 
    transformation with iterative verification. 

    This approach tests the hypothesis that example-based priming combined with iterative 
    verification will improve rule inference and application.
    """
    try:
        # 1. Extract data and validate its integrity
        extracted_data = extract_and_validate_data(question)
        if "Error" in extracted_data:
            return f"Data extraction error: {extracted_data}"

        # 2. Infer transformation rule with example-based priming
        inferred_rule = infer_transformation_rule(extracted_data)
        if "Error" in inferred_rule:
            return f"Rule inference error: {inferred_rule}"

        # 3. Apply the rule with iterative verification
        transformed_grid = apply_rule_with_verification(extracted_data, inferred_rule)
        if "Error" in transformed_grid:
            return f"Transformation error: {transformed_grid}"

        return transformed_grid

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_and_validate_data(question):
    """
    Extracts training examples and test input from the question, validating 
    the extracted data to ensure integrity.
    """
    system_instruction = "You are an expert at extracting and validating data for grid transformation problems."
    prompt = f"""
    Extract training examples and test input from the question. Return the extracted data as a plain text string.
    If any of the data is missing, indicate the type of missing data. Ensure that you extract the training examples and test input
    in the same code block (use triple single quotes to enclose).
    
    Example:
    Question: Grid Transformation Task
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Test Input: [[5, 6], [7, 8]]
    Extracted Data:
    '''
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Test Input: [[5, 6], [7, 8]]
    '''
    
    Question: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def infer_transformation_rule(extracted_data):
    """
    Infers the transformation rule using example-based priming.
    """
    system_instruction = "You are an expert at inferring grid transformation rules from examples."
    prompt = f"""
    Infer the transformation rule from the following training examples and provide a concise description of the rule and how to use it.

    Example:
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Inferred Rule:
    'The transformation reflects the grid along both diagonals. To apply, swap element at (i, j) with element at (N-1-i, N-1-j).'
    
    Extracted Data: {extracted_data}
    Inferred Rule:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error inferring rule: {str(e)}"

def apply_rule_with_verification(extracted_data, inferred_rule, max_attempts=3):
    """
    Applies the inferred rule to the test input and iteratively verifies the result.
    """
    system_instruction = "You are an expert at applying grid transformation rules with iterative verification."
    prompt = f"""
    Apply the following transformation rule to the test input, taken from the extracted data. 
    Verify the result at each stage and return the final transformed grid.
    Example:
    Refined Rules: 'The grid is reflected along both diagonals. To apply, swap element at (i, j) with element at (N-1-i, N-1-j).'
    Extracted Data:
    '''
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Test Input: [[5, 6], [7, 8]]
    '''
    Transformed Grid: "[[8, 7], [6, 5]]"
    
    Refined Rules: {inferred_rule}
    Extracted Data: {extracted_data}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying transformation: {str(e)}"

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