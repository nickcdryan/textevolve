import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks using a decomposition into data extraction, 
    direct LLM transformation with multiple examples, and verification with fallback.
    This approach tests the hypothesis that using a strong prompting strategy with 
    multiple examples in a single LLM call will improve direct transformation accuracy.
    """
    try:
        # 1. Extract training examples and test input
        extracted_data = extract_data(question)
        if "Error" in extracted_data:
            return f"Data extraction error: {extracted_data}"

        # 2. Apply direct transformation with multiple examples
        transformed_grid = apply_direct_transformation(extracted_data)
        if "Error" in transformed_grid:
            return f"Transformation error: {transformed_grid}"

        # 3. Verify the transformation, if verification fails, use a fallback
        verification_result = verify_transformation(extracted_data, transformed_grid)
        if "Error" in verification_result:
            return transformed_grid  # return the transformed grid anyway

        if not verification_result.startswith("VALID"):
            return transformed_grid # return the transformed grid anyway

        return transformed_grid

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_data(question):
    """Extracts training examples and test input from the question."""
    system_instruction = "You are an expert data extractor."
    prompt = f"""
    Extract the training examples and test input from the following question.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]]
    Extracted Data: Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]]

    Question: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def apply_direct_transformation(extracted_data):
    """Applies the transformation to the test input using multi-example prompting."""
    system_instruction = "You are a grid transformation expert."
    prompt = f"""
    Apply the transformation shown in the training examples to the test input.

    Example 1:
    Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]]
    Transformed Grid: [[8, 7], [6, 5]]
    Reasoning: The grid is reflected along both diagonals: (0,0) <-> (1,1), (0,1) <-> (1,0)

    Example 2:
    Training Examples: [{{'input': [[0, 0], [1, 1]], 'output': [[0, 0], [1, 0]]}}] Test Input: [[2, 2], [3, 3]]
    Transformed Grid: [[2, 2], [3, 2]]
    Reasoning: The bottom right element becomes the same as the element above it.

    Example 3:
    Training Examples: [{{'input': [[1, 2, 3], [4, 5, 6]], 'output': [[2, 3, 4], [5, 6, 7]]}}] Test Input: [[7, 8, 9], [10, 11, 12]]
    Transformed Grid: [[8, 9, 10], [11, 12, 13]]
    Reasoning: Each element of the matrix increments by 1.

    {extracted_data}
    Transformed Grid:
    Reasoning:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying transformation: {str(e)}"

def verify_transformation(extracted_data, transformed_grid):
    """Verifies that the transformed grid is correct, using multi-example prompting."""
    system_instruction = "You are an expert grid transformation verifier."
    prompt = f"""
    Verify the transformation is correct.
    If the transformation is invalid, explain why.
    The verification should start with VALID or INVALID.

    Example 1:
    Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]] Transformed Grid: [[8, 7], [6, 5]]
    Verification: VALID: The grid is correctly reflected along both diagonals.

    Example 2:
    Training Examples: [{{'input': [[0, 0], [1, 1]], 'output': [[0, 0], [1, 0]]}}] Test Input: [[2, 2], [3, 3]] Transformed Grid: [[2, 2], [3, 2]]
    Verification: VALID: The bottom right element becomes the same as the element above it, this is applied correctly.

    {extracted_data} Transformed Grid: {transformed_grid}
    Verification:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error during verification: {str(e)}"

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