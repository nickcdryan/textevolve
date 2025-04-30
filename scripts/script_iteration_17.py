import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks by using a "Transformation Suggestion and Verification" approach.

    Hypothesis: By providing several options that are likely transformations to the LLM, along with training data, the LLM is more likely to reason the steps and identify correct solutions.
    This approach reduces reliance on freeform reasoning and focuses on structured selection.
    """
    try:
        # 1. Extract relevant grid data.
        extracted_data = extract_data(question)
        if "Error" in extracted_data:
            return f"Data extraction error: {extracted_data}"

        # 2. Suggest likely transformations and verify their correctness.
        transformed_grid = suggest_and_verify_transformation(extracted_data)
        if "Error" in transformed_grid:
            return f"Transformation suggestion and verification error: {transformed_grid}"

        return transformed_grid

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_data(question):
    """Extracts training and test data from the problem question."""
    system_instruction = "You are an expert at extracting structured data from grid transformation problems."
    prompt = f"""
    Extract the training examples and test input from the question. Format the output as a dictionary-like string.

    Example:
    Question: Grid Transformation Task. Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}]. Test Input: [[5, 6], [7, 8]]
    Extracted Data: {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}

    Question: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def suggest_and_verify_transformation(extracted_data):
    """Suggests likely transformations and verifies their correctness using the training examples."""
    system_instruction = "You are an expert at identifying and verifying grid transformations."
    prompt = f"""
    Given the extracted data, suggest several likely transformations (e.g., reflection, rotation, arithmetic operation) and verify their correctness using the training examples.
    Choose the *best* transformation that leads to the transformation of the grid into what is shown by the test cases.
    Ensure you are only choosing ONE answer and ensure that answer transforms the INPUT grid to the OUTPUT grid of the training cases.
    The answer that you provide must be a well formatted grid.

    Example:
    Extracted Data: {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Likely Transformations and Verification:
    1. Reflection along both diagonals: Input [[1, 2], [3, 4]] becomes [[4, 3], [2, 1]]. This matches the training example.
    2. Row reversal: Input [[1, 2], [3, 4]] becomes [[3, 4], [1, 2]]. This does not match the training example.
    Transformed Grid: [[8, 7], [6, 5]]

    Extracted Data: {extracted_data}
    Likely Transformations and Verification:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error suggesting and verifying transformation: {str(e)}"

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