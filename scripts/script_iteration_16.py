import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks by using a 'Chain of Transformation Descriptions' approach.

    The hypothesis is that by iteratively refining a *description* of the transformation, rather
    than directly generating a transformed grid, the LLM can better capture complex rules.
    The LLM describes the transformation pattern, then applies the description in a separate step, and
    verifies its success.
    """
    try:
        # 1. Extract relevant grid data and perform a verification step.
        extracted_data = extract_data(question)
        if "Error" in extracted_data:
            return f"Data extraction error: {extracted_data}"

        # 2. Describe the transformation pattern
        transformation_description = describe_transformation(extracted_data)
        if "Error" in transformation_description:
            return f"Transformation description error: {transformation_description}"

        # 3. Apply the transformation based on the description.
        transformed_grid = apply_transformation(extracted_data, transformation_description)
        if "Error" in transformed_grid:
            return f"Transformation application error: {transformed_grid}"

        return transformed_grid

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_data(question):
    """Extracts training and test data from the problem question using an LLM, and performs a validation"""
    system_instruction = "You are an expert at extracting structured data, especially from grid transformation problems."
    prompt = f"""
    Extract the training examples and test input from the question. Format the output as a dictionary-like string.
    Then, perform a validation to ensure that it contains both the training examples and the test input, and does not contain any faulty or null data.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]]
    Extracted Data: {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Validation: The extracted data contains the training examples and the test input, and is correctly formatted. VALID

    Question: {question}
    Extracted Data:
    """
    try:
        extracted_data = call_llm(prompt, system_instruction)
        # Add verification for extraction
        if "training_examples" not in extracted_data or "test_input" not in extracted_data:
            return "Error: Missing training examples or test input"
        return extracted_data
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def describe_transformation(extracted_data):
    """Describes the transformation pattern in natural language based on the extracted data."""
    system_instruction = "You are an expert at describing transformation patterns in grid data using natural language."
    prompt = f"""
    Describe the transformation pattern in the provided training examples using natural language.
    Focus on the 'before' and 'after' states. The description should be detailed enough to reproduce the transformation.

    Example:
    Training Examples: {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Transformation Description: The grid is reflected along both diagonals. Element (0,0) is swapped with (1,1) and element (0,1) is swapped with (1,0).

    Training Examples: {extracted_data}
    Transformation Description:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error describing transformation: {str(e)}"

def apply_transformation(extracted_data, transformation_description):
    """Applies the described transformation to the test input grid and generates the transformed grid."""
    system_instruction = "You are an expert at applying transformation rules described in natural language to grid data."
    prompt = f"""
    Apply the described transformation to the test input grid and generate the transformed grid.

    Example:
    Transformation Description: The grid is reflected along both diagonals. Element (0,0) is swapped with (1,1) and element (0,1) is swapped with (1,0).
    Test Input: {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Transformed Grid: [[8, 7], [6, 5]]

    Transformation Description: {transformation_description}
    Test Input: {extracted_data}
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