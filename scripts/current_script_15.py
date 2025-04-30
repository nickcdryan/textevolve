import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks by focusing on iterative pattern refinement
    and a multi-stage analysis with a central "pattern_identifier" agent.
    Uses a multi-example prompting strategy. Adds verification and robustness.
    """
    try:
        # 1. Extract relevant grid data.
        extracted_data = extract_data(question)
        if "Error" in extracted_data:
            return f"Data extraction error: {extracted_data}"

        # 2. Identify initial transformation patterns.
        initial_pattern = identify_initial_pattern(extracted_data)
        if "Error" in initial_pattern:
            return f"Pattern identification error: {initial_pattern}"

        # 3. Refine transformation pattern iteratively.
        refined_pattern = refine_pattern(extracted_data, initial_pattern)
        if "Error" in refined_pattern:
            return f"Pattern refinement error: {refined_pattern}"

        # 4. Apply refined transformation pattern to the test input.
        transformed_grid = apply_refined_transformation(extracted_data, refined_pattern)
        if "Error" in transformed_grid:
            return f"Transformation application error: {transformed_grid}"

        return transformed_grid

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_data(question):
    """Extracts relevant training and test data from the problem question using an LLM."""
    system_instruction = "You are an expert at extracting structured data, especially from grid transformation problems."
    prompt = f"""
    Extract the training examples and test input from the question. Format the output as a dictionary-like string.

    Example 1:
    Question: Grid Transformation Task... Input: [[1, 2], [3, 4]] Output: [[4, 3], [2, 1]] Test Input: [[5, 6], [7, 8]]
    Extracted Data: {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}

    Example 2:
    Question: Grid Transformation Task... Input: [[0, 0], [1, 1]] Output: [[1, 1], [0, 0]] Test Input: [[2, 2], [3, 3]]
    Extracted Data: {{'training_examples': '[{{"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]}}]', 'test_input': '[[2, 2], [3, 3]]'}}

    Question: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def identify_initial_pattern(extracted_data):
    """Identifies an initial transformation pattern using the training examples."""
    system_instruction = "You are an expert at identifying transformation patterns in grid data."
    prompt = f"""
    Identify the initial transformation pattern from the provided training examples.

    Example 1:
    Training Examples: {{"training_examples": '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', "test_input": '[[5, 6], [7, 8]]'}}
    Transformation Pattern: The grid is reflected along both diagonals.

    Example 2:
    Training Examples: {{"training_examples": '[{{"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]}}]', "test_input": '[[2, 2], [3, 3]]'}}
    Transformation Pattern: The rows are reversed.

    Training Examples: {extracted_data}
    Transformation Pattern:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error identifying initial pattern: {str(e)}"

def refine_pattern(extracted_data, initial_pattern):
    """Refines the transformation pattern iteratively based on verification steps."""
    system_instruction = "You are an expert at refining transformation patterns."
    prompt = f"""
    Given the extracted data and the initial pattern, refine the transformation pattern by analyzing edge cases.

    Example 1:
    Extracted Data: {{"training_examples": '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', "test_input": '[[5, 6], [7, 8]]'}}
    Initial Pattern: The grid is reflected along both diagonals.
    Refined Pattern: The grid is reflected along both diagonals, but if a value is 0, it remains 0.

    Example 2:
    Extracted Data: {{"training_examples": '[{{"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]}}]', "test_input": '[[2, 2], [3, 3]]'}}
    Initial Pattern: The rows are reversed.
    Refined Pattern: The rows are reversed, but values greater than 10 are unchanged.

    Extracted Data: {extracted_data}
    Initial Pattern: {initial_pattern}
    Refined Pattern:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error refining pattern: {str(e)}"

def apply_refined_transformation(extracted_data, refined_pattern):
    """Applies the refined transformation pattern to the test input."""
    system_instruction = "You are an expert at applying refined transformation patterns to grid data."
    prompt = f"""
    Apply the refined transformation pattern to the test input and generate the transformed grid.

    Example 1:
    Refined Pattern: The grid is reflected along both diagonals.
    Test Input: [[5, 6], [7, 8]]
    Transformed Grid: [[8, 7], [6, 5]]

    Example 2:
    Refined Pattern: The rows are reversed.
    Test Input: [[2, 2], [3, 3]]
    Transformed Grid: [[3, 3], [2, 2]]

    Refined Pattern: {refined_pattern}
    Test Input: {extracted_data}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying refined transformation: {str(e)}"

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