import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks with enhanced data extraction and multi-stage pattern refinement.

    This version improves data extraction robustness and provides more detailed pattern refinement
    using multiple examples in each LLM call.
    """
    try:
        # 1. Extract relevant grid data with better error handling and examples.
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
    """Extracts training and test data with example-based guidance for improved accuracy."""
    system_instruction = "You are an expert at extracting structured data from grid transformation problems."
    prompt = f"""
    Extract the training examples and test input from the question.
    Format the output as a dictionary-like string. Ensure training examples and the test input are well-formatted.

    Example 1:
    Question: Grid Transformation Task
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Test Input: [[5, 6], [7, 8]]
    Extracted Data:
    {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}

    Example 2:
    Question: Grid Transformation Task
    Training Examples:
    [
        {{"input": [[0, 1, 0], [1, 0, 1]], "output": [[1, 0, 1], [0, 1, 0]]}},
        {{"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}}
    ]
    Test Input: [[3, 0], [0, 3]]
    Extracted Data:
    {{'training_examples': '[{{"input": [[0, 1, 0], [1, 0, 1]], "output": [[1, 0, 1], [0, 1, 0]]}}, {{"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}}]', 'test_input': '[[3, 0], [0, 3]]'}}

    Question: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def identify_initial_pattern(extracted_data):
    """Identifies an initial transformation pattern with detailed example for pattern recognition."""
    system_instruction = "You are an expert at identifying transformation patterns in grid data."
    prompt = f"""
    Identify the initial transformation pattern from the provided training examples.

    Example 1:
    Training Examples:
    {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Transformation Pattern: The grid is reflected along both diagonals.

    Example 2:
    Training Examples:
    {{'training_examples': '[{{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}}, {{"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]}}]', 'test_input': '[[4, 5], [5, 4]]'}}
    Transformation Pattern: The grid is reflected along the main diagonal (top-left to bottom-right).

    Training Examples: {extracted_data}
    Transformation Pattern:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error identifying initial pattern: {str(e)}"

def refine_pattern(extracted_data, initial_pattern):
    """Refines the transformation pattern iteratively using edge case analysis and examples."""
    system_instruction = "You are an expert at refining transformation patterns, particularly in edge cases."
    prompt = f"""
    Given the extracted data and the initial pattern, refine the transformation pattern by analyzing edge cases.

    Example 1:
    Extracted Data:
    {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}
    Initial Pattern: The grid is reflected along both diagonals.
    Refined Pattern: The grid is reflected along both diagonals, with no change to any zero values.

    Example 2:
    Extracted Data:
    {{'training_examples': '[{{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}}, {{"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}}]', 'test_input': '[[3, 0], [0, 3]]'}}
    Initial Pattern: The grid is reflected along the main diagonal (top-left to bottom-right).
    Refined Pattern: The grid is reflected along the main diagonal (top-left to bottom-right). Zero values are maintained.

    Extracted Data: {extracted_data}
    Initial Pattern: {initial_pattern}
    Refined Pattern:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error refining pattern: {str(e)}"

def apply_refined_transformation(extracted_data, refined_pattern):
    """Applies the refined transformation pattern with direct example guidance."""
    system_instruction = "You are an expert at applying refined transformation patterns to grid data."
    prompt = f"""
    Apply the refined transformation pattern to the test input and generate the transformed grid.

    Example 1:
    Refined Pattern: The grid is reflected along both diagonals, with no change to any zero values.
    Test Input: [[5, 6], [7, 8]]
    Transformed Grid: [[8, 7], [6, 5]]

    Example 2:
    Refined Pattern: The grid is reflected along the main diagonal (top-left to bottom-right). Zero values are maintained.
    Test Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Transformed Grid: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

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