import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Component-Wise Analysis and Rule Application" approach.
# This approach analyzes the grid by separating components such as rows, columns, and diagonals, extracts rules for each component,
# and then applies them separately before combining them. A verifier will deduce if the changes are helpful.

def main(question):
    """Transforms a grid by analyzing and applying rules component-wise."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Analyze and apply rules to rows
        transformed_rows = transform_rows(test_input, training_examples)

        # 3. Analyze and apply rules to columns
        transformed_cols = transform_cols(transformed_rows, training_examples)

        # 4. Combine components and return transformed grid
        transformed_grid = transformed_cols

        return transformed_grid

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def preprocess_question(question):
    """Extract training examples and test input from the question string."""
    try:
        training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)

        training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
        test_input = test_input_match.group(1).strip() if test_input_match else ""

        return training_examples, test_input
    except Exception as e:
        return "", ""

def transform_rows(test_input, training_examples):
    """Analyze and transform rows based on training examples."""
    system_instruction = "You are an expert in analyzing and transforming rows of a grid."
    prompt = f"""
    You are an expert in analyzing and transforming rows of a grid. Given a test input and training examples,
    analyze each row independently and apply transformations based on the observed patterns.

    Example:
    Training Examples:
    Input Grid: [[1, 2, 3], [4, 5, 6]]
    Output Grid: [[2, 3, 4], [5, 6, 7]]
    Test Input: [[7, 8, 9], [10, 11, 12]]
    Transformed Rows: [[8, 9, 10], [11, 12, 13]]

    Example 2:
    Training Examples:
    Input Grid: [[0, 1, 0], [1, 0, 1]]
    Output Grid: [[1, 0, 1], [0, 1, 0]]
    Test Input: [[0, 1, 0], [1, 0, 1]]
    Transformed Rows: [[1, 0, 1], [0, 1, 0]]

    Training Examples:
    {training_examples}
    Test Input:
    {test_input}
    Transformed Rows:
    """
    transformed_rows = call_llm(prompt, system_instruction)
    return transformed_rows

def transform_cols(transformed_rows, training_examples):
    """Analyze and transform columns based on training examples."""
    system_instruction = "You are an expert in analyzing and transforming columns of a grid."
    prompt = f"""
    You are an expert in analyzing and transforming columns of a grid. Given transformed rows and training examples,
    analyze each column independently and apply transformations based on the observed patterns.

    Example:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Transformed Rows: [[7, 8], [9, 10]]
    Transformed Columns: [[8, 9], [10, 11]]

    Example 2:
    Training Examples:
    Input Grid: [[0, 1], [1, 0]]
    Output Grid: [[1, 0], [0, 1]]
    Transformed Rows: [[0, 1], [1, 0]]
    Transformed Columns: [[1, 0], [0, 1]]

    Training Examples:
    {training_examples}
    Transformed Rows:
    {transformed_rows}
    Transformed Columns:
    """
    transformed_cols = call_llm(prompt, system_instruction)
    return transformed_cols

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template."""
    try:
        from google import genai
        from google.genai import types
        import os

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