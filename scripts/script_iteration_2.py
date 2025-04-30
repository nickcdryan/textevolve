import os
import re

def main(question):
    """
    Solve grid transformation tasks using a multi-stage LLM reasoning approach.

    This approach uses a "Rule Generation and Application" strategy, where the LLM first attempts to generate the explicit transformation rule based on examples, and then applies the rule.
    This is a fundamentally different approach that focuses on rule explainability and validation, addressing previous issues.
    """
    try:
        # Step 1: Extract training examples and test input from the question
        training_examples, test_input = extract_input_data(question)

        # Step 2: Generate transformation rule
        transformation_rule = generate_transformation_rule(training_examples)

        # Step 3: Apply transformation rule to test input
        transformed_grid = apply_transformation_rule(test_input, transformation_rule)

        # Step 4: Validate the transformation result
        validation_result = validate_transformation(training_examples, test_input, transformed_grid)

        return transformed_grid if validation_result == "VALID" else "INVALID TRANSFORMATION"

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_input_data(question):
    """Extract training examples and test input from the question string."""
    training_examples_match = re.search(r"Training Examples:\n(.*?)\n\nTest Input:", question, re.DOTALL)
    test_input_match = re.search(r"Test Input:\n(.*?)\n", question, re.DOTALL)

    if not training_examples_match or not test_input_match:
        raise ValueError("Could not extract training examples or test input.")

    training_examples = training_examples_match.group(1).strip()
    test_input = test_input_match.group(1).strip()

    return training_examples, test_input

def generate_transformation_rule(training_examples):
    """Generate transformation rule from training examples."""
    system_instruction = "You are an expert at generating transformation rules from grid examples."
    prompt = f"""
    Analyze the training examples and generate a transformation rule that explains how the input grid is transformed into the output grid.

    Example 1:
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Transformation Rule: The transformation reflects the grid along both diagonals.

    Example 2:
    Training Examples:
    [
        {{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}}
    ]
    Transformation Rule: The transformation swaps rows and columns of the input grid (transpose).

    Training Examples:
    {training_examples}
    Transformation Rule:
    """

    return call_llm(prompt, system_instruction)

def apply_transformation_rule(test_input, transformation_rule):
    """Apply transformation rule to test input."""
    system_instruction = "You are an expert at applying transformation rules to grids."
    prompt = f"""
    Apply the transformation rule to the test input. Provide the transformed grid.

    Example 1:
    Transformation Rule: The transformation transposes the input grid.
    Test Input: [[1, 2], [3, 4]]
    Transformed Grid: [[1, 3], [2, 4]]

    Transformation Rule: {transformation_rule}
    Test Input: {test_input}
    Transformed Grid:
    """
    return call_llm(prompt, system_instruction)

def validate_transformation(training_examples, test_input, transformed_grid):
    """Validate the transformation by checking if the rule would be applicable to previous training examples"""
    system_instruction = "You are an expert at validating grid transformations."
    prompt = f"""
    Given training examples, test input and its transformation, validate if the applied transformation is correct.
    Answer with "VALID" or "INVALID".

    Example 1:
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Test Input: [[5, 6], [7, 8]]
    Transformed Grid: [[8, 7], [6, 5]]
    Validation: VALID

    Training Examples: {training_examples}
    Test Input: {test_input}
    Transformed Grid: {transformed_grid}
    Validation:
    """
    return call_llm(prompt, system_instruction)

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