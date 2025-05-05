import os
import re

def main(question):
    """
    Transform grids based on examples using an LLM with improved error handling, detailed logging,
    and multi-example prompting.
    """

    # Preprocess question to extract training examples and test input
    training_examples, test_input = preprocess_question(question)

    # Generate transformation rule using LLM
    transformation_rule = generate_transformation_rule(training_examples)

    # Apply transformation rule to the test input grid
    transformed_grid = apply_transformation_rule(test_input, transformation_rule)

    # Post-process the grid string to ensure correct formatting
    final_output = post_process_grid(transformed_grid)

    return final_output

def preprocess_question(question):
    """Extract training examples and test input from the question string using regex with better reliability."""
    # Updated regex to capture training examples and test input with more flexibility
    training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
    test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform the test input", question, re.DOTALL)

    training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
    test_input = test_input_match.group(1).strip() if test_input_match else ""

    return training_examples, test_input

def generate_transformation_rule(training_examples):
    """Generate a transformation rule from training examples using the LLM with multi-example prompting."""
    prompt = f"""
    You are an expert in identifying grid transformation rules. Given the following training examples,
    generate a concise transformation rule that accurately describes the pattern.

    Example 1:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[0, 1], [1, 0]]
    Rule: Mirror the grid along the diagonal.

    Example 2:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 1], [4, 3]]
    Rule: Swap the first and second elements in each row.

    Example 3:
    Input Grid: [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    Output Grid: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Rule: Mirror the grid along both diagonals.

    Training Examples:
    {training_examples}

    Transformation Rule:
    """
    # LLM call to generate the transformation rule
    transformation_rule = call_llm(prompt, system_instruction="You are a transformation rule generator.")

    return transformation_rule

def apply_transformation_rule(test_input, transformation_rule):
    """Apply the transformation rule to the test input grid using the LLM."""
    prompt = f"""
    You are an expert in applying grid transformation rules. Given the following test input grid
    and transformation rule, apply the rule to the grid and return the transformed grid.

    Example:
    Test Input Grid: [[1, 2], [3, 4]]
    Transformation Rule: Swap the first and second elements in each row.
    Transformed Grid: [[2, 1], [4, 3]]

    Test Input Grid:
    {test_input}
    Transformation Rule:
    {transformation_rule}

    Transformed Grid:
    """
    # LLM call to apply the transformation rule
    transformed_grid = call_llm(prompt, system_instruction="You are a grid transformation expert.")

    return transformed_grid

def post_process_grid(grid_string):
    """Post-process the grid string to ensure correct formatting with improved robustness."""
    # Remove any leading/trailing whitespace
    grid_string = grid_string.strip()
    # Remove any extra square brackets
    grid_string = grid_string.replace(' ', '')

    # Ensure that the result starts with '[[' and ends with ']]'
    if not grid_string.startswith('[['):
        grid_string = '[[' + grid_string
    if not grid_string.endswith(']]'):
        grid_string = grid_string + ']]'

    return grid_string

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
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