import os
import re

def main(question):
    """
    Transforms a grid based on training examples using a multi-stage LLM approach.
    Includes detailed error handling and focuses on robust output formatting.
    """
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Generate a transformation rule based on the training examples
        transformation_rule = generate_transformation_rule(training_examples)

        # 3. Apply the transformation rule to the test input
        transformed_grid = apply_transformation_rule(test_input, transformation_rule)

        # 4. Verify the output grid format
        if not verify_grid_format(transformed_grid):
            transformed_grid = post_process_grid(transformed_grid)
            if not verify_grid_format(transformed_grid):
                return "Error: Invalid grid format after post-processing."

        return transformed_grid

    except Exception as e:
        return f"Error: {str(e)}"

def preprocess_question(question):
    """Extract training examples and test input from the question using regex."""
    try:
        training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)

        training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
        test_input = test_input_match.group(1).strip() if test_input_match else ""

        return training_examples, test_input
    except Exception as e:
        return "", ""

def generate_transformation_rule(training_examples):
    """Generates a transformation rule from training examples."""
    system_instruction = "You are an expert rule generator, focusing on identifying transformation patterns in grids."
    prompt = f"""
    Analyze these training examples and generate a concise transformation rule that can be applied to new input grids.
    The rule should describe the pattern and how it should be applied.
    Example:
    Training Examples:
    Input Grid:
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    Output Grid:
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    Rule: Replace all values with 1.
    Input Grid:
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Output Grid:
    [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
    Rule: Reverse the elements in each row.

    Training Examples:
    {training_examples}

    Transformation Rule:
    """
    return call_llm(prompt, system_instruction)

def apply_transformation_rule(test_input, transformation_rule):
    """Applies the transformation rule to the test input and returns the transformed grid."""
    system_instruction = "You are an expert grid transformer, applying rules to input grids."
    prompt = f"""
    Apply this transformation rule to the test input and generate the transformed grid. Ensure the output is a string with proper double brackets.

    Transformation Rule:
    {transformation_rule}

    Test Input:
    {test_input}

    Example Output:
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

    Transformed Grid:
    """
    return call_llm(prompt, system_instruction)

def verify_grid_format(grid_string):
    """Verifies that the grid string is in the correct format."""
    try:
        return grid_string.startswith("[[") and grid_string.endswith("]]")
    except:
        return False

def post_process_grid(grid_string):
    """Post-processes the grid string to fix common formatting errors."""
    try:
        grid_string = "[[" + grid_string.split("[[")[-1]
        grid_string = grid_string.split("]]")[0] + "]]"
        return grid_string
    except:
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