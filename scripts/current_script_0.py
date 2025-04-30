import os
import re

def main(question):
    """
    Transforms a grid based on patterns in training examples, leveraging LLMs for reasoning and pattern recognition.
    """
    # Step 1: Extract the training examples and test input from the question.
    training_examples, test_input = extract_training_and_test(question)

    # Step 2: Analyze the training examples to infer the transformation pattern.
    pattern = analyze_transformation_pattern(training_examples)

    # Step 3: Apply the transformation pattern to the test input to generate the output.
    output = apply_transformation(test_input, pattern)

    return output

def extract_training_and_test(question):
    """Extracts training examples and test input from the question using LLM reasoning."""

    system_instruction = "You are an expert at extracting relevant information from a text."
    prompt = f"""
    Extract the training examples and test input from the following question.

    Example 1:
    Question: Grid Transformation Task\n\nTraining Examples:\n[{{\"input\":[[0,7,7],[7,7,7],[0,7,7]],\"output\":[[0,0,0,0,7,7,0,7,7],[0,0,0,7,7,7,7,7,7],[0,0,0,0,7,7,0,7,7],[0,7,7,0,7,7,0,7,7],[7,7,7,7,7,7,7,7,7],[0,7,7,0,7,7,0,7,7],[0,0,0,0,7,7,0,7,7],[0,0,0,7,7,7,7,7,7],[0,0,0,0,7,7,0,7,7]]}}]\n\nTest Input:\n[[7,0,7],[7,0,7],[7,7,0]]
    Training Examples: [{{\"input\":[[0,7,7],[7,7,7],[0,7,7]],\"output\":[[0,0,0,0,7,7,0,7,7],[0,0,0,7,7,7,7,7,7],[0,0,0,0,7,7,0,7,7],[0,7,7,0,7,7,0,7,7],[7,7,7,7,7,7,7,7,7],[0,7,7,0,7,7,0,7,7],[0,0,0,0,7,7,0,7,7],[0,0,0,7,7,7,7,7,7],[0,0,0,0,7,7,0,7,7]]}}]
    Test Input: [[7,0,7],[7,0,7],[7,7,0]]

    Example 2:
    Question: Grid Transformation Task\n\nTraining Examples:\n[{{\"input\":[[0,0],[0,1]],\"output\":[[0,0,0,0],[0,1,0,1]]}}]\n\nTest Input:\n[[1,1],[0,0]]
    Training Examples: [{{\"input\":[[0,0],[0,1]],\"output\":[[0,0,0,0],[0,1,0,1]]}}]
    Test Input: [[1,1],[0,0]]

    Question: {question}
    Training Examples:
    Test Input:
    """

    response = call_llm(prompt, system_instruction)
    training_examples = re.search(r"Training Examples:\s*(.*)\n", response).group(1)
    test_input = re.search(r"Test Input:\s*(.*)", response).group(1)

    return training_examples, test_input

def analyze_transformation_pattern(training_examples_str):
    """Analyzes the training examples to determine the transformation pattern using LLM reasoning."""
    system_instruction = "You are an AI that identifies patterns between input and output grids."
    prompt = f"""
    Analyze the following training examples and describe the transformation pattern.

    Example 1:
    Training Examples: [{{"input":[[0,7,7],[7,7,7],[0,7,7]],"output":[[0,0,0,0,7,7,0,7,7],[0,0,0,7,7,7,7,7,7],[0,0,0,0,7,7,0,7,7],[0,7,7,0,7,7,0,7,7],[7,7,7,7,7,7,7,7,7],[0,7,7,0,7,7,0,7,7],[0,0,0,0,7,7,0,7,7],[0,0,0,7,7,7,7,7,7],[0,0,0,0,7,7,0,7,7]]}}]
    Pattern: The pattern is to expand the 3x3 grid to a 9x9 grid by repeating rows and columns.

    Example 2:
    Training Examples: [{{"input":[[0,0],[0,1]],"output":[[0,0,0,0],[0,1,0,1]]}}]
    Pattern: The pattern is to expand the 2x2 grid to a 2x4 grid by repeating columns.

    Training Examples: {training_examples_str}
    Pattern:
    """
    pattern = call_llm(prompt, system_instruction)
    return pattern

def apply_transformation(test_input_str, pattern):
    """Applies the transformation pattern to the test input, leveraging LLM for reasoning."""
    system_instruction = "You are an AI that applies patterns to transform grid data."
    prompt = f"""
    Apply the following pattern to the given test input.

    Example 1:
    Pattern: The pattern is to expand the 3x3 grid to a 9x9 grid by repeating rows and columns.
    Test Input: [[7,0,7],[7,0,7],[7,7,0]]
    Transformed Output: [[7,0,7,0,0,0,7,0,7],[7,0,7,0,0,0,7,0,7],[7,7,0,0,0,0,7,7,0],[7,0,7,0,0,0,7,0,7],[7,0,7,0,0,0,7,0,7],[7,7,0,0,0,0,7,7,0],[7,0,7,7,0,7,0,0,0],[7,0,7,7,0,7,0,0,0],[7,7,0,7,7,0,0,0,0]]

    Example 2:
    Pattern: The pattern is to expand the 2x2 grid to a 2x4 grid by repeating columns.
    Test Input: [[1,1],[0,0]]
    Transformed Output: [[1,1,1,1],[0,0,0,0]]

    Pattern: {pattern}
    Test Input: {test_input_str}
    Transformed Output:
    """
    transformed_output = call_llm(prompt, system_instruction)
    return transformed_output
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