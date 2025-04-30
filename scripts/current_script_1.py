import os
import re

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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

def analyze_transformation_pattern(training_examples):
    """Analyzes the transformation pattern from training examples using LLM with examples."""
    system_instruction = "You are an expert pattern analyst for grid transformations."

    prompt = f"""
    Analyze the transformation pattern from these training examples. Provide a detailed description of the pattern in natural language.

    Example 1:
    Input: [[1, 0], [0, 1]]
    Output: [[0, 1], [1, 0]]
    Pattern: The transformation reflects the grid along the main diagonal (transpose).

    Example 2:
    Input: [[1, 2], [3, 4]]
    Output: [[2, 1], [4, 3]]
    Pattern: The transformation swaps elements in the same row.

    Example 3:
    Input: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    Output: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Pattern: Transformation reflects the grid across the horizontal center.

    Training Examples: {training_examples}
    Pattern:
    """
    return call_llm(prompt, system_instruction)

def apply_transformation(pattern_description, test_input):
    """Applies the transformation pattern to the test input using LLM with examples."""
    system_instruction = "You are an expert at applying grid transformation patterns."

    prompt = f"""
    Apply the transformation pattern to the test input. The transformation pattern is described as: {pattern_description}.

    Example 1:
    Pattern: The transformation reflects the grid along the main diagonal (transpose).
    Input: [[1, 2], [3, 4]]
    Output: [[1, 3], [2, 4]]

    Example 2:
    Pattern: The transformation swaps elements in the same row.
    Input: [[1, 2, 3], [4, 5, 6]]
    Output: [[2, 1, 3], [5, 4, 6]]

    Example 3:
    Pattern: Transformation reflects the grid across the horizontal center.
    Input: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    Output: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]

    Test Input: {test_input}
    Transformed Grid:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to process the grid transformation task."""
    try:
        # Extract training examples and test input from the question string
        training_examples_match = re.search(r"Training Examples:\n(.*?)\n\nTest Input:", question, re.DOTALL)
        test_input_match = re.search(r"Test Input:\n(.*?)\n", question, re.DOTALL)

        if not training_examples_match or not test_input_match:
            return "Error: Could not extract training examples or test input."

        training_examples = training_examples_match.group(1).strip()
        test_input = test_input_match.group(1).strip()

        # Analyze the transformation pattern
        pattern_description = analyze_transformation_pattern(training_examples)

        # Apply the transformation to the test input
        transformed_grid = apply_transformation(pattern_description, test_input)

        return transformed_grid

    except Exception as e:
        return f"Error during processing: {str(e)}"