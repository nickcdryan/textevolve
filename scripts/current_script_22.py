import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Semantic Chunking and Focused Refinement" approach.
# 1. Divide the training examples into semantic chunks (e.g., "rows change", "columns change", "diagonals change", "corners change")
# 2. Have a "Refinement Agent" focus on learning only from relevant semantic chunks.
# Add multiple examples with detailed reasoning to each prompt to improve pattern extraction and generalization.

def main(question):
    """Transforms a grid by semantic chunking and focused refinement."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Chunk the training examples
        chunked_examples = chunk_training_examples(training_examples)

        # 3. Transform the test input using focused refinement
        transformed_grid = transform_with_refinement(test_input, chunked_examples)

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

def chunk_training_examples(training_examples):
    """Chunks the training examples into semantic groups."""
    system_instruction = "You are a semantic chunker who divides examples into relevant groups."
    prompt = f"""
    You are a semantic chunker who divides training examples into relevant groups such as rows, columns, diagonals and corners based on their transformation behavior.

    Example:
    Training Examples:
    Input Grid: [[1, 2, 3], [4, 5, 6]]
    Output Grid: [[2, 3, 4], [4, 5, 6]]
    Chunks: Row change

    Input Grid: [[1, 2, 3], [4, 5, 6]]
    Output Grid: [[1, 4, 3], [2, 5, 6]]
    Chunks: Column change

    Training Examples:
    {training_examples}
    Chunks:
    """
    chunked_examples = call_llm(prompt, system_instruction)
    return chunked_examples

def transform_with_refinement(test_input, chunked_examples):
    """Transforms the test input using focused refinement with semantic chunks."""
    system_instruction = "You are a Refinement Agent who transforms grids based on semantic chunks."
    prompt = f"""
    You are a Refinement Agent who transforms the test input grid based on the training examples based on semantic chunks.

    Example:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Chunks: Row change

    Test Input: [[5, 6], [7, 8]]
    Transformed Grid: [[6, 7], [7, 8]]

    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[1, 3], [2, 4]]
    Chunks: Column Change

    Test Input: [[5, 6], [7, 8]]
    Transformed Grid: [[5, 8], [6, 7]]
    
    Test Input:
    {test_input}

    Chunked Examples:
    {chunked_examples}
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

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