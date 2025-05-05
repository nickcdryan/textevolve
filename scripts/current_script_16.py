import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Iterative Value Propagation and Spatial Contextualization" approach.
# We will decompose the grid transformation into iterative steps where values propagate based on their spatial context.
# The core idea is that each cell's value is determined not just by a single transformation rule but by considering its neighbors.
# We hypothesize that by iteratively contextualizing each cell based on its neighbors and previous values, we can capture more complex transformations.
# This approach will use multiple validation steps to determine successful parts of the pipeline.

def main(question):
    """Transforms a grid by iteratively propagating values based on spatial context."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Initialize the value propagation process
        propagated_grid = initialize_propagation(test_input)

        # 3. Iteratively propagate values based on spatial context
        for _ in range(3):  # Run 3 iterations
            propagated_grid = propagate_values(propagated_grid, training_examples)

        return propagated_grid

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def preprocess_question(question):
    """Extract training examples and test input from the question string using regex."""
    try:
        training_examples_match = re.search(r"=== TRAINING EXAMPLES ===\n(.*?)\n=== TEST INPUT ===", question, re.DOTALL)
        test_input_match = re.search(r"=== TEST INPUT ===\n(.*?)\nTransform", question, re.DOTALL)

        training_examples = training_examples_match.group(1).strip() if training_examples_match else ""
        test_input = test_input_match.group(1).strip() if test_input_match else ""

        return training_examples, test_input
    except Exception as e:
        return "", ""

def initialize_propagation(test_input):
    """Initializes the value propagation process by converting the test input into a usable format."""
    system_instruction = "You are an expert in grid initialization for value propagation."
    prompt = f"""
    You are an expert in grid initialization for value propagation. Convert the raw test input string into a list of lists of integers.

    Example:
    Input: [[1, 2], [3, 4]]
    Output: [[1, 2], [3, 4]]

    Now, convert this test input:
    {test_input}
    """
    initial_grid = call_llm(prompt, system_instruction)
    return initial_grid

def propagate_values(grid, training_examples):
    """Propagates values based on spatial context using the LLM."""
    system_instruction = "You are an expert in value propagation within grids, considering spatial contexts."
    prompt = f"""
    You are an expert in value propagation within grids, considering spatial contexts.
    Given a grid and training examples, propagate the values based on their neighbors and positions. This means you adjust the values in each cell based on patterns from the examples.

    Example:
    Training Examples:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[0, 1], [1, 0]]
    Current Grid: [[5, 0], [0, 5]]
    Propagated Grid: [[0, 5], [5, 0]]

    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Current Grid: [[5, 6], [7, 8]]
    Propagated Grid: [[6, 7], [8, 9]]
    
    Now, for this new grid and training examples:
    Training Examples:
    {training_examples}
    Current Grid:
    {grid}
    Propagated Grid:
    """
    propagated_grid = call_llm(prompt, system_instruction)
    return propagated_grid

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