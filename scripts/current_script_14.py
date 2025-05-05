import os
import re
import math

# Hypothesis: This exploration will implement a "Grid Transformation by Contextual Feature Highlighting and Targeted Modification" approach.
# We hypothesize that emphasizing specific contextual features within the grids and then selectively modifying based on these features will help the LLM generalize.
# This approach is different because it will extract context features from the grid, have the llm review them, and then perform modification to specific locations in the grid.
# The goal is to lean on the LLM's ability to see high level relationships.

def main(question):
    """Transforms a grid by highlighting contextual features and applying targeted modifications."""
    try:
        # 1. Extract the training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Highlight key contextual features in the test input using LLM
        highlighted_features = highlight_contextual_features(test_input, training_examples)

        # 3. Apply targeted modifications based on highlighted features
        transformed_grid = apply_targeted_modifications(test_input, highlighted_features)

        return transformed_grid

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

def highlight_contextual_features(test_input, training_examples):
    """Highlights key contextual features in the test input using the LLM."""
    system_instruction = "You are an expert in identifying key contextual features in grid patterns."
    prompt = f"""
    You are an expert in identifying key contextual features in grid patterns, such as patterns formed by numbers and their locations and neighbors. Given a test input grid and training examples, highlight the most important contextual features that might be relevant for transformation. Focus on identifying spatial patterns, recurring number sequences, and local neighbor relationships.
    Example 1:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Test Input: [[5, 6], [7, 8]]
    Highlighted Features: Each number is incremented by 1.
    Example 2:
    Training Examples:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[0, 1], [1, 0]]
    Test Input: [[5, 0], [0, 5]]
    Highlighted Features: The positions of '5' and '0' are swapped.
    Now, for this new task:
    Training Examples:
    {training_examples}
    Test Input:
    {test_input}
    Highlighted Features:
    """
    highlighted_features = call_llm(prompt, system_instruction)
    return highlighted_features

def apply_targeted_modifications(test_input, highlighted_features):
    """Applies targeted modifications to the test input based on the highlighted features, ensuring proper format."""
    system_instruction = "You are an expert in applying targeted modifications to grid data based on contextual features."
    prompt = f"""
    You are an expert in applying targeted modifications to grid data based on contextual features. Given a test input grid and highlighted features, apply targeted modifications to reconstruct the output grid. Ensure the output is a string with proper double brackets.
    Example:
    Test Input: [[5, 6], [7, 8]]
    Highlighted Features: Each number is incremented by 1.
    Reconstructed Grid: [[6, 7], [8, 9]]
    Now, for this new task:
    Test Input:
    {test_input}
    Highlighted Features:
    {highlighted_features}
    Reconstructed Grid:
    """
    reconstructed_grid = call_llm(prompt, system_instruction)
    return reconstructed_grid

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