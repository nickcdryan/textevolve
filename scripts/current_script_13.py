import os
import re
import math

# Hypothesis: This exploration will focus on a "Transformation by Feature Vector Analysis and Reconstruction" approach.
# Instead of directly manipulating the grid or generating transformation rules, we will represent each grid as a feature vector,
# analyze the relationship between the input and output feature vectors, and then reconstruct the output grid from a transformed feature vector.
# This approach aims to abstract away from the specific grid structure and focus on higher-level relationships.
# Also, this is an attempt to address the common failure modes of incorrect pattern identification and output formatting.

def main(question):
    """Transforms a grid based on feature vector analysis and reconstruction."""
    try:
        # 1. Extract the training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Analyze and transform feature vectors
        transformed_feature_vector = analyze_and_transform_features(training_examples, test_input)

        # 3. Reconstruct the output grid from the transformed feature vector
        reconstructed_grid = reconstruct_grid(transformed_feature_vector, test_input)

        return reconstructed_grid

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

def analyze_and_transform_features(training_examples, test_input):
    """Analyzes the training examples, transforms the features of the test input, and returns the transformed feature vector."""
    system_instruction = "You are an expert in feature extraction and transformation for grid data."
    prompt = f"""
    You are an expert in feature extraction and transformation for grid data. Extract features from training examples, determine the transformation between input and output feature vectors, and apply that transformation to the test input's feature vector.

    Example 1:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Test Input: [[5, 6], [7, 8]]
    Feature Transformation: Increment each number in the feature vector by 1.
    Transformed Feature Vector: [6, 7, 8, 9]

    Example 2:
    Training Examples:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[0, 1], [1, 0]]
    Test Input: [[5, 0], [0, 5]]
    Feature Transformation: Swap the positions of the "5" and "0".
    Transformed Feature Vector: [0, 5, 5, 0]

    Now, for this new task:
    Training Examples:
    {training_examples}
    Test Input:
    {test_input}
    Feature Transformation and Transformed Feature Vector:
    """
    transformed_feature_vector = call_llm(prompt, system_instruction)
    return transformed_feature_vector

def reconstruct_grid(transformed_feature_vector, test_input):
    """Reconstructs the output grid from the transformed feature vector, ensuring proper format."""
    system_instruction = "You are an expert in reconstructing grid data from feature vectors."
    prompt = f"""
    You are an expert in reconstructing grid data from feature vectors, and output formatting.
    Given a test input and a transformed feature vector, reconstruct the output grid.
    Ensure the output is a string with proper double brackets, and the grid has the same dimensions as the test input.

    Example:
    Test Input: [[5, 6], [7, 8]]
    Transformed Feature Vector: [6, 7, 8, 9]
    Reconstructed Grid: [[6, 7], [8, 9]]

    Now, for this new task:
    Test Input:
    {test_input}
    Transformed Feature Vector:
    {transformed_feature_vector}
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