import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Visual Representation and Analogy Reasoning" approach.
# This approach converts the grid into a visual representation and uses analogy reasoning to transform it.
# It aims to enhance pattern recognition by leveraging visual cues and analogies, a distinctly different approach than before.
# Add verification steps to understand which parts are successful and where it is breaking.

def main(question):
    """Transforms a grid by converting it into a visual representation and using analogy reasoning."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Convert test input grid to a visual representation
        visual_representation = grid_to_visual(test_input)

        # 3. Transform visual representation based on training examples using analogy reasoning
        transformed_visual = transform_visual_analogy(visual_representation, training_examples)

        # 4. Convert transformed visual representation back to a grid
        transformed_grid = visual_to_grid(transformed_visual)

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

def grid_to_visual(test_input):
    """Converts the grid into a visual representation (string with special characters)."""
    system_instruction = "You are an expert in converting grids to visual representations."
    prompt = f"""
    You are an expert in converting grids to visual representations.
    Given a grid, represent each number with a unique character. Make the visualization clear and representative of the original structure.

    Example:
    Input: [[1, 2], [3, 4]]
    Visual:
    A B
    C D

    Input:
    {test_input}
    Visual:
    """
    visual_representation = call_llm(prompt, system_instruction)
    return visual_representation

def transform_visual_analogy(visual_representation, training_examples):
    """Transforms the visual representation based on analogy reasoning using LLM."""
    system_instruction = "You are an AI expert in visual transformation and analogy reasoning."
    prompt = f"""
    You are an AI expert in visual transformation and analogy reasoning.
    Given a visual representation of a grid and training examples of grid transformations, transform the visual representation based on the patterns.

    Example:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Input Visual:
    A B
    C D
    Output Visual:
    B C
    D E
    Current Visual:
    E F
    G H
    Transformed Visual:
    F G
    H I

    Training Examples:
    {training_examples}
    Current Visual:
    {visual_representation}
    Transformed Visual:
    """
    transformed_visual = call_llm(prompt, system_instruction)
    return transformed_visual

def visual_to_grid(transformed_visual):
    """Converts the transformed visual representation back to a grid."""
    system_instruction = "You are an expert in converting visual representations back to numerical grids."
    prompt = f"""
    You are an expert in converting visual representations back to numerical grids.
    Given a transformed visual representation, convert it back to its numerical grid format.

    Example:
    Visual:
    B C
    D E
    Transformed Grid: [[2, 3], [4, 5]]

    Visual:
    {transformed_visual}
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