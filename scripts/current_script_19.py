import os
import re
import math

# Hypothesis: This exploration will implement a "Transformation by Rule Extraction and Decomposition with Multi-Agent Reasoning" approach.
# We will use multi-agent reasoning to have one agent extract rules and another agent decompose the rule and then apply the transformations.
# Agent 1: Rule Extraction Agent (extracts rules from training examples)
# Agent 2: Rule Decomposition & Application Agent (decomposes rule into steps, applies it to test input)

def main(question):
    """Transforms a grid based on multi-agent reasoning and rule decomposition."""
    try:
        # 1. Extract training examples and test input
        training_examples, test_input = preprocess_question(question)

        # 2. Extract transformation rule using Rule Extraction Agent
        transformation_rule = extract_transformation_rule(training_examples)

        # 3. Decompose and apply the rule using Rule Decomposition & Application Agent
        transformed_grid = decompose_and_apply_rule(test_input, transformation_rule)

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

def extract_transformation_rule(training_examples):
    """Extracts the transformation rule from the training examples using LLM."""
    system_instruction = "You are a Rule Extraction Agent. Extract transformation rules from grid examples."
    prompt = f"""
    You are a Rule Extraction Agent. Given training examples of grid transformations, extract the underlying transformation rule in a concise, human-readable format.

    Example 1:
    Training Examples:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Transformation Rule: Each number is incremented by 1.

    Example 2:
    Training Examples:
    Input Grid: [[1, 0], [0, 1]]
    Output Grid: [[0, 1], [1, 0]]
    Transformation Rule: The positions of '1' and '0' are swapped.

    Training Examples:
    {training_examples}
    Transformation Rule:
    """
    transformation_rule = call_llm(prompt, system_instruction)
    return transformation_rule

def decompose_and_apply_rule(test_input, transformation_rule):
    """Decomposes the transformation rule and applies it to the test input using LLM."""
    system_instruction = "You are a Rule Decomposition & Application Agent. Decompose rules into steps and apply them to grids."
    prompt = f"""
    You are a Rule Decomposition & Application Agent. Given a test input grid and a transformation rule, decompose the rule into a series of steps and apply them to reconstruct the output grid, ensuring proper format.
        
    Example 1:
    Test Input: [[5, 6], [7, 8]]
    Transformation Rule: Each number is incremented by 1.
    Decomposed Steps:
    - Increment 5 by 1 to get 6.
    - Increment 6 by 1 to get 7.
    - Increment 7 by 1 to get 8.
    - Increment 8 by 1 to get 9.
    Reconstructed Grid: [[6, 7], [8, 9]]

    Example 2:
    Test Input: [[5, 0], [0, 5]]
    Transformation Rule: The positions of '5' and '0' are swapped.
    Decomposed Steps:
    - Find the positions of '5' and '0'.
    - Swap '5' and '0' in the grid.
    Reconstructed Grid: [[0, 5], [5, 0]]

    Test Input:
    {test_input}
    Transformation Rule:
    {transformation_rule}
    Decomposed Steps and Reconstructed Grid:
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