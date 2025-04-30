import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks by using a novel decomposition and
    iterative refinement approach. This approach aims to address the
    limitations of previous iterations by focusing on verifiable sub-goals
    and avoiding reliance on monolithic LLM calls.
    """
    try:
        # Step 1: Extract structured information using LLM
        extraction_result = extract_grid_info(question)
        if "Error" in extraction_result:
            return f"Extraction failed: {extraction_result}"

        # Step 2: Hypothesize potential transformation patterns
        pattern_hypotheses = hypothesize_transformation(extraction_result)
        if "Error" in pattern_hypotheses:
            return f"Hypothesis generation failed: {pattern_hypotheses}"

        # Step 3: Apply the most promising hypothesis
        transformed_grid = apply_hypothesis(extraction_result, pattern_hypotheses)

        return transformed_grid

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_grid_info(question):
    """
    Extracts structured information (training examples and test input) from
    the input question. This utilizes a multi-example prompt for better
    accuracy.
    """
    system_instruction = "You are an expert in extracting structured data from text."
    prompt = f"""
    Extract training examples and test input from the following text.

    Example 1:
    Text: Grid Transformation Task
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Test Input: [[5, 6], [7, 8]]
    Extracted Data:
    {{
        "training_examples": '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]'
        "test_input": "[[5, 6], [7, 8]]"
    }}

    Example 2:
    Text: Grid Transformation Task
    Training Examples:
    [
        {{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}}
    ]
    Test Input: [[9, 10], [11, 12]]
    Extracted Data:
    {{
        "training_examples": '[{{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}}]'
        "test_input": "[[9, 10], [11, 12]]"
    }}

    Text: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def hypothesize_transformation(extraction_result):
    """
    Hypothesizes potential transformation patterns from training examples.
    This function prioritizes generating multiple hypotheses, rather than
    a single, potentially incorrect, description.
    """
    system_instruction = "You are an expert in hypothesizing transformation patterns."
    prompt = f"""
    Given the following training examples, generate three different hypotheses
    about the transformation pattern.

    Example:
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}
    ]
    Hypotheses:
    1. The transformation transposes the input grid and reflects it along both diagonals.
    2. The transformation reflects the input grid along both diagonals and then transposes it.
    3.  The transformation swaps elements such that input[i][j] becomes output[N-1-i][N-1-j].

    Training Examples:
    {extraction_result}
    Hypotheses:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error generating hypotheses: {str(e)}"

def apply_hypothesis(extraction_result, pattern_hypotheses):
    """
    Applies the most promising hypothesis to the test input.
    """
    system_instruction = "You are an expert in applying transformation hypotheses."
    prompt = f"""
    Given the following test input and transformation hypotheses, apply the
    first hypothesis to the test input and provide the transformed grid.

    Example:
    Test Input: [[5, 6], [7, 8]]
    Hypotheses:
    1. The transformation transposes the input grid and reflects it along both diagonals.
    Transformed Grid: [[8, 7], [6, 5]]

    Test Input: {extraction_result}
    Hypotheses:
    {pattern_hypotheses}
    Transformed Grid:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error applying hypothesis: {str(e)}"

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