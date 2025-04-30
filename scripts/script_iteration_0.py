import os
import re
import math

def main(question):
    """
    Solve grid transformation tasks by analyzing training examples and applying the learned pattern to the test input.
    Leverages LLM for pattern recognition and transformation.
    """
    try:
        # Step 1: Analyze and extract the transformation pattern with multiple examples
        pattern_analysis_result = analyze_transformation_pattern(question)
        if "Error" in pattern_analysis_result:
            return f"Pattern analysis failed: {pattern_analysis_result}"

        # Step 2: Apply the pattern to the test input
        transformation_result = apply_transformation(question, pattern_analysis_result)
        if "Error" in transformation_result:
            return f"Transformation failed: {transformation_result}"

        return transformation_result  # Already formatted as a string

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def analyze_transformation_pattern(question):
    """Analyze training examples to extract the transformation pattern."""
    system_instruction = "You are an expert pattern analyst who extracts transformation rules from grid examples."
    prompt = f"""
    Analyze the training examples to identify the transformation pattern.
    Provide a description of the pattern that can be used to transform the test input.

    Example 1:
    Training Examples:
    [
        {{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}},
        {{"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]}}
    ]
    Test Input: [[4,5],[5,4]]
    Pattern: The transformation transposes the input grid.

    Example 2:
    Training Examples:
    [
        {{"input": [[1, 2], [3, 4]], "output": [[2, 4], [1, 3]]}},
        {{"input": [[5, 6], [7, 8]], "output": [[6, 8], [5, 7]]}}
    ]
    Test Input: [[9,10],[11,12]]
    Pattern: The transformation swaps the first row with the second row and applies an offset.

    Training Examples:
    {question}
    Pattern:
    """

    return call_llm(prompt, system_instruction)

def apply_transformation(question, pattern_description):
    """Apply the extracted transformation pattern to the test input."""
    system_instruction = "You are an expert transformer who transforms grids based on given patterns."
    prompt = f"""
    Apply the transformation pattern to the test input and provide the transformed grid as a string.
    
    Example Input:
    Training Examples:
    [
        {{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}}
    ]
    Test Input: [[4,5],[5,4]]
    Pattern: The transformation transposes the input grid.
    Transformed Grid: [[5, 4], [4, 5]]

    
    Training Examples:
    {question}
    Test Input:
    {question}
    Pattern: {pattern_description}
    Transformed Grid:
    """

    return call_llm(prompt, system_instruction)

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