import os
import re
import math

# Hypothesis: This exploration will focus on a "Decomposed Spatial Analysis with Targeted Transformations" approach.
# Instead of directly translating the entire grid or generating rules, we decompose the problem into:
# 1. Identifying key spatial features in the input grid (locations of unique numbers, symmetry, etc.)
# 2. Analyzing how these features change between input and output examples.
# 3. Based on these changes, applying targeted transformations to the test input, focusing on preserving or modifying these identified features.
#
# This approach is designed to address the limitations of previous attempts by explicitly focusing on spatial reasoning and feature-based transformation,
# rather than relying solely on the LLM's ability to "guess" the correct transformation, or generating/executing code (which has proven brittle).
# Also, we will use verification calls to verify the changes and what features are retained during transformation

def main(question):
    """Transforms a grid based on decomposed spatial analysis and targeted transformations."""
    try:
        # 1. Identify spatial features
        spatial_features = identify_spatial_features(question)

        # 2. Analyze transformation patterns using identified features
        transformation_patterns = analyze_transformation_patterns(question, spatial_features)

        # 3. Apply targeted transformations to the test input based on analysis
        transformed_grid = apply_targeted_transformations(question, transformation_patterns)

        # Verify output
        transformed_grid = verify_final_output(question, transformed_grid)

        return transformed_grid
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def identify_spatial_features(question):
    """Identifies key spatial features in the input grid."""
    system_instruction = "You are an expert in identifying key spatial features in grid patterns, such as number locations and symmetries."
    prompt = f"""
    You are an expert in identifying key spatial features in grid patterns.
    Given a question containing training examples and a test input, identify and extract key spatial features of the *input* grids:

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Spatial Features:
    {{
      "unique_numbers": [1, 2, 3, 4],
      "number_locations": {{1: [[0, 0]], 2: [[0, 1]], 3: [[1, 0]], 4: [[1, 1]]}},
      "symmetry": "None"
    }}

    Example 2:
    Input Grid: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Spatial Features:
    {{
      "unique_numbers": [0, 1],
      "number_locations": {{0: [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]], 1: [[0, 1], [1, 0], [1, 2], [2, 1]]}},
      "symmetry": "Diagonal"
    }}

    Now, for this new question, identify the spatial features of the *input* grids:
    {question}
    """
    return call_llm(prompt, system_instruction)

def analyze_transformation_patterns(question, spatial_features):
    """Analyzes how spatial features change between input and output examples."""
    system_instruction = "You are an expert in analyzing grid transformations and identifying patterns in how spatial features change."
    prompt = f"""
    You are an expert in analyzing grid transformations and identifying patterns in how spatial features change.
    Given a question containing training examples and extracted spatial features, analyze how these features are transformed from the *input* to the *output* grids:

    Example 1:
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    Spatial Features:
    {{
      "unique_numbers": [1, 2, 3, 4],
      "number_locations": {{1: [[0, 0]], 2: [[0, 1]], 3: [[1, 0]], 4: [[1, 1]]}},
      "symmetry": "None"
    }}
    Transformation Patterns:
    "All numbers are incremented by 1. Symmetry remains None."

    Example 2:
    Input Grid: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    Output Grid: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    Spatial Features:
    {{
      "unique_numbers": [0, 1],
      "number_locations": {{0: [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]], 1: [[0, 1], [1, 0], [1, 2], [2, 1]]}},
      "symmetry": "Diagonal"
    }}
    Transformation Patterns:
    "0s and 1s are swapped. Diagonal symmetry is maintained."

    Now, for this new question, analyze the transformation patterns based on the spatial features:
    {question}
    Spatial Features:
    {spatial_features}
    """
    return call_llm(prompt, system_instruction)

def apply_targeted_transformations(question, transformation_patterns):
    """Applies targeted transformations to the test input based on the analyzed patterns."""
    system_instruction = "You are an expert in applying targeted transformations to grid inputs based on analyzed patterns."
    prompt = f"""
    You are an expert in applying targeted transformations to grid inputs based on analyzed patterns.
    Given a question containing a test input and analyzed transformation patterns, apply these patterns to transform the input grid.
    Return ONLY the transformed grid, WITHOUT any additional text or explanations.

    Question: {question}
    Transformation Patterns: {transformation_patterns}

    Example of correctly formatted output, starting with '[[' and ending with ']]':
    [[1, 2], [3, 4]]
    Result:
    [[2, 3], [4, 5]]

    Now transform the test input. Your output *MUST* start with '[[' and end with ']]':
    Transformed Grid:
    """
    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

def verify_final_output(question, transformed_grid):
    """Verifies that the transformation maintains a specific output and keeps specified traits"""
    system_instruction = "You are an expert grid output verifier. Your job is to confirm if the transformations were successful by confirming that it maintained a specific format"
    prompt = f"""
    You are an expert at confirming final results.
    The main transformation should create an output that starts with '[[' and ends with ']]'. Is this the case?
    Also check if a previous spatial feature, if it existed, is properly transferred into the solution

    Transformed Grid: {transformed_grid}
    Question: {question}

    Answer:
    """
    is_formatted_correctly = call_llm(prompt, system_instruction)

    return is_formatted_correctly

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