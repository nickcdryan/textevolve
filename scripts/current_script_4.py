import os
import re
import math

def main(question):
    """
    Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach uses a novel "spatial encoding and transformation prediction" strategy.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """
    Solves the grid transformation problem by first extracting spatial encodings, then predicting the transformation.
    """

    system_instruction = "You are an expert at identifying grid transformation patterns. You will encode spatial information and predict transformations."

    # STEP 1: Extract spatial encoding and predict potential transformations
    spatial_encoding_prompt = f"""
    You are tasked with identifying spatial relationships and predicting transformations in grids. Study the examples carefully.

    Example 1:
    Input Grid:
    [[1, 0], [0, 1]]
    Predicted Transformation:
    "Expand the grid by repeating each element diagonally, creating a larger grid."

    Example 2:
    Input Grid:
    [[2, 8], [8, 2]]
    Predicted Transformation:
    "Expand each element into a 2x2 block, creating a larger grid."

    Now, analyze this input and predict the most likely transformation. Respond with ONLY the predicted transformation:
    Test Input:
    {problem_text}
    """

    # Attempt to extract the rule
    predicted_transformation = call_llm(spatial_encoding_prompt, system_instruction)

    # STEP 2: Verify predicted transformation using few-shot examples and apply transformation with more explicit prompting.
    verified_transformation_prompt = f"""
    You have extracted the following predicted transformation:
    {predicted_transformation}

    Verify that this is the correct transformation based on these training example grids:

    Example 1:
    Input Grid:
    [[1, 0], [0, 1]]
    Output Grid:
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    Input Grid:
    [[2, 8], [8, 2]]
    Output Grid:
    [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]
    
    Now apply the verified transformation to the following test input grid:
    {problem_text}

    Provide the transformed grid as a 2D array formatted as a string, WITHOUT any additional explanation or comments.
    """

    # Attempt to generate the transformed grid
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(verified_transformation_prompt, system_instruction)
            # Basic validation - check if it looks like a grid
            if "[" in transformed_grid_text and "]" in transformed_grid_text:
                return transformed_grid_text
            else:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # Fallback approach if all attempts fail
    return "[[0,0,0],[0,0,0],[0,0,0]]"

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