import os
import re
import math

# Hypothesis: This exploration will focus on a "Transformation by Analogy" approach. Instead of generating rules or classifying the transformation type,
# the LLM will be prompted to directly translate the TRAINING EXAMPLES' transformation to the TEST INPUT, drawing a direct analogy between the two scenarios.
# We hypothesize that this direct translation can be more effective than explicit rule extraction, as it relies more on pattern completion than formal reasoning.
# The goal is to lean on the LLM's ability to see high level relationships. Also, this is an attempt to address the common failure modes that we are currently dealing with.

def main(question):
    """Transforms a grid by drawing a direct analogy from the training examples."""
    try:
        # 1. Translate the training examples to the test input
        transformed_grid = translate_transformation(question)
        return transformed_grid
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def translate_transformation(question, max_attempts=3):
    """Translates the training examples' transformation to the test input."""
    system_instruction = "You are an expert in drawing analogies between grid transformation examples."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert in drawing analogies between grid transformation examples.
        Given a question containing training examples and a test input, translate the transformations shown in the training examples to the test input.
        Focus on *how* the input grid is changed in the training examples and apply a *similar* change to the test input.
        The transformed grid should be returned in string representation that begins with '[[' and ends with ']]'. Do not describe any analysis or reasoning.

        Example 1:
        Training Input: [[1, 2], [3, 4]]
        Training Output: [[2, 3], [4, 5]]
        Test Input: [[5, 6], [7, 8]]
        Transformed Grid: [[6, 7], [8, 9]]

        Example 2:
        Training Input: [[1, 2], [3, 4]]
        Training Output: [[2, 1], [4, 3]]
        Test Input: [[5, 6], [7, 8]]
        Transformed Grid: [[6, 5], [8, 7]]
        
        Example 3:
        Training Input: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        Training Output: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        Test Input: [[5, 6, 5], [6, 5, 6], [5, 6, 5]]
        Transformed Grid: [[6, 5, 6], [5, 6, 5], [6, 5, 6]]

        Now, for this new question, translate the transformation:
        {question}
        """
        transformed_grid = call_llm(prompt, system_instruction)

        # Verification step: check if the output is a valid grid
        verification_result = verify_grid_format(question, transformed_grid) #use the same validator
        if verification_result["is_valid"]:
            return transformed_grid
        else:
            print(f"Transformation failed (attempt {attempt+1}/{max_attempts}): {verification_result['feedback']}")

    return "Failed to transform the grid correctly after multiple attempts."

def verify_grid_format(question, transformed_grid):
    """Verifies that the transformed grid is in the proper format."""
    try:
        if not (transformed_grid.startswith("[[") and transformed_grid.endswith("]]")):
            return {"is_valid": False, "feedback": "Output should start with '[[' and end with ']]'."}

        # Basic check for grid structure
        grid_rows = transformed_grid.strip("[]").split("],[")
        if not all("," in row for row in grid_rows):
            return {"is_valid": False, "feedback": "Rows are not comma separated."}

        return {"is_valid": True}
    except Exception as e:
        return {"is_valid": False, "feedback": f"Error during grid validation: {str(e)}"}

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