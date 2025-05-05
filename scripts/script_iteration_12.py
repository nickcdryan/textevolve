import os
import re
import math

# Hypothesis: This exploration will focus on a "Decomposed Transformation Analysis with Iterative Refinement" approach.
# The LLM will first attempt to identify the core transformation by looking at differences between the grids, and also use some spatial analysis
# We hypothesize that by prompting for transformation in iterative steps, we will have a better capability in identifying complex patterns.
# The goal is to lean on the LLM's ability to see high level relationships. Also, this is an attempt to address the common failure modes that we are currently dealing with.
# We will validate in-process with verifier calls after each step, to determine if we were successful and where the model is breaking.

def main(question):
    """Transforms a grid by analyzing transformations in iterative steps."""
    try:
        # 1. Analyze the transformation and decompose into steps
        transformation_analysis = analyze_transformation(question)

        # 2. Apply the transformation to the test input
        transformed_grid = apply_transformation(question, transformation_analysis)

        return transformed_grid
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def analyze_transformation(question, max_attempts=3):
    """Analyzes the grid transformation and decomposes it into steps."""
    system_instruction = "You are an expert in analyzing grid transformations and breaking them down into manageable steps."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert in analyzing grid transformations and breaking them down into manageable steps.
        Given a question containing training examples, analyze the transformations and decompose them into a sequence of steps. Focus on understanding the core change and the spatial relationships involved.

        Example 1:
        Input Grid: [[1, 2], [3, 4]]
        Output Grid: [[2, 3], [4, 5]]
        Transformation Analysis: Increment each element by 1.

        Example 2:
        Input Grid: [[1, 2], [3, 4]]
        Output Grid: [[2, 1], [4, 3]]
        Transformation Analysis: Swap the first and second elements in each row.

        Example 3:
        Input Grid: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        Output Grid: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        Transformation Analysis: Swap all 0s with 1s, and all 1s with 0s

        Now, for this new question, analyze the transformation:
        {question}
        """
        analysis = call_llm(prompt, system_instruction)

        # Verification step: Validate with a verifier
        if verify_analysis(question, analysis):
            return analysis
        else:
            print(f"Transformation analysis failed (attempt {attempt+1}/{max_attempts}).")
            if attempt == max_attempts - 1:
                print("Returning failed analysis, please debug")
                return analysis
            continue

def apply_transformation(question, transformation_analysis, max_attempts=3):
    """Applies the analyzed transformation to the test input."""
    system_instruction = "You are an expert in applying grid transformations."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert in applying grid transformations.
        Given a question containing a test input and a transformation analysis, apply the transformation to the input grid. Return ONLY the transformed grid as string
        Example 1:
        Question: Test input = [[0, 0], [0, 0]], Analysis: Increment every value by 1
        Transformed Grid: [[1, 1], [1, 1]]

        Example 2:
        Question: Test input = [[1, 2], [3, 4]], Analysis: swap the diagonal
        Transformed Grid: [[4, 2], [3, 1]]

        Example 3:
        Question: Test input = [[1, 2], [3, 4]], Analysis: increment the rows by their indices in the array (start at index 0)
        Transformed Grid: [[1, 2], [4, 5]]

        Now, for this new question, apply the transformation:
        {question}

        Transformation Analysis:
        {transformation_analysis}

        Transformed Grid (string):
        """
        transformed_grid = call_llm(prompt, system_instruction)

        # Verification step: Verify the transformed grid format
        if verify_grid_format(transformed_grid):
            return transformed_grid
        else:
            print(f"Transformation failed (attempt {attempt+1}/{max_attempts}).")
            if attempt == max_attempts - 1:
                return transformed_grid
            continue

def verify_grid_format(grid_string):
    """Verifies that the grid string is in the proper format."""
    try:
        if not (grid_string.startswith("[[") and grid_string.endswith("]]")):
            return False

        # Basic check for grid structure
        grid_rows = grid_string.strip("[]").split("],[")
        if not all("," in row for row in grid_rows):
            return False

        return True
    except Exception as e:
        return False

def verify_analysis(question, analysis):
    #Implement a more thorough approach to validating and scoring code - NOT USED
    if not isinstance(analysis, str):
        return False
    if len(analysis) == 0:
        return False

    #Add more logic here
    return True

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