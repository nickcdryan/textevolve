import os
import re
import math

# Hypothesis: Instead of attempting to decompose the problem into multiple LLM calls, can we create an effective "one-shot" approach
# by providing highly detailed examples and leveraging in-context learning? Can we reduce the failure rate by minimizing the number of calls?
# By including examples of the expected reasoning steps, can we guide the model to produce correct outputs and transformations?

def main(question):
    """Transforms a grid based on highly detailed examples, using a one-shot approach with LLM."""
    try:
        transformed_grid = transform_grid_with_detailed_examples(question)
        return transformed_grid
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def transform_grid_with_detailed_examples(question, max_attempts=3):
    """Transforms the grid using a one-shot approach with detailed examples."""
    system_instruction = "You are an expert grid transformer. Follow the examples closely."
    prompt = f"""
    You are a grid transformation expert. Analyze the training examples and transform the test input accordingly.
    Provide detailed reasoning steps before giving the final transformed grid.
    The final answer MUST be a string representation of the grid, starting with '[[' and ending with ']]'.

    Example 1:
    Input Grid: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 7, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 7, 2, 7, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    Reasoning: The transformation appears to replace all '0' values surrounding each '2' or '7' value with the value of the '2' that is closest to each '0'
    Output Grid: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 7, 0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 2, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0], [7, 7, 2, 7, 7, 0, 0, 0, 0, 0, 0, 0], [0, 2, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 7, 0, 2, 0, 2, 0, 7, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0, 2, 7, 2, 0, 0], [0, 0, 0, 0, 0, 0, 7, 7, 2, 7, 7, 0], [0, 0, 0, 0, 0, 0, 0, 2, 7, 2, 0, 0], [0, 0, 0, 0, 0, 0, 2, 0, 7, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    Example 2:
    Input Grid: [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    Reasoning: The transformation appears to replicate the '2' and '8' values across the grid. It appears to create repeating columns of 2 and 8.
    Output Grid: [[0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0]]

    Now transform the input grid below, providing detailed reasoning first, and then the final grid string. The final grid string MUST start with '[[' and end with ']]':
    {question}
    """

    for attempt in range(max_attempts):
        transformed_grid = call_llm(prompt, system_instruction)
        # Validation using regex: Check for start and end characters.
        if not (transformed_grid.startswith("[[") and transformed_grid.endswith("]]")):
            print(f"Validation failed: Output does not start with '[[' and end with ']]'.")
            continue

        # Attempt minimal cleaning of extra characters outside the grid:
        cleaned_grid = transformed_grid.split('[[', 1)[1].rsplit(']]', 1)[0]
        # Ensure grid format is correct (replace spaces, check number format...)
        cleaned_grid = '[[' + cleaned_grid.replace(" ", "") + ']]'

        try:
            #Attempt basic parsing to confirm it at least "looks" like data
            grid_rows = cleaned_grid.strip("[]").split("],[")
            #Minimal validation: are we getting at least rows of data back?
            if len(grid_rows) > 0:
                 return cleaned_grid
            else:
                 print(f"Transformed grid appears to be empty.")
                 continue

        except:
            print(f"Transformed grid is not a valid format.")
            continue # Try again

    return "Failed to transform grid correctly after multiple attempts."

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