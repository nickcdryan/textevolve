import os
import re
import math

def solve_grid_transformation(question):
    """Solves a grid transformation problem using a novel LLM-driven approach."""

    # HYPOTHESIS: Instead of extracting explicit rules, can we get the LLM to generate multiple plausible output grids, 
    # and then use a secondary LLM to select the most likely one, acting as an ensemble? This tests whether the LLM can do
    # direct pattern matching better than rule extraction.

    # Step 1: Generate multiple output grids
    num_grids = 3  # Generate multiple candidate grids
    generated_grids = generate_multiple_grids(question, num_grids)

    # Step 2: Select the best grid using a secondary LLM
    best_grid = select_best_grid(question, generated_grids)

    return best_grid

def generate_multiple_grids(question, num_grids):
    """Generates multiple possible output grids using the LLM."""
    system_instruction = "You are an expert grid transformer that generates multiple plausible output grids based on given examples."
    grids = []

    for i in range(num_grids):
        prompt = f"""
        Given the following grid transformation problem, generate a plausible output grid.
        Consider different possible transformation patterns and generate a UNIQUE, plausible output.
        This is attempt {i+1}/{num_grids}, so consider a different pattern from previous attempts.

        Example 1:
        Input Grid:
        [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
        Output Grid:
        [[0, 2, 0], [2, 2, 0], [0, 2, 0]]

        Example 2:
        Input Grid:
        [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
        Output Grid:
        [[2, 2, 2], [0, 0, 0], [2, 2, 2]]

        Problem:
        {question}

        Output Grid:
        """

        output_grid = call_llm(prompt, system_instruction)
        grids.append(output_grid)
    return grids

def select_best_grid(question, generated_grids):
    """Selects the best output grid from a list of generated grids using an LLM."""
    system_instruction = "You are an expert grid evaluator that selects the best output grid based on the problem description."
    prompt = f"""
    Given the following grid transformation problem and a list of generated output grids, select the best one.
    Consider the transformation patterns in the examples and select the grid that best follows those patterns.

    Example:
    Problem:
    Grid Transformation Task
    Input Grid:
    [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    Output Grid Options:
    1: [[0, 2, 0], [2, 2, 0], [0, 2, 0]]
    2: [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
    Best Grid: 1 (Correct transformation from 1 to 2 in the input grid)

    Problem:
    {question}

    Output Grid Options:
    {chr(10).join([f"{i+1}: {grid}" for i, grid in enumerate(generated_grids)])}

    Best Grid (Enter the number corresponding to best grid):
    """

    best_grid_number = call_llm(prompt, system_instruction)
    try:
        best_grid_index = int(best_grid_number) - 1
        if 0 <= best_grid_index < len(generated_grids):
            return generated_grids[best_grid_index]
        else:
            return "Error: Invalid grid number."
    except ValueError:
        return "Error: Invalid grid number format."

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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"