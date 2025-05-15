import os
import re
import math

# EXPLORATION: Contextual Grid Completion with Value Propagation using a Pattern-Based Template and LLM Verification
# HYPOTHESIS: By using a structured template that identifies key locations and context numbers within the grid, and having the LLM extrapolate from training examples to "complete" the grid by filling in missing values, we can improve generalization. The system will first identify locations and the context number, then have an LLM perform the extrapolation. This will have an LLM use a template to help identify patterns and perform a simple propagation to identify missing numbers.
# This approach directly addresses previous weaknesses of failing to generalize pattern transformations and accurately transforming numbers.
# It focuses on a simpler processing state to ensure that the overall structure can be more accurately represented for transformations.

def solve_grid_transformation(question):
    """Solves grid transformation problems using contextual grid completion and pattern-based template."""
    try:
        # Step 1: Identify Locations and Context Numbers
        template_result = identify_locations_and_context(question)
        if not template_result["is_valid"]:
            return f"Error: Could not identify template locations. {template_result['error']}"

        locations_and_context = template_result["locations_and_context"]

        # Step 2: Perform Grid Completion using LLM
        completed_grid = perform_grid_completion(question, locations_and_context)
        return completed_grid
    except Exception as e:
        return f"Error in solve_grid_transformation: {str(e)}"

def identify_locations_and_context(question):
    """Identifies key locations and context numbers from training examples."""
    system_instruction = "You are an expert at identifying key locations and context numbers in grid transformation problems."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and identify key locations and context numbers within the grid. These are locations and numbers that appear to be most related to the overall transformation. Output the locations and context numbers in a structured format.

    Example 1:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[2, 2, 2],
     [1, 1, 1],
     [2, 2, 2]]
    Locations and Context:
    Key Locations: Top row, Bottom row
    Context Number: 1 (the 'inner' number that doesn't change)

    Problem:
    {question}

    Locations and Context:
    """

    locations_and_context = call_llm(prompt, system_instruction)

    # Basic validation to ensure *something* was output
    if locations_and_context and locations_and_context.strip():
        return {"is_valid": True, "locations_and_context": locations_and_context, "error": None}
    else:
        return {"is_valid": False, "locations_and_context": None, "error": "Failed to identify locations and context."}

def perform_grid_completion(question, locations_and_context):
    """Completes the test input grid using the LLM based on learned patterns and identified template."""
    system_instruction = "You are an expert at completing grids based on learned patterns and identified locations and context."

    prompt = f"""
    Given the following grid transformation problem, the key locations, and context numbers, complete the test input grid. Apply the transformation patterns observed in the training examples and fill in any missing values. Ensure correct format.

    Example 1:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[2, 2, 2],
     [1, 1, 1],
     [2, 2, 2]]
    Locations and Context:
    Key Locations: Top row, Bottom row
    Context Number: 1

    Test Input:
    [[0, 0, 0],
     [3, 3, 3],
     [0, 0, 0]]

    Completed Grid:
    [[4, 4, 4],
     [3, 3, 3],
     [4, 4, 4]]

    Problem:
    {question}
    Locations and Context:
    {locations_and_context}
    Completed Grid:
    """

    completed_grid = call_llm(prompt, system_instruction)
    return completed_grid

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
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