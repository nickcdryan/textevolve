import os
import re
import math

# EXPLORATION: Pattern-Based Region Swapping with LLM Guided Template Completion and Verification
# HYPOTHESIS: We can solve grid transformation problems by having the LLM identify specific regions within the grid, infer a swapping pattern between these regions, and then complete a template based on this pattern, validating each step. This will reduce the complexity of the LLM reasoning by creating a structured process.

def solve_grid_transformation(question):
    """Solves grid transformation problems by identifying regions and swapping patterns."""
    # Step 1: Identify Key Regions
    region_identification_result = identify_key_regions(question)
    if not region_identification_result["is_valid"]:
        return f"Error: Could not identify key regions. {region_identification_result['error']}"

    regions = region_identification_result["regions"]

    # Step 2: Infer Swapping Pattern
    swapping_pattern_result = infer_swapping_pattern(question, regions)
    if not swapping_pattern_result["is_valid"]:
        return f"Error: Could not infer swapping pattern. {swapping_pattern_result['error']}"

    swapping_pattern = swapping_pattern_result["swapping_pattern"]

    # Step 3: Complete Template
    completed_grid = complete_template(question, regions, swapping_pattern)
    return completed_grid

def identify_key_regions(question):
    """Identifies key regions (e.g., corners, rows, columns) in the grid."""
    system_instruction = "You are an expert at identifying key regions in grid transformation problems."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and identify key regions within the grid. Key regions are parts of the grid (e.g., corners, rows, columns) that appear to be involved in the transformation.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
    Output Grid:
    [[7, 8, 9],
     [4, 5, 6],
     [1, 2, 3]]
    Key Regions: Top row, Bottom row

    Problem:
    {question}
    Key Regions:
    """

    regions = call_llm(prompt, system_instruction)

    # Validation: Ensure that *something* was output and avoid empty outputs
    if regions and regions.strip():
        return {"is_valid": True, "regions": regions, "error": None}
    else:
        return {"is_valid": False, "regions": None, "error": "Failed to identify key regions."}

def infer_swapping_pattern(question, regions):
    """Infers the swapping pattern between the identified regions."""
    system_instruction = "You are an expert at inferring swapping patterns between grid regions."

    prompt = f"""
    Given the following grid transformation problem and the identified key regions, infer the swapping pattern between these regions. The swapping pattern describes how the values in these regions are exchanged or rearranged.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
    Output Grid:
    [[7, 8, 9],
     [4, 5, 6],
     [1, 2, 3]]
    Key Regions: Top row, Bottom row
    Swapping Pattern: Top row and bottom row are swapped.

    Problem:
    {question}
    Key Regions: {regions}
    Swapping Pattern:
    """

    swapping_pattern = call_llm(prompt, system_instruction)

    # Validation: Check if the swapping patter is not empty and a sensible statement
    if swapping_pattern and swapping_pattern.strip():
        return {"is_valid": True, "swapping_pattern": swapping_pattern, "error": None}
    else:
        return {"is_valid": False, "swapping_pattern": None, "error": "Failed to infer swapping pattern."}

def complete_template(question, regions, swapping_pattern):
    """Completes the grid template based on the inferred swapping pattern."""
    system_instruction = "You are an expert at completing grid templates based on identified regions and swapping patterns."

    prompt = f"""
    Given the following grid transformation problem, identified key regions, and inferred swapping pattern, complete the test input grid based on this pattern. Ensure the output is a list of lists.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
    Output Grid:
    [[7, 8, 9],
     [4, 5, 6],
     [1, 2, 3]]
    Key Regions: Top row, Bottom row
    Swapping Pattern: Top row and bottom row are swapped.
    Test Input:
    [[10, 11, 12],
     [13, 14, 15],
     [16, 17, 18]]
    Completed Grid:
    [[16, 17, 18],
     [13, 14, 15],
     [10, 11, 12]]

    Problem:
    {question}
    Key Regions: {regions}
    Swapping Pattern: {swapping_pattern}
    Completed Grid:
    """

    completed_grid = call_llm(prompt, system_instruction)
    return completed_grid

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