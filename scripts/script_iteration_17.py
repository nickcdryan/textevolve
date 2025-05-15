import os
import re
import math

# HYPOTHESIS: Instead of trying to explicitly define and apply transformation rules,
# let's see if the LLM can directly generate a *transformation script* (in natural language)
# that, when followed, produces the correct output. This approach prioritizes
# the LLM's ability to sequence actions based on pattern recognition. We'll use
# a strong validation step to ensure the transformation script is reasonable.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by generating and following a transformation script."""

    # Step 1: Generate a Transformation Script
    transformation_script_result = generate_transformation_script(question, max_attempts=max_attempts)
    if not transformation_script_result["is_valid"]:
        return f"Error: Could not generate a valid transformation script. {transformation_script_result['error']}"

    transformation_script = transformation_script_result["transformation_script"]

    # Step 2: Follow the Transformation Script
    transformed_grid = follow_transformation_script(question, transformation_script)
    return transformed_grid

def generate_transformation_script(question, max_attempts=3):
    """Generates a transformation script (natural language) describing how to transform the input grid."""
    system_instruction = "You are an expert at generating clear, step-by-step transformation scripts for grid problems."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and generate a detailed, step-by-step script
    that describes how to transform the input grid into the output grid. The script should be written in natural language and be easy to follow.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2, 3], [4, 5, 6]]
    Output Grid:
    [[6, 5, 4], [3, 2, 1]]
    Transformation Script:
    1. Reverse each row in the input grid.
    2. Reverse the order of the rows themselves.

    Problem:
    {question}

    Transformation Script:
    """

    transformation_script = call_llm(prompt, system_instruction)

    # Verification Step: Ensure the script is reasonable and not nonsensical
    verification_prompt = f"""
    Verify that the given transformation script is clear, concise, and describes a reasonable transformation.
    Transformation Script: {transformation_script}
    Is the script valid? (VALID/INVALID)
    """
    validation_result = call_llm(verification_prompt)

    if "VALID" in validation_result:
        return {"is_valid": True, "transformation_script": transformation_script, "error": None}
    else:
        return {"is_valid": False, "transformation_script": None, "error": "Invalid transformation script."}

def follow_transformation_script(question, transformation_script):
    """Follows the transformation script to transform the test input grid."""
    system_instruction = "You are an expert at following transformation scripts to transform grids."

    prompt = f"""
    Given the following grid transformation problem and the transformation script, follow the script to transform the input grid into the output grid.

    Problem: {question}
    Transformation Script: {transformation_script}

    Example:
    Problem: Input Grid: [[1, 2], [3, 4]] Transformation Script: Reverse each row. Then reverse the order of rows.
    Output Grid: [[4, 3], [2, 1]]

    Generate the output grid based on the transformation script.
    """
    output_grid = call_llm(prompt, system_instruction)
    return output_grid

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