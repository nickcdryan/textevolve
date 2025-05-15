import os
import re
import math

# EXPLORATION: LLM-Guided Template Matching with Iterative Contextual Refinement
# HYPOTHESIS: We can improve grid transformation by having the LLM identify similar training grids, then use these matched training templates for localized context refinement of the target input grid transformations.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by matching templates and refining transformation based on context."""

    # Step 1: Match template
    template_matching_result = match_template(question)
    if not template_matching_result["is_valid"]:
        return f"Error: Could not match template. {template_matching_result['error']}"

    training_example = template_matching_result["training_example"]

    # Step 2: Refine transformation
    refined_transformation = refine_transformation(question, training_example)

    return refined_transformation

def match_template(question):
    """Matches the input grid to the most similar training grid."""
    system_instruction = "You are an expert at matching grid transformation input grids to the most similar training example input grids."

    prompt = f"""
    Given the following grid transformation problem, analyze the test input grid and identify the most similar input grid from the training examples. Return the entire matching training example (input and output grids).

    Example:
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
    === TEST INPUT ===
    [[0, 0, 0],
     [5, 5, 5],
     [0, 0, 0]]
    Most similar Training Example:
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[2, 2, 2],
     [1, 1, 1],
     [2, 2, 2]]
    

    Problem:
    {question}
    Most similar Training Example:
    """

    training_example = call_llm(prompt, system_instruction)

    # Validation: Check if a training example was returned
    if training_example and training_example.strip():
        return {"is_valid": True, "training_example": training_example, "error": None}
    else:
        return {"is_valid": False, "training_example": None, "error": "Failed to identify similar training example."}

def refine_transformation(question, training_example):
    """Refines the transformation based on localized context using the matched training example."""
    system_instruction = "You are an expert at refining grid transformations based on context from matched training examples."

    prompt = f"""
    Given the following grid transformation problem and the most similar training example, refine the transformation based on localized context.

    Example:
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
    === TEST INPUT ===
    [[0, 0, 0],
     [5, 5, 5],
     [0, 0, 0]]
    Most similar Training Example:
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[2, 2, 2],
     [1, 1, 1],
     [2, 2, 2]]
    Refined Transformation:
    [[2, 2, 2],
     [5, 5, 5],
     [2, 2, 2]]

    Problem:
    {question}
    Most similar Training Example: {training_example}
    Refined Transformation:
    """

    refined_transformation = call_llm(prompt, system_instruction)
    return refined_transformation

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