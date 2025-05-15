import os
import re
import math

# EXPLORATION: Coordinate Transformation Rule Extraction with Local Contextual Validation
# HYPOTHESIS: We can improve grid transformation accuracy by extracting explicit coordinate-based transformation rules and then validating these rules based on the local context of each cell.
# This differs from previous attempts by focusing on explicit coordinate manipulation and local context validation, rather than broad visual feature analysis.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by extracting coordinate-based rules and validating them locally."""

    # 1. Extract Coordinate Transformation Rules
    rule_extraction_result = extract_coordinate_transformation_rules(question)
    if not rule_extraction_result["is_valid"]:
        return f"Error: Could not extract transformation rules. {rule_extraction_result['error']}"
    rules = rule_extraction_result["rules"]

    # 2. Apply Transformation with Local Context Validation
    transformed_grid = apply_transformation_with_validation(question, rules)
    return transformed_grid

def extract_coordinate_transformation_rules(question):
    """Extracts coordinate-based transformation rules from the training examples."""
    system_instruction = "You are an expert at extracting coordinate-based transformation rules from grid transformation problems."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and identify coordinate-based transformation rules.
    Focus on how the position of an element changes from the input grid to the output grid.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2],
     [3, 4]]
    Output Grid:
    [[4, 3],
     [2, 1]]
    Transformation Rules:
    - Element at (0, 0) moves to (1, 1)
    - Element at (0, 1) moves to (1, 0)
    - Element at (1, 0) moves to (0, 1)
    - Element at (1, 1) moves to (0, 0)

    Problem:
    {question}
    Transformation Rules:
    """

    rules = call_llm(prompt, system_instruction)

    # Validation: Ensure rules are present
    if rules and rules.strip():
        return {"is_valid": True, "rules": rules, "error": None}
    else:
        return {"is_valid": False, "rules": None, "error": "Failed to extract transformation rules."}

def apply_transformation_with_validation(question, rules):
    """Applies the transformation rules to the test input grid with local context validation."""
    system_instruction = "You are an expert at applying transformation rules to grids, validating each transformation based on local context."

    prompt = f"""
    Given the following grid transformation problem and coordinate-based transformation rules, apply the rules to the test input grid.
    Validate each transformation based on the local context (neighboring cells) of the target cell.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2],
     [3, 4]]
    Output Grid:
    [[4, 3],
     [2, 1]]
    Transformation Rules:
    - Element at (0, 0) moves to (1, 1)
    - Element at (0, 1) moves to (1, 0)
    - Element at (1, 0) moves to (0, 1)
    - Element at (1, 1) moves to (0, 0)
    Test Input:
    [[5, 6],
     [7, 8]]
    Completed Grid:
    [[8, 7],
     [6, 5]]

    Problem:
    {question}
    Transformation Rules: {rules}
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