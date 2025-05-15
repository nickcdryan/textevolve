import os
import re
import math

# EXPLORATION: LLM-Driven Transformation with Pattern Decomposition, Component Analysis, and Rule-Based Transformation
# HYPOTHESIS: We can improve grid transformation accuracy by decomposing the problem into (1) pattern identification, (2) analysis of transformation components (source region, target region, transformation operator), (3) applying a rule-based transformation, and incorporating examples throughout.
# This approach differs from previous attempts by focusing on analyzing the transformation components and creating an intermediate rule-based representation. It also incorporates verification steps throughout.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems by encoding visual patterns, breaking them down into components, deriving a rule, and applying it."""
    try:
        # 1. Analyze Visual Features for Pattern Decomposition and Component Identification
        feature_analysis_result = analyze_visual_features(question)
        if not feature_analysis_result["is_valid"]:
            return f"Error: Could not identify transformation components. {feature_analysis_result['error']}"
        transformation_rule = feature_analysis_result["transformation_rule"]

        # 2. Apply Transformation based on Rule with Validation
        transformed_grid = apply_transformation(question, transformation_rule)
        return transformed_grid

    except Exception as e:
        return f"Error in solve_grid_transformation: {str(e)}"

def analyze_visual_features(question):
    """Analyzes visual features, identifies transformation components, and derives a rule from the training examples."""
    system_instruction = "You are an expert at analyzing visual patterns in grid transformation problems, identifying source and target regions, and deriving a rule based on the identified components."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and identify the key components of the transformation. Specifically, identify the source region(s), the target region(s), and the transformation operation that maps the source to the target.
    Express the transformation as a rule that can be applied to the test input.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[1, 1, 1],
     [0, 0, 0],
     [1, 1, 1]]
    Transformation Rule:
    "The row with 1s swaps positions with the row above and below it. Values in new rows become 1."

    Problem:
    {question}
    Transformation Rule:
    """

    transformation_rule = call_llm(prompt, system_instruction)

    # Verification: Ensure a rule is present
    if transformation_rule and transformation_rule.strip():
        return {"is_valid": True, "transformation_rule": transformation_rule, "error": None}
    else:
        return {"is_valid": False, "transformation_rule": None, "error": "Failed to derive a transformation rule."}

def apply_transformation(question, transformation_rule):
    """Applies the transformation rule to the test input grid with validation."""
    system_instruction = "You are an expert at applying a stated transformation rule to a grid, based on analysis from training examples."

    prompt = f"""
    Given the following grid transformation problem and transformation rule, apply the rule to the test input grid. Ensure the generated grid consistently reflects the rule.

    Problem:
    {question}
    Transformation Rule: {transformation_rule}

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[1, 1, 1],
     [0, 0, 0],
     [1, 1, 1]]
    Transformation Rule: "The row with 1s swaps positions with the row above and below it. Values in new rows become 1."
    Test Input:
    [[0, 0, 0],
     [2, 2, 2],
     [0, 0, 0]]
    Completed Grid:
    [[2, 2, 2],
     [0, 0, 0],
     [2, 2, 2]]

    Based on this problem, transformation and example, apply the identified transformation rule to the test input grid to generate the transformed grid, and follow all instructions. What is the completed grid?
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