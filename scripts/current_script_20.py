import os
import re
import math

# EXPLORATION: Explicit Coordinate-Based Transformation with Contextual Awareness
# HYPOTHESIS: By prompting the LLM to generate transformation rules based on explicit coordinates and contextual awareness (surrounding values), 
# and including the output of each major processing state, we can create more robust general transformations.
# In addition to this main objective, the location of the LLM failures and code execution will be tracked using print statements.

def solve_grid_transformation(question):
    """Solves grid transformation problems by analyzing and applying coordinate-based transformations."""

    # Step 1: Analyze visual features and generate transformation rules based on coordinates and context
    analysis_result = analyze_grid_transformation(question)
    if not analysis_result["is_valid"]:
        return f"Error: Could not analyze the transformation. {analysis_result['error']}"

    # Step 2: Apply coordinate-based transformations
    transformed_grid = apply_coordinate_transformation(question, analysis_result["transformation_rules"])
    return transformed_grid

def analyze_grid_transformation(question):
    """Analyzes visual features and generates transformation rules based on coordinates and context."""
    system_instruction = "You are an expert at analyzing visual features of grid transformations, focusing on coordinate-based rules."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and generate transformation rules that are based on explicit coordinates and context (surrounding values). Output the transformations in a coordinate-based style. Show each rule.
    
    Example 1:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
    Output Grid:
    [[9, 8, 7],
     [6, 5, 4],
     [3, 2, 1]]
    Transformation Rules:
    Rule 1: value[0,0] becomes value[2,2]
    Rule 2: value[0,1] becomes value[2,1]
    Rule 3: value[0,2] becomes value[2,0]
    Rule 4: value[1,0] becomes value[1,2]
    Rule 5: value[1,1] remains value[1,1]
    ... and so on.

    Problem:
    {question}

    Transformation Rules:
    """

    transformation_rules = call_llm(prompt, system_instruction)

    # Validation Step: Ensure the rules are non-empty and coordinate-based
    if transformation_rules and transformation_rules.strip():
        return {"is_valid": True, "transformation_rules": transformation_rules, "error": None}
    else:
        return {"is_valid": False, "transformation_rules": None, "error": "Failed to generate transformation rules."}

def apply_coordinate_transformation(question, transformation_rules):
    """Applies the transformation rules to the test input grid."""
    system_instruction = "You are an expert at applying transformation rules based on coordinates."

    prompt = f"""
    Given the following grid transformation problem and transformation rules, apply the rules to the test input grid. Only output the transformed grid.
    
    Example 1:
    Problem:
    Input Grid:
    [[1, 2],
     [3, 4]]
    Transformation Rules:
    Rule 1: value[0,0] becomes value[1,1]
    Rule 2: value[0,1] becomes value[1,0]
    Rule 3: value[1,0] becomes value[0,1]
    Rule 4: value[1,1] becomes value[0,0]
    Output Grid:
    [[4, 3],
     [2, 1]]

    Problem:
    {question}
    Transformation Rules:
    {transformation_rules}
    Output Grid:
    """

    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

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