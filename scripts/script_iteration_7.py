import os
import re
import math

# Hypothesis: This exploration will focus on a "rule generation and application" approach. The LLM will first attempt to
# explicitly *generate* a symbolic rule or a set of rules from the training examples, then *apply* this rule to the test input.
# This contrasts with previous approaches that attempt to either directly transform the grid or classify the transformation type.
# The hypothesis is that explicitly formulating a symbolic representation of the rule will improve generalization and robustness.
# Also, by attempting to perform verifications after each step of rule generation and applying to transformations, it will be
# easier to identify where and why the system is breaking. We also implement multiple examples in every single LLM prompt.

def main(question):
    """Transforms a grid based on explicit rule generation and application."""
    try:
        # 1. Generate transformation rule from examples
        rule_generation_result = generate_transformation_rule(question)
        if not rule_generation_result["is_valid"]:
            return f"Error: Rule generation failed - {rule_generation_result['feedback']}"

        transformation_rule = rule_generation_result["transformation_rule"]

        # 2. Apply transformation rule to test input
        transformed_grid_result = apply_transformation_rule(question, transformation_rule)
        if not transformed_grid_result["is_valid"]:
            return f"Error: Transformation failed - {transformed_grid_result['feedback']}"

        transformed_grid = transformed_grid_result["transformed_grid"]

        return transformed_grid
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def generate_transformation_rule(question, max_attempts=3):
    """Generates a transformation rule from the training examples."""
    system_instruction = "You are an expert in identifying and formulating transformation rules from grid examples."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert in identifying and formulating transformation rules from grid examples.
        Given a question containing training examples, identify the transformation rule that maps the input grid to the output grid.
        The transformation rule should be expressed in a symbolic, human-readable form.

        Example 1:
        Input Grid: [[1, 2], [3, 4]]
        Output Grid: [[2, 3], [4, 5]]
        Transformation Rule: Add 1 to each element in the grid.

        Example 2:
        Input Grid: [[1, 2], [3, 4]]
        Output Grid: [[2, 1], [4, 3]]
        Transformation Rule: Mirror the grid horizontally.

        Example 3:
        Input Grid: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        Output Grid: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        Transformation Rule: Swap the value to the opposing number

        Now, for this new question, generate the transformation rule:
        {question}
        """
        response = call_llm(prompt, system_instruction)

        # Verification step: check if the response is a valid transformation rule
        verification_result = verify_rule_format(question, response)
        if verification_result["is_valid"]:
            return {"is_valid": True, "transformation_rule": response}
        else:
            print(f"Rule generation failed (attempt {attempt+1}/{max_attempts}): {verification_result['feedback']}")

    return {"is_valid": False, "feedback": "Failed to generate a valid transformation rule after multiple attempts."}

def apply_transformation_rule(question, transformation_rule, max_attempts=3):
    """Applies the transformation rule to the test input."""
    system_instruction = "You are an expert in applying transformation rules to grid inputs."

    for attempt in range(max_attempts):
        prompt = f"""
        You are an expert in applying transformation rules to grid inputs.
        Given a question containing a test input and a transformation rule, apply the rule to the input and generate the transformed grid.
        The transformed grid should be returned in string representation that begins with '[[' and ends with ']]'.

        Example 1:
        Input Grid: [[5, 6], [7, 8]]
        Transformation Rule: Add 1 to each element in the grid.
        Transformed Grid: [[6, 7], [8, 9]]

        Example 2:
        Input Grid: [[5, 6], [7, 8]]
        Transformation Rule: Mirror the grid horizontally.
        Transformed Grid: [[6, 5], [8, 7]]

        Example 3:
        Input Grid: [[1, 0], [0, 1]]
        Transformation Rule: Swap the value to the opposing number
        Transformed Grid: [[0, 1], [1, 0]]

        Now, for this new question, apply the transformation rule:
        {question}
        Transformation Rule: {transformation_rule}
        Transformed Grid:
        """
        transformed_grid = call_llm(prompt, system_instruction)

        # Verification step: check if the output is a valid grid
        verification_result = verify_grid_format(question, transformed_grid)
        if verification_result["is_valid"]:
            return {"is_valid": True, "transformed_grid": transformed_grid}
        else:
            print(f"Transformation failed (attempt {attempt+1}/{max_attempts}): {verification_result['feedback']}")
    return {"is_valid": False, "feedback": "Failed to transform the grid correctly after multiple attempts."}

def verify_rule_format(question, rule):
    """Verifies that the rule is in the proper format."""
    #Implement a more thorough approach to validating and scoring code - NOT USED
    if not isinstance(rule, str):
        return {"is_valid": False, "feedback": "Rule is not a string."}
    if len(rule) == 0:
        return {"is_valid": False, "feedback": "Rule is empty string."}

    #Add more logic here
    return {"is_valid": True}

def verify_grid_format(question, transformed_grid):
    """Verifies that the transformed grid is in the proper format."""
    try:
        if not (transformed_grid.startswith("[[") and transformed_grid.endswith("]]")):
            return {"is_valid": False, "feedback": "Output should start with '[[' and end with ']]'."}

        # Basic check for grid structure
        grid_rows = transformed_grid.strip("[]").split("],[")
        if not all("," in row for row in grid_rows):
            return {"is_valid": False, "feedback": "Rows are not comma separated."}

        return {"is_valid": True}
    except Exception as e:
        return {"is_valid": False, "feedback": f"Error during grid validation: {str(e)}"}

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