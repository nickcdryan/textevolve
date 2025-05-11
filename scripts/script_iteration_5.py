import os
import re
import math

def main(question):
    """
    Transforms a grid based on patterns in training examples using LLM-driven spatial relationship identification and contextual pattern application.
    This approach uses explicit spatial context and value-based transformation reasoning.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem, max_attempts=3):
    """Solve grid transformation problems using pattern recognition, spatial analysis, and value-based rules."""

    # Hypothesis: Explicitly defining spatial context and value-based rules enhances the LLM's ability to learn and apply transformations.
    system_instruction = "You are an expert at grid transformation tasks. You analyze spatial relationships and apply value-based transformation rules."

    # Step 1: Extract the training examples and the test input grid.
    extraction_prompt = f"""
    Extract the training examples and the test input grid from the problem description.

    Example:
    Problem: Grid Transformation Task... Input Grid: [[1,2],[3,4]] ... Output Grid: [[5,6],[7,8]] ... TEST INPUT: [[9,10],[11,12]]
    Extracted:
    {{
      "examples": [
        "Input Grid: [[1,2],[3,4]] ... Output Grid: [[5,6],[7,8]]"
      ],
      "test_input": "[[9,10],[11,12]]"
    }}

    Problem: {problem}
    Extracted:
    """
    extracted_info = call_llm(extraction_prompt, system_instruction)
    print(f"Extracted Info: {extracted_info}")

    # Step 2: Infer value-based transformation rules with spatial context.
    rule_inference_prompt = f"""
    Analyze the training examples and infer value-based transformation rules, considering spatial context (e.g., adjacent cells).

    Example:
    Examples: Input Grid: [[1, 0], [0, 1]] ... Output Grid: [[2, 0], [0, 2]]
    Rule: If a cell has value 1, transform it to 2. Otherwise, maintain the original value.

    Examples: {extracted_info}
    Rule:
    """
    transformation_rule = call_llm(rule_inference_prompt, system_instruction)
    print(f"Transformation Rule: {transformation_rule}")

    # Step 3: Apply the transformation rule to the test input with spatial context.
    transformation_prompt = f"""
    Apply the following value-based transformation rule, considering spatial context, to the test input grid.

    Rule: {transformation_rule}
    Test Input Grid: {extracted_info}

    Example:
    Rule: If a cell has value 1, transform it to 2. Test Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[2, 0], [0, 2]]

    Transformed Grid:
    """
    transformed_grid = call_llm(transformation_prompt, system_instruction)
    print(f"Transformed Grid: {transformed_grid}")

    # Step 4: Verify the transformed grid by checking if the spatial context and value-based rules are followed
    verification_prompt = f"""
    Verify the transformed grid based on spatial context and value-based rules. Explain whether the rules are satisfied.

    Rule: {transformation_rule}
    Test Input Grid: {extracted_info}
    Transformed Grid: {transformed_grid}

    Example:
    Rule: If a cell has value 1, transform it to 2. Input Grid: [[1,0],[0,1]]. Output Grid: [[2,0],[0,2]]. Verification: The rule is followed.

    Verification:
    """
    verification_result = call_llm(verification_prompt, system_instruction)
    print(f"Verification Result: {verification_result}")

    if "The rule is followed." in verification_result:
        return transformed_grid
    else:
        return "Unable to transform the grid correctly."

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