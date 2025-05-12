import os
import re
import math

def main(question):
    """
    Transforms a grid based on patterns in training examples.
    Uses LLM-driven pattern extraction and refinement with LOCALIZED CONTEXTUAL ANALYSIS and EXPLICIT DIMENSION INFERENCE, with a dedicated verification step.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem, max_attempts=3):
    """Solve grid transformation problems by analyzing localized context and explicitly inferring output dimensions."""

    # Hypothesis: Combining localized contextual analysis with explicit dimension inference will significantly improve the LLM's ability to perform grid transformations.
    system_instruction = "You are an expert in grid transformation tasks. You are skilled at identifying localized patterns, understanding spatial relationships, and explicitly determining the output grid's dimensions."

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

    # Step 2: Infer the output grid dimensions and localized transformation rules, focusing on contextual analysis.
    dimension_inference_prompt = f"""
    Analyze the training examples and infer the localized transformation rules, *considering the surrounding context of each cell*, and the dimensions of the output grid.

    Example 1:
    Examples: Input Grid: [[1, 0], [0, 1]] ... Output Grid: [[2, 0], [0, 2]]
    Output Grid Dimensions: 2x2
    Localized Rule: If a cell has value 1, transform it to 2.

    Example 2:
    Examples: Input Grid: [[0, 1, 0]] ... Output Grid: [[0, 2, 0]]
    Output Grid Dimensions: 1x3
    Localized Rule: Change values of '1' to '2', but leave '0' unchanged.

    Examples: {extracted_info}
    Output Grid Dimensions:
    Localized Rule:
    """
    dimension_inference_result = call_llm(dimension_inference_prompt, system_instruction)
    print(f"Dimension Inference Result: {dimension_inference_result}")

    # Step 3: Extract the test input grid and Localized Rule extracted and apply transformation

    test_input_match = re.search(r'"test_input":\s*"(\[\[.*?\]\])"', extracted_info)
    if test_input_match:
        test_input_grid = test_input_match.group(1)
    else:
        test_input_grid = "[[0,0,0],[0,0,0],[0,0,0]]"  # Default grid if extraction fails

    localized_rule_match = re.search(r"Localized Rule:\s*(.*)", dimension_inference_result)
    if localized_rule_match:
        localized_rule = localized_rule_match.group(1)
    else:
        localized_rule = "No rule identified." # Default if extraction fails
        
    transformation_prompt = f"""
    Apply the following localized transformation rule to the test input grid.

    Rule: {localized_rule}
    Test Input Grid: {test_input_grid}
    Output Grid Dimensions: {dimension_inference_result}

    Example 1:
    Rule: Each element is doubled. Test Input Grid: [[1, 2], [3, 4]]. Output Grid Dimensions: 2x2
    Transformed Grid: [[2, 4], [6, 8]]

    Transformed Grid:
    """
    transformed_grid = call_llm(transformation_prompt, system_instruction)
    print(f"Transformed Grid: {transformed_grid}")

    # Step 4: Verify the transformed grid.
    verification_prompt = f"""
    Verify that the transformed grid follows the localized transformation rule AND has the correct dimensions.

    Rule: {localized_rule}
    Test Input Grid: {test_input_grid}
    Transformed Grid: {transformed_grid}

    Example:
    Rule: double each number. Input: [[1,2],[3,4]]. Output: [[2,4],[6,8]]. Verification: CORRECT

    Verification: Is the transformation rule followed, AND are the output grid dimensions correct?
    """
    verification_result = call_llm(verification_prompt, system_instruction)

    if "INCORRECT" not in verification_result:
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